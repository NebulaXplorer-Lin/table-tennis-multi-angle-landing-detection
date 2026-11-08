"""Export/build and numerically validate one stage, using real unlabeled inputs."""
from pathlib import Path
import argparse,copy,json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'pingpang')]
import cv2,numpy as np,torch,onnx,tensorrt as trt
from core.types import RunContext
from participant.solution_public_cached import Solution
from participant.public_trt_backend import TensorRTEngine
from ttnet.data_tools.geometry import letterbox
from ttnet.training.dataset import normalize


class TableExport(torch.nn.Module):
    def __init__(self,model):
        super().__init__(); self.model=model
    def forward(self,x):
        prediction,spatial=self.model(x)
        return (prediction[0] if isinstance(prediction,tuple) else prediction),spatial


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def samples(root,config,stage):
    videos=json.loads((root/'manifest.json').read_text(encoding='utf-8'))['videos']
    for v in videos:
        capture=cv2.VideoCapture(str(root/v['path']))
        try:
            for center in (0,1,20,v['n_frames']-1):
                images=[];valid=[];offsets=[]
                ids=range(center-1,center+2) if stage=='global' else [center]
                for fid in ids:
                    ok=0<=fid<v['n_frames']
                    valid.append(ok);offsets.append((fid-center)/v['fps'] if ok else 0.)
                    if ok:
                        capture.set(cv2.CAP_PROP_POS_FRAMES,fid)
                        success,image=capture.read()
                        if not success:raise RuntimeError((v['video_id'],fid))
                        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    if stage=='global':
                        w,h=config['global_input']
                        image=letterbox(image,w,h)[0] if ok else np.zeros((h,w,3),np.uint8)
                        images.append(normalize(image))
                    else:
                        image=letterbox(image,*config['table_frontend']['input'],padding_value=114)[0]
                        images.append(image.transpose(2,0,1).copy().astype(np.float32)/255.)
                x=torch.from_numpy(np.stack(images)).cuda()
                args=(x[None],torch.tensor([valid],device='cuda'),torch.tensor([offsets],device='cuda')) if stage=='global' else (x,)
                yield dict(video_id=v['video_id'],frame_id=center),args
        finally:capture.release()


def build(onnx_path,engine_path,stage,precision):
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,2<<30)
    config.clear_flag(trt.BuilderFlag.TF32)
    protected=[]
    if precision=='fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        if stage=='global':
            # Every exported GroupNorm uses explicit reductions. Keep its whole
            # arithmetic scope in FP32, including squared deviations.
            model=onnx.load(str(onnx_path))
            scopes={n.name.rsplit('/',1)[0]+'/' for n in model.graph.node if n.op_type=='ReduceMean' and '/' in n.name.strip('/')}
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            kinds={trt.LayerType.ELEMENTWISE,trt.LayerType.REDUCE,trt.LayerType.UNARY,trt.LayerType.SCALE,trt.LayerType.ACTIVATION}
            for i in range(network.num_layers):
                layer=network.get_layer(i)
                if layer.type in kinds and (layer.type==trt.LayerType.REDUCE or any(layer.name.startswith(p) for p in scopes)):
                    layer.precision=trt.float32
                    for j in range(layer.num_outputs):layer.set_output_type(j,trt.float32)
                    protected.append(layer.name)
    started=time.perf_counter()
    engine=builder.build_serialized_network(network,config)
    if engine is None:raise RuntimeError('TensorRT engine build failed')
    engine_path.write_bytes(bytes(engine))
    return dict(seconds=time.perf_counter()-started,protected_fp32_layers=protected)


def benchmark(fn,args):
    with torch.inference_mode(),torch.autocast('cuda',enabled=True):
        for _ in range(50):fn(*args)
        values=[]
        for _ in range(3):
            for _ in range(200):
                torch.cuda.synchronize();start=time.perf_counter()
                fn(*args);torch.cuda.synchronize();values.append((time.perf_counter()-start)*1000)
    return dict(p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)),mean_ms=float(np.mean(values)),samples_ms=values)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['global','table'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--precision',choices=['fp16','fp32'],default='fp16');ap.add_argument('--resume-engine',action='store_true');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    state_path=a.output/(a.stage+'_state.json')
    save(state_path,dict(status='loading'))
    solution=Solution();ctx=RunContext(str(a.input),str(a.output),'gpu');solution.prepare(ctx)
    model=solution.system.global_model if a.stage=='global' else TableExport(solution.system.table_encoder)
    iterator=samples(a.input,solution.config,a.stage);_,example=next(iterator);iterator.close()
    with torch.inference_mode(),torch.autocast('cuda',enabled=True):reference=model(*example)
    output_dtypes=[str(x.dtype).split('.')[-1] for x in reference]
    onnx_path=a.output/(a.stage+'.onnx');engine_path=a.output/(a.stage+'.engine')
    if engine_path.exists() and not a.resume_engine:raise FileExistsError(engine_path)
    names=['images','frame_valid','time_offsets_sec'] if a.stage=='global' else ['images']
    outputs=['heatmap','descriptor'] if a.stage=='global' else ['detections','spatial']
    if a.resume_engine:
        onnx.checker.check_model(str(onnx_path))
        built=json.loads((a.output/(a.stage+'_build.json')).read_text(encoding='utf-8')) if (a.output/(a.stage+'_build.json')).exists() else dict(seconds=None,resumed_after_validation_failure=True)
    else:
        save(state_path,dict(status='exporting'))
        with torch.inference_mode():
            torch.onnx.export(model,example,str(onnx_path),input_names=names,output_names=outputs,
                              opset_version=17,do_constant_folding=True)
        onnx.checker.check_model(str(onnx_path))
        save(state_path,dict(status='building'))
        built=build(onnx_path,engine_path,a.stage,a.precision)
        save(a.output/(a.stage+'_build.json'),built)
    runtime=TensorRTEngine(engine_path,names,outputs)
    assert runtime.inputs==names and runtime.outputs==outputs,(runtime.inputs,runtime.outputs)
    def deployed(*args):return tuple(x.to(getattr(torch,t)) for x,t in zip(runtime(*args),output_dtypes))
    save(state_path,dict(status='numeric_validation'))
    rows=[];bench_args=None
    with torch.inference_mode(),torch.autocast('cuda',enabled=True):
        for identity,args in samples(a.input,solution.config,a.stage):
            expected=model(*args);actual=deployed(*args)
            values={}
            for name,ref,got in zip(outputs,expected,actual):
                assert ref.shape==got.shape
                ref=ref.float();got=got.float()
                assert torch.isfinite(ref).all() and torch.isfinite(got).all(),(identity,name)
                delta=(ref-got).abs()
                values[name]=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                                 relative_l2=float(torch.linalg.vector_norm(delta)/(torch.linalg.vector_norm(ref)+1e-12)))
                if name=='heatmap':values[name]['argmax_same']=int(ref.argmax())==int(got.argmax())
            rows.append(dict(**identity,outputs=values))
            if bench_args is None and identity['frame_id']==20:bench_args=args
    save(a.output/(a.stage+'_numeric.json'),rows)
    save(state_path,dict(status='benchmarking'))
    benchmarks=dict(pytorch_amp=benchmark(model,bench_args),tensorrt=benchmark(deployed,bench_args))
    save(a.output/(a.stage+'_benchmark.json'),benchmarks)
    manifest_path=a.output/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else dict(checkpoint_sha256=ctx.info['checkpoint_sha256'],
                         tensorrt=trt.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name())
    assert manifest['checkpoint_sha256']==ctx.info['checkpoint_sha256']
    manifest[a.stage]=dict(engine=engine_path.name,inputs=names,outputs=outputs,output_dtypes=output_dtypes,precision=a.precision,build=built)
    save(manifest_path,manifest)
    summary=dict(status='completed',stage=a.stage,precision=a.precision,samples=len(rows),build_seconds=built['seconds'],
                 p50_pytorch=benchmarks['pytorch_amp']['p50_ms'],p50_trt=benchmarks['tensorrt']['p50_ms'],
                 max_errors={name:max(row['outputs'][name]['max_abs'] for row in rows) for name in outputs})
    save(state_path,summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
