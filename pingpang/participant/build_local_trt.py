"""Build a separate Local engine; validate actual ROI tensors and benchmark."""
from pathlib import Path
import copy,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'pingpang')]
import torch,onnx,tensorrt as trt
from core.types import RunContext
from participant.solution_public_cached import Solution
from participant.public_trt_backend import TensorRTEngine
from participant.build_public_trt import build,benchmark,save
from ttnet.models.high_resolution import export_stable_norms


def main():
    dest=Path(sys.argv[1]);engines=dest/'engines';engines.mkdir(exist_ok=True)
    state=dest/'build-state.json';save(state,dict(phase='loading'))
    samples=torch.load(dest/'local_inputs.pt',map_location='cpu',weights_only=False)
    solution=Solution();ctx=RunContext('',str(dest),'gpu');solution.prepare(ctx)
    reference=solution.system.local_model
    exported=copy.deepcopy(reference)
    export_stable_norms(exported)
    exported.export_pool_kernel=(solution.config['local_input'][1]//4,solution.config['local_input'][0]//4)
    exported.cuda().eval()
    example=tuple(x.cuda() for x in samples[0]['inputs'])
    with torch.inference_mode(),torch.autocast('cuda',enabled=True):
        expected=reference(*example)
    dtypes=[str(x.dtype).split('.')[-1] for x in expected]
    names=['images','frame_valid','time_offsets_sec'];outputs=['heatmap','descriptor']
    modelpath=engines/'local.onnx';enginepath=engines/'local.engine'
    if enginepath.exists():raise FileExistsError(enginepath)
    save(state,dict(phase='exporting'))
    with torch.inference_mode():
        torch.onnx.export(exported,example,str(modelpath),input_names=names,output_names=outputs,opset_version=17,do_constant_folding=True)
    onnx.checker.check_model(str(modelpath))
    save(state,dict(phase='building'))
    built=build(modelpath,enginepath,'global','fp16')
    save(engines/'local_build.json',built)
    runtime=TensorRTEngine(enginepath,names,outputs)
    def deployed(*args):return tuple(x.to(getattr(torch,dtype)) for x,dtype in zip(runtime(*args),dtypes))
    rows=[];bench_args=None
    save(state,dict(phase='numeric_validation'))
    with torch.inference_mode(),torch.autocast('cuda',enabled=True):
        for sample in samples:
            args=tuple(x.cuda() for x in sample['inputs'])
            expected=reference(*args);actual=deployed(*args);values={}
            for name,ref,got in zip(outputs,expected,actual):
                assert ref.shape==got.shape
                ref=ref.float();got=got.float()
                assert torch.isfinite(ref).all() and torch.isfinite(got).all()
                delta=(ref-got).abs()
                values[name]=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),relative_l2=float(torch.linalg.vector_norm(delta)/(torch.linalg.vector_norm(ref)+1e-12)))
                if name=='heatmap':values[name]['argmax_same']=int(ref.argmax())==int(got.argmax())
            rows.append(dict(video_id=sample['video_id'],frame_id=sample['frame_id'],call_index=sample['call_index'],outputs=values))
            if bench_args is None and sample['call_index']==20:bench_args=args
    save(engines/'local_numeric.json',rows)
    save(state,dict(phase='benchmarking'))
    benches=dict(pytorch_amp=benchmark(reference,bench_args),tensorrt=benchmark(deployed,bench_args))
    save(engines/'local_benchmark.json',benches)
    manifest=dict(checkpoint_sha256=ctx.info['checkpoint_sha256'],tensorrt=trt.__version__,gpu=torch.cuda.get_device_name(),
                  local=dict(engine=enginepath.name,inputs=names,outputs=outputs,output_dtypes=dtypes,precision='FP16 with FP32 normalization',build=built))
    save(engines/'manifest.json',manifest)
    summary=dict(phase='completed',samples=len(rows),build_seconds=built['seconds'],p50_pytorch=benches['pytorch_amp']['p50_ms'],p50_trt=benches['tensorrt']['p50_ms'],
                 argmax_same=sum(x['outputs']['heatmap']['argmax_same'] for x in rows),max_errors={n:max(x['outputs'][n]['max_abs'] for x in rows) for n in outputs})
    save(state,summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
