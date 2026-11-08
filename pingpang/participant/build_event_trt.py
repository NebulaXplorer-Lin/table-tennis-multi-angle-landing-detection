"""Export complete TableFusionEventHead and protect explicit LayerNorm arithmetic."""
from pathlib import Path
import sys,copy,json,inspect,textwrap,types
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'pingpang')]
import torch,onnx,tensorrt as trt
from core.types import RunContext
from participant.solution_public_trt_local import Solution
from participant.public_trt_backend import TensorRTEngine
from participant.build_public_trt import build,benchmark,save

NAMES=['features','motion','frame_valid','roi','context','roi_valid','context_valid']
def typed_where(condition,a,b):
    if not torch.is_tensor(b):b=torch.full_like(a,b)
    return torch.where(condition,a,b)

def export_where(model):
    original=model.forward.__func__
    source=textwrap.dedent(inspect.getsource(original)).replace('torch.where(', 'typed_where(')
    namespace=dict(original.__globals__,typed_where=typed_where)
    exec(compile(source,'<export-only typed Where>','exec'),namespace)
    model.forward=types.MethodType(namespace['forward'],model)
class ExportNorm(torch.nn.LayerNorm):
    def forward(self,x):
        if not torch.onnx.is_in_onnx_export():return super().forward(x)
        y=x.float();mean=y.mean(-1,keepdim=True);centered=y-mean
        y=centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+self.eps)
        return (y*self.weight.float()+self.bias.float()).to(x.dtype)

def norms(m):
    for name,child in list(m.named_children()):
        if isinstance(child,torch.nn.LayerNorm):
            new=ExportNorm(child.normalized_shape,eps=child.eps);new.load_state_dict(child.state_dict());setattr(m,name,new)
        else:norms(child)

class FlatHead(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,features,motion,frame_valid,roi,context,roi_valid,context_valid):
        return self.model(features,motion,frame_valid,dict(roi=roi,context=context,roi_valid=roi_valid,context_valid=context_valid))

if __name__=='__main__':
    dest=Path(sys.argv[1]);folder=dest/'engines';folder.mkdir(exist_ok=True)
    state=dest/'build_state.json';save(state,dict(phase='loading'))
    samples=torch.load(dest/'event_inputs.pt',map_location='cpu',weights_only=False)
    solution=Solution();ctx=RunContext('',str(dest),'gpu');solution.prepare(ctx)
    reference=FlatHead(solution.system.head).eval();exported=copy.deepcopy(reference);norms(exported);export_where(exported.model);exported.cuda().eval()
    example=tuple(x.cuda() for x in samples[0]['inputs'])
    with torch.inference_mode(),torch.autocast('cuda',enabled=True):dtype=reference(*example).dtype
    path=folder/'event.onnx';engine=folder/'event.engine';assert not engine.exists()
    save(state,dict(phase='exporting'))
    with torch.inference_mode():torch.onnx.export(exported,example,str(path),input_names=NAMES,output_names=['logits'],opset_version=17,do_constant_folding=True)
    onnx.checker.check_model(str(path),full_check=True);save(state,dict(phase='building'))
    built=build(path,engine,'global','fp16');save(folder/'event_build.json',built)
    runtime=TensorRTEngine(engine,NAMES,['logits'])
    def deployed(*args):return runtime(*args)[0].to(dtype)
    rows=[];save(state,dict(phase='validating'))
    with torch.inference_mode(),torch.autocast('cuda',enabled=True):
        for sample in samples:
            args=tuple(x.cuda() for x in sample['inputs']);ref=reference(*args).float();got=deployed(*args).float()
            assert ref.shape==got.shape and torch.isfinite(got).all()
            a=ref.sigmoid();b=got.sigmoid()
            rows.append(dict(video_id=sample['video_id'],frame_id=sample['frame_id'],logits_max_abs=float((ref-got).abs().max()),
                probability_max_abs=float((a-b).abs().max()),original_probabilities=a[0].tolist(),new_probabilities=b[0].tolist(),
                bounce_threshold_flip=bool((a[0,0]>=solution.threshold)!=(b[0,0]>=solution.threshold))))
    save(folder/'numeric.json',rows)
    assert not any(r['bounce_threshold_flip'] for r in rows),'Threshold crossing on captured inputs; review numeric.json'
    benches=dict(pytorch_amp=benchmark(reference,example),tensorrt=benchmark(deployed,example));save(folder/'benchmark.json',benches)
    save(folder/'manifest.json',dict(checkpoint_sha256=ctx.info['checkpoint_sha256'],tensorrt=trt.__version__,gpu=torch.cuda.get_device_name(),
        event=dict(engine='event.engine',inputs=NAMES,outputs=['logits'],output_dtype=str(dtype).split('.')[-1],precision='FP16 with FP32 LayerNorm arithmetic',build=built)))
    summary=dict(phase='completed',samples=len(rows),max_logit_delta=max(r['logits_max_abs'] for r in rows),max_probability_delta=max(r['probability_max_abs'] for r in rows),
        p50_pytorch=benches['pytorch_amp']['p50_ms'],p50_trt=benches['tensorrt']['p50_ms'],build_seconds=built['seconds'])
    save(state,summary);print(json.dumps(summary,indent=2),flush=True)
