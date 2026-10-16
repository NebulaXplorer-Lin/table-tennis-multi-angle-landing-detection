"""Measure frozen feature batches and event micro-batches on actual video data."""
import argparse
import time
from pathlib import Path
import torch
from ttnet.common import DATA, save
from ttnet.configuration import load_config
from ttnet.evaluation.ball_corpus import BallCorpus
from ttnet.events.engine import EventSystem
from ttnet.events.batched_features import independent_features
from ttnet.models.event import event_loss


def benchmark(output):
    cfg,_=load_config('configs/experiments/event_comparison_A.json');torch.set_num_threads(4)
    system=EventSystem(cfg,torch.device('cuda'));system.configure('frozen',False)
    corpus=BallCorpus(DATA);vid='22_001';ids=list(range(128));results=[];reference=None
    for size in (1,8,16,32,64):
        with torch.inference_mode(),torch.autocast('cuda',enabled=True):
            independent_features(system,corpus,vid,ids[:size])
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter();features={};records={}
            for start in range(0,len(ids),size):
                d,r,_=independent_features(system,corpus,vid,ids[start:start+size]);features.update(d);records.update(r)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
        values=torch.stack([features[i] for i in ids]).float().cpu()
        if reference is None:reference=values;baseline_records=records
        geometry_changed=sum((records[i]['prediction']['valid']!=baseline_records[i]['prediction']['valid'] or
            abs(records[i]['prediction']['x']-baseline_records[i]['prediction']['x'])>1.) for i in ids)
        result=dict(batch=size,fps=len(ids)/elapsed,elapsed_sec=elapsed,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    descriptor_max_abs_difference=float((values-reference).abs().max()),geometry_changes_over_1px=geometry_changed)
        results.append(result);print(result,flush=True)
    # Same effective batch 8: compare sequential accumulation to vectorized loss.
    head=[];system.configure('frozen',True)
    f=reference[:8].to('cuda')[:,None,:].expand(-1,13,-1).contiguous();m=torch.zeros(8,13,9,device='cuda');m[:,:,7]=1
    v=torch.ones(8,13,dtype=torch.bool,device='cuda');target=torch.zeros(8,2,device='cuda');mask=torch.ones_like(target,dtype=torch.bool);mask[::2,1]=False
    for micro in (1,2,4,8):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for step in range(40):
            system.head.zero_grad(set_to_none=True)
            for start in range(0,8,micro):
                with torch.autocast('cuda',enabled=True):
                    logits=system.head(f[start:start+micro],m[start:start+micro],v[start:start+micro])
                    loss=event_loss(logits,target[start:start+micro],mask[start:start+micro],per_sample=True).sum()/8
                loss.backward()
        torch.cuda.synchronize();head.append(dict(micro_batch=micro,effective_batch=8,optimizer_group_ms=(time.perf_counter()-started)*1000/40,
                                            peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    best=max(results,key=lambda r:r['fps'])
    result=dict(gpu=torch.cuda.get_device_name(),precision='AMP',video=vid,frames=len(ids),feature=results,event=head,
                selected_feature_batch=best['batch'],selected_micro_batch=8,selected_validation_batch=128,
                note='Feature extraction throughput, not full video FPS. Event measurement is forward/backward per effective batch8, without optimizer steps. AMP batch-size numerical differences are reported.')
    save(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();benchmark(a.output)
