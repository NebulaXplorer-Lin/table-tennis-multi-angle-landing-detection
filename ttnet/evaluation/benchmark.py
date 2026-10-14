from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import av
import cv2
import numpy as np
import torch

from ttnet.common import DATA,PROJECT,jsonl,now,save,sha
from ttnet.data_tools.geometry import letterbox,map_point,crop_image,crop_point,crop_transform
from ttnet.training.dataset import normalize
from ttnet.training.scales import input_size
from .inference import load_model
from .metrics import decode_heatmap,accept


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize()


def static_compute(model,shape,device):
    macs=0;hooks=[]
    def hook(m,args,out):
        nonlocal macs
        if isinstance(m,torch.nn.Conv2d):
            macs+=out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
    for m in model.modules():
        if isinstance(m,torch.nn.Conv2d):hooks.append(m.register_forward_hook(hook))
    with torch.inference_mode():model(torch.zeros(shape,device=device))
    for h in hooks:h.remove()
    return dict(parameters=sum(p.numel() for p in model.parameters()),macs=macs,flops=2*macs,
                input_shape=shape,coverage='Conv2d only; excludes normalization, pooling, activations, resize, addition, preprocessing')


@torch.inference_mode()
def benchmark(args):
    device=torch.device(args.device);torch.set_num_threads(4);cv2.setNumThreads(1)
    initialized=time.perf_counter()
    gm,gi=load_model(args.global_checkpoint,device)
    lm,li=load_model(args.local_checkpoint,device) if args.local_checkpoint else (None,None)
    gw,gh=input_size(gi.get('config'),'global')
    lw,lh=input_size(li.get('config'),'local') if li else (256,256)
    amp=args.amp and device.type=='cuda'
    xs=[torch.zeros((1,3,gh,gw),device=device)]
    if lm:xs.append(torch.zeros((1,3,lh,lw),device=device))
    models=[gm]+([lm] if lm else [])
    sample_results=[]
    for model,x in zip(models,xs):
        for _ in range(50):
            with torch.autocast(device_type=device.type,enabled=amp):model(x)
        sync(device);samples=[]
        for repeat in range(3):
            for _ in range(200):
                sync(device);start=time.perf_counter()
                with torch.autocast(device_type=device.type,enabled=amp):model(x)
                sync(device);samples.append((time.perf_counter()-start)*1000)
        sample_results.append(dict(p50_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),samples_ms=samples))
    initialization_sec=time.perf_counter()-initialized
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    audits=jsonl(args.dataset/'splits/val.jsonl');video_results=[]
    all_frames=0;all_seconds=0.
    for e in audits:
        sync(device);started=time.perf_counter();count=0
        with av.open(str(PROJECT/e['video'])) as container:
            container.streams.video[0].thread_type='AUTO'
            for frame in container.decode(video=0):
                rgb=frame.to_ndarray(format='rgb24');h,w=rgb.shape[:2]
                scaled,t=letterbox(rgb,gw,gh)
                x=torch.from_numpy(normalize(scaled)[None]).to(device)
                with torch.autocast(device_type=device.type,enabled=amp):heat,_=gm(x)
                px,py,conf=decode_heatmap(heat.float().cpu().numpy())
                px,py=map_point(px,py,t,inverse=True)
                g=accept(dict(x=px,y=py,confidence=conf,source='global'),gi['threshold'],dict(width=w,height=h))
                chosen=g
                if lm and g['valid']:
                    ct=crop_transform(px,py,w,h,lw,li.get('config',{}).get('local_crop_side_1080',256));crop=crop_image(rgb,ct)
                    cx=torch.from_numpy(normalize(crop)[None]).to(device)
                    with torch.autocast(device_type=device.type,enabled=amp):local_heat,_=lm(cx)
                    lx,ly,lc=decode_heatmap(local_heat.float().cpu().numpy());lx,ly=crop_point(lx,ly,ct,inverse=True)
                    local=accept(dict(x=lx,y=ly,confidence=lc,source='local'),li['threshold'],dict(width=w,height=h))
                    if local['valid']:chosen=local
                # Include output serialization in the end-to-end measurement.
                json.dumps(dict(video_id=e['video_id'],frame_id=count,prediction=chosen),allow_nan=False)
                count+=1
        sync(device);elapsed=time.perf_counter()-started
        video_results.append(dict(video_id=e['video_id'],frames=count,seconds=elapsed,fps=count/elapsed))
        all_frames+=count;all_seconds+=elapsed
        print(f"E2E {e['video_id']}: {count/elapsed:.1f} FPS",flush=True)
    out=args.run/'evaluations/benchmark';out.mkdir(parents=True,exist_ok=True)
    results=dict(measured=now(),backend='PyTorch',precision='FP16 AMP' if amp else 'FP32',device=str(device),
                 gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                 initialization_including_forward_benchmark_sec=initialization_sec,global_forward=sample_results[0],
                 local_forward=sample_results[1] if lm else None,e2e_fps=all_frames/all_seconds,
                 e2e_frames=all_frames,e2e_seconds=all_seconds,per_video=video_results,
                 peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                 global_compute=static_compute(gm,(1,3,gh,gw),device),
                 local_compute=static_compute(lm,(1,3,lh,lw),device) if lm else None,
                 global_sha256=sha(args.global_checkpoint),local_sha256=sha(args.local_checkpoint) if lm else None,
                 protocol='all held-out original videos; PyAV decode, resize, normalize, copy, localization and JSON serialization; batch=1; feature-window assembly and disk output excluded; GPU may be shared with other tasks')
    save(out/'metrics.json',results)
    print(json.dumps({k:v for k,v in results.items() if k not in ['global_forward','local_forward','per_video']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,default=DATA)
    p.add_argument('--global-checkpoint',type=Path,required=True);p.add_argument('--local-checkpoint',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--amp',action='store_true');benchmark(p.parse_args())
