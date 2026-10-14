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
from ttnet.temporal import temporal_mode,slots_for_mode,sample_window
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
    with torch.inference_mode():model(*dummy_inputs(model,shape,device))
    for h in hooks:h.remove()
    return dict(parameters=sum(p.numel() for p in model.parameters()),macs=macs,flops=2*macs,
                input_shape=shape,coverage='Conv2d only; excludes normalization, pooling, activations, resize, addition, preprocessing')


def dummy_inputs(model,shape,device):
    x=torch.zeros(shape,device=device)
    if model.temporal_mode=='single':return (x,)
    return (x,torch.ones((shape[0],shape[1]),device=device,dtype=torch.bool),torch.zeros((shape[0],shape[1]),device=device))


def window_inputs(images,timestamps,center,mode,device,half_span=1/30):
    if mode=='single':return (torch.from_numpy(images[center][None]).to(device),)
    index=sample_window(timestamps,center,mode,half_span)
    blank=np.zeros_like(images[center])
    x=np.stack([blank if i is None else images[i] for i in index['frame_ids']])[None]
    return (torch.from_numpy(x).to(device),torch.tensor([index['frame_valid']],device=device),
            torch.tensor([index['time_offsets_sec']],device=device,dtype=torch.float32))


@torch.inference_mode()
def benchmark(args):
    device=torch.device(args.device);torch.set_num_threads(4);cv2.setNumThreads(1)
    initialized=time.perf_counter()
    gm,gi=load_model(args.global_checkpoint,device)
    lm,li=load_model(args.local_checkpoint,device) if args.local_checkpoint else (None,None)
    gw,gh=input_size(gi.get('config'),'global')
    lw,lh=input_size(li.get('config'),'local') if li else (256,256)
    gmode=gm.temporal_mode;gslots=gm.slots
    lmode=lm.temporal_mode if lm else 'single';lslots=lm.slots if lm else 1
    gshape=(1,3,gh,gw) if gmode=='single' else (1,gslots,3,gh,gw)
    lshape=(1,3,lh,lw) if lmode=='single' else (1,lslots,3,lh,lw)
    amp=args.amp and device.type=='cuda'
    xs=[dummy_inputs(gm,gshape,device)]
    if lm:xs.append(dummy_inputs(lm,lshape,device))
    models=[gm]+([lm] if lm else [])
    sample_results=[]
    for model,x in zip(models,xs):
        for _ in range(50):
            with torch.autocast(device_type=device.type,enabled=amp):model(*x)
        sync(device);samples=[]
        for repeat in range(3):
            for _ in range(200):
                sync(device);start=time.perf_counter()
                with torch.autocast(device_type=device.type,enabled=amp):model(*x)
                sync(device);samples.append((time.perf_counter()-start)*1000)
        sample_results.append(dict(p50_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),samples_ms=samples))
    initialization_sec=time.perf_counter()-initialized
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    audits=jsonl(args.dataset/'splits/val.jsonl');video_results=[]
    all_frames=0;all_seconds=0.
    for e in audits:
        sync(device);started=time.perf_counter();count=0
        frames=[];timestamps=[]
        with av.open(str(PROJECT/e['video'])) as container:
            container.streams.video[0].thread_type='AUTO'
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format='rgb24'));timestamps.append(float(frame.pts*frame.time_base))
        h,w=frames[0].shape[:2]
        global_images=[];transform=None
        for rgb in frames:
            scaled,transform=letterbox(rgb,gw,gh);global_images.append(normalize(scaled))
        for center,rgb in enumerate(frames):
            x=window_inputs(global_images,timestamps,center,gmode,device,gi.get('config',{}).get('window_half_span_sec',1/30))
            with torch.autocast(device_type=device.type,enabled=amp):heat,_=gm(*x)
            px,py,conf=decode_heatmap(heat.float().cpu().numpy());px,py=map_point(px,py,transform,inverse=True)
            g=accept(dict(x=px,y=py,confidence=conf,source='global'),gi['threshold'],dict(width=w,height=h));chosen=g
            if lm and g['valid']:
                ct=crop_transform(px,py,w,h,lw,li.get('config',{}).get('local_crop_side_1080',256))
                index=sample_window(timestamps,center,lmode,li.get('config',{}).get('window_half_span_sec',1/30))
                local_images=[np.zeros((3,lh,lw),np.float32) if fid is None else normalize(crop_image(frames[fid],ct)) for fid in index['frame_ids']]
                if lmode=='single':cx=(torch.from_numpy(local_images[0][None]).to(device),)
                else:cx=(torch.from_numpy(np.stack(local_images)[None]).to(device),torch.tensor([index['frame_valid']],device=device),
                         torch.tensor([index['time_offsets_sec']],device=device,dtype=torch.float32))
                with torch.autocast(device_type=device.type,enabled=amp):local_heat,_=lm(*cx)
                lx,ly,lc=decode_heatmap(local_heat.float().cpu().numpy());lx,ly=crop_point(lx,ly,ct,inverse=True)
                local=accept(dict(x=lx,y=ly,confidence=lc,source='local'),li['threshold'],dict(width=w,height=h))
                if local['valid']:chosen=local
            json.dumps(dict(video_id=e['video_id'],frame_id=count,prediction=chosen),allow_nan=False);count+=1
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
                 global_compute=static_compute(gm,gshape,device),
                 local_compute=static_compute(lm,lshape,device) if lm else None,
                 temporal_mode=gmode,lookahead_frames=1 if gmode=='fixed3' else None,
                 global_sha256=sha(args.global_checkpoint),local_sha256=sha(args.local_checkpoint) if lm else None,
                 protocol='all held-out original videos; whole-video offline buffering for future frames; decode, per-frame global preprocessing, window assembly, copies, localization and JSON serialization; batch=1; disk output excluded')
    save(out/'metrics.json',results)
    from ttnet.results import write_results
    write_results()
    print(json.dumps({k:v for k,v in results.items() if k not in ['global_forward','local_forward','per_video']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,default=DATA)
    p.add_argument('--global-checkpoint',type=Path,required=True);p.add_argument('--local-checkpoint',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--amp',action='store_true');benchmark(p.parse_args())
