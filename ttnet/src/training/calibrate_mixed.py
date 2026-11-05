"""Real mixed-source optimizer-step calibration, sequential and two-process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader

from ttnet.common import PROJECT, SRC, save, sha
from ttnet.configuration import load_config
from ttnet.models.build import build_localizer
from ttnet.models.localizer import LocalizationLoss
from ttnet.temporal import move_inputs
from ttnet.training.cache import ensure_global_cache
from ttnet.training.dataset import BallDataset
from ttnet.training.mixed_sampling import MixedSourceSampler
from ttnet.training.train import seed_worker


def worker(args):
    cfg,_=load_config(PROJECT/f'configs/experiments/ball_mixed_{"baseline" if args.variant=="A" else "hr_se"}.json')
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg['micro_batch']=args.micro
    cache=ensure_global_cache(Path(cfg['data']),[cfg['global_input']],2)
    ds=BallDataset(Path(cfg['data']),'train',augment=True,temporal_mode='fixed3',input_cache=cache)
    ds.augmentation_seed=42
    sampler=MixedSourceSampler(ds.rows,42,16)
    loader=DataLoader(ds,batch_size=args.micro,sampler=sampler,num_workers=2,pin_memory=True,
                      worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(9149))
    iterator=iter(loader)
    device=torch.device('cuda')
    model=build_localizer(cfg,'global').to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda');criterion=LocalizationLoss()
    def update():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(16//args.micro):
            batch,target,valid,_=next(iterator)
            with torch.autocast('cuda'):
                heat,_=model(*move_inputs(batch,device))
                loss=criterion(heat,target.to(device),valid.to(device))*args.micro/16
            scaler.scale(loss).backward()
        scaler.step(optimizer);scaler.update()
    for _ in range(5):update()
    torch.cuda.synchronize()
    save(args.output.with_suffix('.ready.json'),dict(ready=True))
    if args.gate:
        deadline=time.monotonic()+180
        while not args.gate.exists():
            if time.monotonic()>deadline:raise TimeoutError('calibration gate')
            time.sleep(.1)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for step in range(args.steps):
        update()
        if step%25==0:print(f'{args.variant} micro={args.micro}: {step}/{args.steps}',flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    save(args.output,dict(variant=args.variant,micro_batch=args.micro,effective_batch=16,steps=args.steps,
        elapsed_sec=elapsed,samples_per_sec=args.steps*16/elapsed,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu=torch.cuda.get_device_name(0),sampling=sampler.manifest))


def run_workers(output,pair,parallel,steps):
    output.mkdir(parents=True,exist_ok=True)
    gate=output/'go';jobs=[]
    for variant,micro in zip(('A','B'),pair):
        result=output/f'{variant}.json'
        cmd=[sys.executable,'-u','-m','ttnet.training.calibrate_mixed','--worker','--variant',variant,
             '--micro',str(micro),'--steps',str(steps),'--output',str(result)]
        if parallel:cmd+=['--gate',str(gate)]
        stream=(output/f'{variant}.log').open('w')
        process=subprocess.Popen(cmd,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
        jobs.append((process,stream,result))
        if not parallel:
            if process.wait():raise RuntimeError(f'calibration failed: {output}/{variant}.log')
    if parallel:
        deadline=time.monotonic()+240
        while not all(path.with_suffix('.ready.json').exists() for _,_,path in jobs):
            if any(p.poll() is not None and p.returncode for p,_,_ in jobs) or time.monotonic()>deadline:
                for p,_,_ in jobs:
                    if p.poll() is None:p.terminate()
                for p,s,_ in jobs:p.wait();s.close()
                raise RuntimeError(f'calibration failed: {output}')
            time.sleep(.2)
        gate.touch()
    codes=[process.wait() for process,_,_ in jobs]
    for _,stream,_ in jobs:stream.close()
    if any(codes):raise RuntimeError(f'calibration failed: {output}; exits={codes}')
    reports=[json.loads(path.read_text()) for _,_,path in jobs]
    return dict(pair=pair,parallel=parallel,workers=reports,
                elapsed_sec=(max if parallel else sum)(r['elapsed_sec'] for r in reports),
                conservative_memory_bytes=(sum if parallel else max)(r['peak_reserved_bytes'] for r in reports))


def calibrate(output,steps=100,selected_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    total=torch.cuda.get_device_properties(0).total_memory
    sequential=run_workers(output/'serial_8_4',(8,4),False,steps)
    cases=[]
    for pair in (((8,4),) if selected_only else ((8,4),(16,4),(8,8))):
        try:
            report=run_workers(output/f'parallel_{pair[0]}_{pair[1]}',pair,True,steps)
            report['within_memory_budget']=report['conservative_memory_bytes']<total*.8-512*1024**2
            cases.append(report)
            print(json.dumps(report),flush=True)
        except RuntimeError as error:
            cases.append(dict(pair=pair,error=str(error),within_memory_budget=False))
    eligible=[r for r in cases if r.get('within_memory_budget')]
    best=min(eligible,key=lambda r:r['elapsed_sec']) if eligible else None
    if best and best['pair']!=[8,4] and tuple(best['pair'])!=(8,4):
        sequential=run_workers(output/'serial_selected',best['pair'],False,steps)
    parallel=bool(best and best['elapsed_sec']<=sequential['elapsed_sec']*.95)
    selected=best if parallel else sequential
    result=dict(status='calibrated',gpu=torch.cuda.get_device_name(0),total_memory_bytes=total,
                implementation_sha256={name:sha(SRC/name) for name in ('models/high_resolution.py','models/localizer.py','training/dataset.py','training/mixed_sampling.py')},
                steps=steps,serial=sequential,parallel_cases=cases,parallel=parallel,
                micro_batches=dict(zip(('A','B'),selected['pair'])),
                estimated_speedup=sequential['elapsed_sec']/best['elapsed_sec'] if best else None,
                policy='effective batch16; measured real mixed input+backward+AdamW; conservative total reserved<80% less512MiB; parallel needs>=5% completion-time gain')
    save(output/'calibration.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--variant',choices=['A','B']);p.add_argument('--micro',type=int)
    p.add_argument('--gate',type=Path);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--selected-only',action='store_true')
    a=p.parse_args()
    if a.worker:worker(a)
    else:print(json.dumps(calibrate(a.output,a.steps,a.selected_only)),flush=True)


if __name__=='__main__':main()
