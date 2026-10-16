"""Train paired frozen/joint EventHeads with match-held-out validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import cv2
import numpy as np
import torch

from ttnet.common import CONFIGS, EVENT_DATA, PROJECT, identity, now, save, save_jsonl, sha
from ttnet.events.data import EventCorpus, assemble_window, balanced_samples, sample_identity
from ttnet.events.engine import EventSystem, build_cache, load_cache, cache_identity, feature_cache_path
from ttnet.events.metrics import classification, select_thresholds
from ttnet.models.event import event_loss


def state_digest(module):
    digest=hashlib.sha256()
    for key,tensor in sorted(module.state_dict().items()):
        digest.update(key.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_checkpoint(path, value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def cached_forward(system,corpus,cache,sample):
    vid=sample['video_id'];features,records=cache[vid]
    # Keep full-video cache on CPU; only transfer the selected window.
    selected={fid:features[fid].to(system.device) for fid in corpus.window_ids(vid,sample['frame_id']) if fid is not None}
    return system.head(*assemble_window(corpus,vid,sample['frame_id'],selected,records,system.device))


def cached_forward_batch(system,corpus,cache,samples):
    inputs=[]
    for sample in samples:
        vid=sample['video_id'];features,records=cache[vid]
        inputs.append(assemble_window(corpus,vid,sample['frame_id'],features,records,torch.device('cpu')))
    tensors=[torch.cat([item[k] for item in inputs]).to(system.device,non_blocking=True) for k in range(3)]
    return system.head(*tensors)


def online_forward(system,corpus,sample):
    vid=sample['video_id'];center=sample['frame_id']
    ids=[i for i in corpus.window_ids(vid,center) if i is not None]
    # Extra predecessor makes first-slot velocity identical to the frozen cache.
    if ids[0]-1 in corpus.videos[vid]['frames']:ids=[ids[0]-1]+ids
    features,records,losses=system.forward_frames(corpus,vid,ids)
    logits=system.head(*assemble_window(corpus,vid,center,features,records,system.device))
    return logits,losses


def infer_cache(system,corpus,part):
    cache={}; sums={'global':0.,'local':0.};counts={'global':0,'local':0}
    with torch.inference_mode(),torch.autocast(device_type=system.device.type,enabled=system.config['amp'] and system.device.type=='cuda'):
        for vid in corpus.split[part]:
            system.reset_tracking(vid)
            if system.config.get('tracker_type')=='association':
                from ttnet.events.context import stream_context_features
                needed=sorted(corpus.videos[vid]['frames'])
                if needed!=list(range(len(needed))):
                    extracted=stream_context_features(system,corpus,vid,needed)
                    descriptors={};records={}
                    for fid,f,r in extracted:descriptors[fid]=f.float().cpu();records[fid]=r
                    cache[vid]=(descriptors,records)
                    continue
            descriptors={};records={}
            for fid in sorted(corpus.videos[vid]['frames']):
                d,r,loss=system.forward_frames(corpus,vid,[fid])
                descriptors[fid]=d[fid].float().cpu();records.update(r)
                for k in sums:
                    if corpus.videos[vid]['frames'][fid]['label_status'] in ('positive','negative') and (k=='global' or r[fid]['crop_transform']):
                        sums[k]+=float(loss[k]);counts[k]+=1
            cache[vid]=(descriptors,records)
    return cache,{k:sums[k]/max(1,counts[k]) for k in sums}


def validate(system,corpus,cache,part='val'):
    rows=[];loss_sum=weight=0.
    with torch.inference_mode(),torch.autocast(device_type=system.device.type,enabled=system.config['amp'] and system.device.type=='cuda'):
        samples=corpus.samples[part];size=system.config.get('validation_batch',1)
        for start in range(0,len(samples),size):
            chosen=samples[start:start+size]
            logits=cached_forward_batch(system,corpus,cache,chosen) if len(chosen)>1 else cached_forward(system,corpus,cache,chosen[0])
            target=torch.tensor([s['target'] for s in chosen],device=system.device)
            mask=torch.tensor([s['supervised'] for s in chosen],device=system.device)
            n=int(mask.sum());loss_sum+=float(event_loss(logits,target,mask))*n;weight+=n
            probabilities=logits.float().sigmoid().cpu().tolist()
            for sample,p in zip(chosen,probabilities):
                if sample['original']:
                    rows.append(dict(sample,target=sample['hard_target'],soft_target=sample['target'],probabilities=p))
    return loss_sum/max(1,weight),rows


def rng_state():
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng(state):
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])


def train(args):
    cfg=json.loads(args.config.read_text());cfg.update(mode=args.mode,stage='event',smoke=args.smoke)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run with GPU access')
    torch.set_num_threads(cfg['torch_threads']);cv2.setNumThreads(1)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    corpus=EventCorpus(args.dataset)
    groups=[g for part in ('train','val','test') for g in corpus.split[part]]
    if len(groups)!=len(set(groups)):raise ValueError('match leakage in split')
    if any(abs(v['meta']['fps']-cfg['target_fps'])>1e-9 or
           v['meta']['source_identity']['soft_radius_sec']!=cfg['soft_radius_sec'] for v in corpus.videos.values()):
        raise ValueError('sampling/target configuration differs from prepared data')
    continuous_selection=cfg.get('best_metric')=='pp_landing_macro_f1'
    if continuous_selection:
        if args.mode!='frozen' or cfg.get('sampling_protocol')!='openttgames_pp_50_50':
            raise ValueError('four-way protocol requires frozen localization and 50/50 source sampling')
        if not cfg.get('pp_annotations'):raise ValueError('reviewed PP annotations are required')
        from ttnet.events.continuous import attach_pp
        attach_pp(corpus,cfg['pp_annotations'])
    if not corpus.metadata['complete'] or corpus.metadata['pilot']:
        if not args.smoke:raise ValueError('formal training requires complete, non-pilot dataset')
    dataset_path=cfg.get('dataset',str(args.dataset)) if args.resume else str(args.dataset.resolve())
    cfg.update(dataset_id=corpus.identity,split_id=corpus.split['split_id'],dataset=dataset_path,
               train_samples_sha256=sample_identity(corpus.samples['train']),val_samples_sha256=sample_identity(corpus.samples['val']),
               global_sha256=sha(cfg['global_checkpoint']),local_sha256=sha(cfg['local_checkpoint']))
    system=EventSystem(cfg,device)
    cfg['head_initial_sha256']=state_digest(system.head)
    system.config=cfg
    if args.run.exists() and not args.resume:raise FileExistsError(args.run)
    args.run.mkdir(parents=True,exist_ok=True)
    if args.resume and json.loads((args.run/'config.yaml').read_text())!=cfg:raise ValueError('resume config/data mismatch')
    save(args.run/'config.yaml',cfg)
    attempts=args.run/'attempts';attempts.mkdir(exist_ok=True)
    attempt=attempts/f'{len(list(attempts.iterdir()))+1:03d}';attempt.mkdir()
    save(attempt/'environment.json',dict(started=now(),torch=torch.__version__,cuda=torch.version.cuda,
        device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        command=os.sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True),
        code_hashes={str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'ttnet/src/events').glob('*.py')},
        head_code_sha256=sha(PROJECT/'ttnet/src/models/event.py')))
    checkpoints=args.run/'checkpoints';checkpoints.mkdir(exist_ok=True)
    save(args.run/'run.json',dict(status='preparing',started=now(),mode=args.mode,stage='event'))
    system.configure(args.mode,True)
    params=[dict(params=list(system.head.parameters()),lr=cfg['lr'])]
    if args.mode=='joint':params.append(dict(params=list(system.global_model.parameters())+list(system.local_model.parameters()),lr=cfg['locator_lr']))
    optimizer=torch.optim.AdamW(params,weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler(device.type,enabled=cfg['amp'] and device.type=='cuda')
    start=0;start_step=0;resume_totals=None;resume_elapsed=0.;best=float('inf');reference=float('inf');patience=0
    if continuous_selection:best=reference=(float('inf'),float('inf'))
    if args.resume and not (checkpoints/'last.pt').exists():
        if not args.restart_incomplete:raise FileNotFoundError('no checkpoint; --restart-incomplete explicitly restarts from original initialization')
        save(attempt/'recovery.json',dict(reason='previous execution stopped before its first checkpoint',
                                         resumed_weights=False,restarted_from_initial_weights=True,updated=now()))
    elif args.resume:
        state=torch.load(checkpoints/'last.pt',map_location='cpu',weights_only=False)
        system.load_state_dict(state['system']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start=state['epoch'];best=state['best'];reference=state['reference'];patience=state['patience'];set_rng(state['rng'])
        start_step=state.get('step',0);resume_totals=state.get('train_totals');resume_elapsed=state.get('epoch_elapsed_sec',0.)
    cache=None
    if args.mode=='frozen' and not args.smoke:
        if args.cache is None:args.cache=feature_cache_path(system,corpus)
        build_cache(system,corpus,args.cache)
        cache=load_cache(corpus,args.cache,expected=cache_identity(system,corpus))
    locator_before=(state_digest(system.global_model),state_digest(system.local_model))
    if args.smoke:
        if args.mode=='frozen':
            sample=next(s for s in corpus.samples['train'] if s['original'] and s['group']=='bounce')
            vid=sample['video_id'];ids=[i for i in corpus.window_ids(vid,sample['frame_id']) if i is not None]
            if ids[0]-1 in corpus.videos[vid]['frames']:ids=[ids[0]-1]+ids
            if cfg.get('tracker_type')=='association':
                from ttnet.events.context import stream_context_features
                with torch.inference_mode():
                    extracted=list(stream_context_features(system,corpus,vid,ids))
                d={i:f for i,f,r in extracted};r={i:r for i,f,r in extracted}
            else:
                with torch.inference_mode():
                    d,r,_=system.forward_frames(corpus,vid,ids)
            cache={vid:({i:f.detach().cpu().clone() for i,f in d.items()},r)}
        smoke(system,corpus,cache,optimizer,scaler,args.run,locator_before)
        return
    def save_progress(epoch,step,totals,elapsed):
        atomic_checkpoint(checkpoints/'last.pt',dict(system=system.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),epoch=epoch,step=step,best=best,reference=reference,patience=patience,
            config=cfg,rng=rng_state(),train_totals=totals,epoch_elapsed_sec=elapsed))
    if not (checkpoints/'last.pt').exists():save_progress(start,0,None,0.)
    save(attempt/'checkpoint_policy.json',dict(initial_checkpoint=True,every_optimizer_steps=25,also_before_validation=True))
    for epoch in range(start,cfg['epochs']):
        started=time.perf_counter();system.configure(args.mode,True);optimizer.zero_grad(set_to_none=True)
        if continuous_selection:
            from ttnet.events.continuous import mixed_samples
            schedule=mixed_samples(corpus.samples['train'],cfg['seed'],epoch)
        else:schedule=balanced_samples(corpus.samples['train'],cfg['seed'],epoch)
        totals={k:0. for k in ('event','global','local','total')}
        skip=start_step if epoch==start else 0
        carried_elapsed=resume_elapsed if epoch==start else 0.
        if skip:
            if skip>len(schedule) or (skip%cfg['effective_batch'] and skip!=len(schedule)):raise ValueError('checkpoint is not at an optimizer boundary')
            if resume_totals is None:raise ValueError('checkpoint lacks accumulated loss totals')
            totals=resume_totals
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        micro=cfg.get('micro_batch',1) if args.mode=='frozen' else 1
        for step in range(0,len(schedule),micro):
            if step<skip:continue
            chosen=schedule[step:step+micro];sample=chosen[0];count=len(chosen)
            # Last, incomplete accumulation group is normalized by its actual size.
            group_count=min(cfg['effective_batch'],len(schedule)-(step//cfg['effective_batch'])*cfg['effective_batch'])
            with torch.autocast(device_type=device.type,enabled=cfg['amp'] and device.type=='cuda'):
                if args.mode=='frozen':
                    logits=cached_forward_batch(system,corpus,cache,chosen) if count>1 else cached_forward(system,corpus,cache,sample)
                    loc={k:logits.new_zeros(()) for k in ('global','local')}
                else:logits,loc=online_forward(system,corpus,sample)
                e=event_loss(logits,torch.tensor([s['target'] for s in chosen],device=device),torch.tensor([s['supervised'] for s in chosen],device=device),per_sample=True).mean()
                loss=cfg['loss_weights']['event']*e+sum(cfg['loss_weights'][k]*loc[k] for k in loc)
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at epoch {epoch+1} step {step}')
            scaler.scale(loss*count/group_count).backward()
            if (step+count)%cfg['effective_batch']==0 or step+count==len(schedule):
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            for k,v in dict(event=e,**loc,total=loss).items():totals[k]+=float(v.detach())*count
            if (step+count)%(25*cfg['effective_batch'])==0 or step+count==len(schedule):
                save_progress(epoch,step+count,totals,carried_elapsed+time.perf_counter()-started)
            if step%100==0:
                print(f'{args.mode} epoch {epoch+1} sample {step}/{len(schedule)} event={float(e.detach()):.5f}',flush=True)
                save(args.run/'run.json',dict(status='training',stage='event',mode=args.mode,epoch=epoch+1,step=step,
                                             samples=len(schedule),updated=now()))
        save(args.run/'run.json',dict(status='validating',stage='event',mode=args.mode,epoch=epoch+1,
                                     step=len(schedule),samples=len(schedule),updated=now()))
        system.configure(args.mode,False)
        val_cache,val_loc=(cache,{}) if args.mode=='frozen' else infer_cache(system,corpus,'val')
        val_loss,rows=validate(system,corpus,val_cache)
        continuous=None;sweep=None;continuous_rows=None
        if continuous_selection:
            from ttnet.events.continuous import validate_continuous
            continuous,sweep,continuous_rows=validate_continuous(system,corpus,val_cache,cfg['threshold_grid'])
            thresholds=[continuous['threshold'],.5]
            selection=(-continuous['macro_f1'],continuous['landing']['fp'])
            improved=selection<best
            if improved:best=reference=selection;patience=0
            else:patience+=1
            metrics=classification([r for r in rows if r.get('source')!='pp'],thresholds)
        else:
            thresholds=select_thresholds(rows,cfg['threshold_grid'])
            metrics=classification(rows,thresholds)
            improved=val_loss<best
            if improved:best=val_loss
            if val_loss<reference*(1-cfg['loss_relative_min_delta']):reference=val_loss;patience=0
            else:patience+=1
        record=dict(epoch=epoch+1,train={k:v/len(schedule) for k,v in totals.items()},val_event_loss=val_loss,
                    val_localization_loss=val_loc,val=metrics,thresholds=thresholds,
                    lr=[g['lr'] for g in optimizer.param_groups],elapsed_sec=carried_elapsed+time.perf_counter()-started,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                    schedule_sha256=sample_identity(schedule),patience=patience,
                    continuous=continuous,threshold_sweep=sweep,selection_metric=cfg.get('best_metric','val_event_masked_bce'))
        state=dict(system=system.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),epoch=epoch+1,
                   step=0,best=best,reference=reference,patience=patience,config=cfg,rng=rng_state(),thresholds=thresholds)
        if improved:
            atomic_checkpoint(checkpoints/'best.pt',state)
            save(args.run/'best_selection.json',record);save_jsonl(args.run/'predictions/best_val_anchors.jsonl',rows)
            if continuous_rows is not None:
                save(args.run/'predictions/best_pp_continuous.json',continuous_rows)
        atomic_checkpoint(checkpoints/'last.pt',state)
        with (attempt/'epochs.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True)
        if args.mode=='frozen' and locator_before!=(state_digest(system.global_model),state_digest(system.local_model)):
            raise AssertionError('frozen locator changed')
        save(args.run/'run.json',dict(status='training',stage='event',mode=args.mode,epochs_completed=epoch+1,
                                     best_selection=best,updated=now()))
        if patience>=cfg['patience']:break
    save(args.run/'run.json',dict(status='trained_pending_eval',stage='event',mode=args.mode,epochs_completed=epoch+1,
                                 finished=now(),stop_reason='early_stopping' if patience>=cfg['patience'] else 'max_epochs'))


def smoke(system,corpus,cache,optimizer,scaler,run,locator_before):
    system.configure(system.mode,True)
    sample=next(s for s in corpus.samples['train'] if s['original'] and s['group']=='bounce')
    history=[];gradient=None;local_gradient=None
    if system.mode=='frozen' and system.config.get('micro_batch',1)>1:
        count=system.config['micro_batch'];probe=cached_forward_batch(system,corpus,cache,[sample]*count)
        probe_loss=event_loss(probe,torch.tensor([sample['target']]*count,device=system.device),
                              torch.tensor([sample['supervised']]*count,device=system.device),per_sample=True).mean()
        probe_loss.backward();optimizer.zero_grad(set_to_none=True)
    for step in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=system.device.type,enabled=system.config['amp'] and system.device.type=='cuda'):
            if system.mode=='frozen':logits=cached_forward(system,corpus,cache,sample);loc={}
            else:logits,loc=online_forward(system,corpus,sample)
            e=event_loss(logits,torch.tensor([sample['target']],device=system.device),torch.tensor([sample['supervised']],device=system.device))
            if step==0 and system.mode=='joint':
                g=torch.autograd.grad(e,next(system.global_model.encoder.parameters()),retain_graph=True,allow_unused=True)[0]
                gradient=float(g.abs().sum()) if g is not None else 0.
                g=torch.autograd.grad(e,next(system.local_model.encoder.parameters()),retain_graph=True,allow_unused=True)[0]
                local_gradient=float(g.abs().sum()) if g is not None else 0.
            loss=e+sum(loc.values())
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        history.append(float(e.detach()))
    if system.mode=='joint' and not gradient>0:raise AssertionError('event loss did not reach global encoder')
    if system.mode=='frozen':
        if any(p.grad is not None for model in (system.global_model,system.local_model) for p in model.parameters()):
            raise AssertionError('frozen locator received gradient')
        if locator_before!=(state_digest(system.global_model),state_digest(system.local_model)):raise AssertionError('frozen weights changed')
    if not history[-1]<history[0]:raise AssertionError('smoke event loss did not decrease')
    atomic_checkpoint(run/'checkpoints/last.pt',dict(system=system.state_dict(),config=system.config))
    save(run/'smoke.json',dict(passed=True,micro_batch_checked=system.config.get('micro_batch',1),event_losses=history,event_to_global_gradient_l1=gradient,
                              event_to_local_gradient_l1=local_gradient,
                              local_path_note='may be unavailable when all local observations are invalid; covered with valid crops by unit test'))
    save(run/'run.json',dict(status='completed_smoke',stage='event',mode=system.mode,finished=now()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['frozen','joint'],required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,default=CONFIGS/'event.json')
    p.add_argument('--dataset',type=Path)
    p.add_argument('--cache',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--restart-incomplete',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.dataset is None:
        a.dataset=Path(json.loads((a.run/'config.yaml').read_text())['dataset']) if a.resume else EVENT_DATA
    if a.resume and '--config' not in os.sys.argv:a.config=a.run/'config.yaml'
    if a.run.exists() and not a.resume:p.error('run already exists; choose a new run or --resume')
    try:train(a)
    except Exception as exc:
        if a.run.exists():save(a.run/'run.json',dict(status='failed',stage='event',mode=a.mode,error=repr(exc),finished=now()))
        raise


if __name__=='__main__':main()
