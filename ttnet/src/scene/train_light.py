"""Native table CNN trainer, isolated from the already-running B trainer."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
import torch

from ttnet.common import PROJECT, SRC, now, save, save_jsonl, sha
from ttnet.data_tools.prepare_table import TableCorpus, sample_epoch
from ttnet.models.table_light import NativeTableEncoder
from ttnet.scene.train_visual import loader, evaluate


def load_recipe(path):
    cfg = json.loads(Path(path).read_text())
    required = set(json.loads((PROJECT/'configs/table/shared_B.json').read_text()))
    if set(cfg)!=required or cfg['schema']!='table_visual_recipe_v1' or cfg['source']!='independent_cnn':
        raise ValueError('invalid native table recipe')
    if cfg['pretrained'] is not None or cfg['global_checkpoint'] is not None or cfg['channels']!=32 or cfg['input']!=[512,288]:
        raise ValueError('native table CNN uses single-frame512x288, map32 and no pretrained weights')
    for key in ('epochs','patience','micro_batch','effective_batch','samples_per_epoch'):
        if type(cfg[key]) is not int or cfg[key]<1:
            raise ValueError('invalid '+key)
    if cfg['effective_batch']%cfg['micro_batch'] or type(cfg['workers']) is not int or cfg['workers']<0:
        raise ValueError('invalid accumulation/workers')
    if not 1<=cfg['thin_weight']<=3 or min(cfg['encoder_lr'],cfg['head_lr'])<=0 or cfg['weight_decay']<0 or cfg['warmup_epochs']<0:
        raise ValueError('invalid sampling/optimizer configuration')
    a = cfg['augmentation']
    if set(a)!={'horizontal_flip','translate','scale'} or not 0<=a['horizontal_flip']<=1 or not 0<=a['translate']<=.05 or not 0<a['scale'][0]<=a['scale'][1]:
        raise ValueError('invalid native table augmentation')
    if not 0<=cfg['confidence']<=1 or not 0<cfg['nms_iou']<=1:
        raise ValueError('invalid table confidence')
    path = Path(cfg['dataset']);cfg['dataset']=str(path if path.is_absolute() else PROJECT/path)
    return cfg


def parameter_groups(model, cfg):
    head_ids = {id(p) for p in model.head.parameters()}
    buckets = defaultdict(list)
    for p in model.parameters():
        if p.requires_grad:
            buckets[(id(p) in head_ids,p.ndim>1)].append(p)
    return [dict(params=ps,lr=cfg['head_lr'] if head else cfg['encoder_lr'],
                 initial_lr=cfg['head_lr'] if head else cfg['encoder_lr'],weight_decay=cfg['weight_decay'] if decay else 0.,
                 name=('head' if head else 'encoder')+('_decay' if decay else '_no_decay')) for (head,decay),ps in buckets.items()]


def train(cfg, corpus, run, device, resume=False):
    from ttnet.events.train import atomic_checkpoint, rng_state, set_rng
    if run.exists() and not resume:
        raise FileExistsError(run)
    state = torch.load(run/'last.pt',map_location='cpu',weights_only=False) if resume else None
    dependencies = dict(table_annotations=corpus.identity,global_sha256=None,pretrained_sha256=None,
                        implementation={p:sha(SRC/p) for p in ('scene/train_light.py','models/table_light.py','scene/train_visual.py','models/table_encoder.py','data_tools/prepare_table.py')})
    if state and (state['config']!=cfg or state['dependencies']!=dependencies):
        raise ValueError('resume recipe or dependencies changed')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    model = NativeTableEncoder(cfg['input']).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model,cfg))
    scaler = torch.amp.GradScaler('cuda',enabled=cfg['amp'] and device.type=='cuda')
    start,best,patience,history = 0,-1.,0,[]
    if state:
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start,best,patience,history=state['epoch'],state['best'],state['patience'],state['history'];set_rng(state['rng'])
    run.mkdir(parents=True,exist_ok=True)
    save(run/'config.json',cfg);save(run/'dependencies.json',dependencies)
    save(run/'parameter_groups.json',[{k:v for k,v in g.items() if k!='params'}|dict(parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups])
    def checkpoint(epoch):
        return dict(schema='table_visual_v1',config=cfg,model=model.state_dict(),architecture=model.architecture,
                    global_sha256=None,dependencies=dependencies,epoch=epoch,best=best,patience=patience,history=history,
                    optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state())
    if not (run/'last.pt').exists():
        atomic_checkpoint(run/'last.pt',checkpoint(start))
    for epoch in range(start,cfg['epochs']):
        selected = sample_epoch(corpus.positives('train'),corpus.groups,cfg['seed'],epoch,cfg['samples_per_epoch'],cfg['thin_weight'])
        save_jsonl(run/f'sampling/epoch_{epoch+1:03d}.jsonl',[dict(video_id=r['video_id'],frame_id=r['frame_id']) for r in selected])
        batches=loader(corpus,selected,cfg,epoch,True);accumulation=cfg['effective_batch']//cfg['micro_batch']
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for step,batch in enumerate(batches):
            denominator=min(accumulation,len(batches)-(step//accumulation)*accumulation)
            progress=epoch+(step+1)/len(batches)
            factor=min(1.,progress/max(cfg['warmup_epochs'],1e-9))*.5*(1+math.cos(math.pi*progress/cfg['epochs']))
            for group in optimizer.param_groups:
                group['lr']=group['initial_lr']*factor
            with torch.autocast(device_type=device.type,enabled=cfg['amp'] and device.type=='cuda'):
                prediction,_=model(batch['images'].to(device));loss,_=model.loss(prediction,batch['target'].to(device))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite native table loss')
            scaler.scale(loss/denominator).backward()
            if (step+1)%accumulation==0 or step+1==len(batches):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if step%50==0 or step+1==len(batches):
                info=dict(status='training',epoch=epoch+1,step=step+1,steps=len(batches),loss=losses[-1],updated=now())
                save(run/'run.json',info);print(info,flush=True)
        save(run/'run.json',dict(status='validating',epoch=epoch+1,updated=now()))
        metrics=evaluate(model,None,corpus,cfg,device);score=metrics['video_mean_iou'];improved=score>best
        best=max(best,score);patience=0 if improved else patience+1
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),video_mean_iou=score))
        saved=checkpoint(epoch+1);atomic_checkpoint(run/'last.pt',saved)
        if improved:
            atomic_checkpoint(run/'best.pt',saved)
        save(run/'history.json',history);print(history[-1],flush=True)
        if patience>=cfg['patience']:
            break
    model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=False)['model'])
    evaluate(model,None,corpus,cfg,device,run/'evaluation')
    save(run/'run.json',dict(status='trained_evaluated',best_video_iou=best))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path)
    p.add_argument('--train',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cpu')
    args=p.parse_args();cfg=load_recipe(args.config);corpus=TableCorpus(cfg['dataset'])
    if not args.train:
        if args.resume:
            p.error('--resume requires --train')
        print(json.dumps(dict(status='prepared_no_training',config=cfg,stats=corpus.audit()),indent=2));return
    if args.run is None:
        p.error('--train requires --run')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    try:
        train(cfg,corpus,args.run,torch.device(args.device),args.resume)
    except BaseException as error:
        if args.run.exists():
            save(args.run/'run.json',dict(status='failed',error=repr(error),updated=now()))
        raise


if __name__=='__main__':
    main()
