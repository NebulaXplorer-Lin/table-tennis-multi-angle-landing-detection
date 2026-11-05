"""Train an independent table spatial encoder + MLP with RGB color jitter."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from ttnet.common import PROJECT, SRC, save, save_jsonl, sha
from ttnet.data_tools.geometry import letterbox
from ttnet.data_tools.prepare_table import TableCorpus, TableDataset
from ttnet.events.train import atomic_checkpoint, rng_state, set_rng, state_digest
from ttnet.models.table_light import NativeTableMLP
from ttnet.scene.table_mlp_thin import now, quota_order, vertical_loss
from ttnet.scene.train_light import parameter_groups
from ttnet.scene.train_visual import loader, evaluate


def load_recipe(path):
    cfg=json.loads(path.read_text())
    expected=set(json.loads((PROJECT/'configs/table/independent_mlp_color.json').read_text()))
    if set(cfg)!=expected or cfg['schema']!='independent_table_mlp_v1' or cfg['source']!='independent_cnn':
        raise ValueError('invalid independent MLP recipe')
    if cfg['pretrained'] is not None or cfg['global_checkpoint'] is not None or cfg['input']!=[512,288] or cfg['channels']!=32 or cfg['table_pool']!=[16,4]:
        raise ValueError('independent MLP uses native RGB, map32, pool16x4 and fresh weights')
    for key in ('epochs','patience','micro_batch','effective_batch','samples_per_epoch','max_thin_frame_repeats'):
        if type(cfg[key]) is not int or cfg[key]<1:
            raise ValueError('invalid '+key)
    if cfg['effective_batch']%cfg['micro_batch'] or cfg['samples_per_epoch']%cfg['effective_batch'] or cfg['workers']<0:
        raise ValueError('batches must divide the exposure budget without a singleton tail')
    if not 0<cfg['thin_fraction']<1 or not 0<cfg['max_thin_video_fraction']<=1 or cfg['selection']!='val_video_mean_iou':
        raise ValueError('invalid quota or checkpoint selection')
    if min(cfg['encoder_lr'],cfg['head_lr'],cfg['height_floor'],cfg['height_epsilon'],cfg['smooth_l1_beta'])<=0 or min(cfg['relative_y_weight'],cfg['log_height_weight'],cfg['weight_decay'],cfg['warmup_epochs'])<0:
        raise ValueError('invalid optimizer/loss values')
    a=cfg['augmentation']
    if set(a)!={'horizontal_flip','translate','scale','color','preserve_positive'} or a['preserve_positive'] is not True or not 0<=a['horizontal_flip']<=1 or not 0<=a['translate']<=.05 or not 0<a['scale'][0]<=a['scale'][1]:
        raise ValueError('invalid affine augmentation')
    c=a['color']
    if set(c)!={'probability','brightness','contrast','saturation','hue'} or not 0<=c['probability']<=1:
        raise ValueError('invalid color jitter fields')
    for name in ('brightness','contrast','saturation'):
        if len(c[name])!=2 or not 0<c[name][0]<=c[name][1]:
            raise ValueError('invalid color factor: '+name)
    if len(c['hue'])!=2 or not -.5<=c['hue'][0]<=c['hue'][1]<=.5 or cfg['augment_per_draw'] is not True:
        raise ValueError('invalid hue or per-occurrence augmentation')
    cfg['dataset']=str((PROJECT/cfg['dataset']).resolve())
    return cfg


def augmentation_examples(corpus, selected, cfg, root):
    """Save exact training draws beside unmodified inputs, with transformed GT."""
    dataset=TableDataset(corpus,selected,cfg,epoch=0,training=True)
    root.mkdir(exist_ok=True)
    records=[]
    for i in range(min(8,len(selected))):
        row=selected[i];item=dataset[i];original,t=letterbox(corpus.image(row['video_id'],row['frame_id']),*cfg['input'])
        image=np.round(item['images'].numpy().transpose(1,2,0)*255).astype(np.uint8)
        original=original[:,:,::-1].copy();image=image[:,:,::-1].copy()
        from ttnet.data_tools.geometry import map_point
        box=row['table_bbox_xyxy'];ob=[*map_point(*box[:2],t),*map_point(*box[2:],t)]
        nb=item['target'].numpy()*np.array(cfg['input']*2)
        for im,b in ((original,ob),(image,nb)):
            cv2.rectangle(im,tuple(round(v) for v in b[:2]),tuple(round(v) for v in b[2:]),(0,255,0),2)
        panel=np.concatenate([original,image],axis=1)
        name=f'{i:02d}_{row["video_id"]}_{row["frame_id"]:06d}.jpg'
        if not cv2.imwrite(str(root/name),panel):
            raise OSError(root/name)
        records.append(dict(draw=i,video_id=row['video_id'],frame_id=row['frame_id'],target=item['target'].tolist(),image=name))
    save(root/'manifest.json',dict(epoch=0,legend='left original; right exact affine+color training draw; green GT',draws=records))


def train(cfg, run, device, resume=False):
    if run.exists() and not resume:
        raise FileExistsError(run)
    corpus=TableCorpus(cfg['dataset'])
    files=('scene/train_independent_mlp.py','scene/train_light.py','scene/train_visual.py','scene/table_mlp_thin.py',
           'scene/table_mlp_control.py','scene/table.py','models/table_light.py','models/table_encoder.py',
           'data_tools/prepare_table.py','data_tools/geometry.py','data_tools/frame_store.py')
    dependencies=dict(table_annotations=corpus.identity,global_sha256=None,pretrained_sha256=None,
                      implementation={p:sha(SRC/p) for p in files})
    if resume and (json.loads((run/'config.json').read_text())!=cfg or json.loads((run/'dependencies.json').read_text())!=dependencies):
        raise ValueError('independent MLP resume config/code/dataset changed')
    saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False) if resume and (run/'last.pt').exists() else None
    if saved and (saved['config']!=cfg or saved['dependencies']!=dependencies):
        raise ValueError('checkpoint recipe or dependencies differ')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    model=NativeTableMLP(cfg['input'],cfg['table_pool']).to(device)
    optimizer=torch.optim.AdamW(parameter_groups(model,cfg));scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'] and device.type=='cuda')
    start=0;history=[];patience=0;best_scores=dict(video=-1.,frame=-1.,thin=-1.);stopped=False
    if saved:
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);set_rng(saved['rng'])
        start=saved['epoch'];history=saved['history'];patience=saved['patience'];best_scores=saved['best_scores'];stopped=saved['stopped']
    run.mkdir(parents=True,exist_ok=True);save(run/'config.json',cfg);save(run/'dependencies.json',dependencies)
    save(run/'parameter_groups.json',[{k:v for k,v in g.items() if k!='params'}|dict(parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups])
    status=dict(status='preparing',started=now(),updated=now(),run=str(run),architecture=model.architecture,
                parameters=sum(p.numel() for p in model.parameters()),training_range='independent_encoder_and_MLP',global_dependency=None)
    if resume and (run/'task.json').exists():
        status['started']=json.loads((run/'task.json').read_text()).get('started',status['started'])
    def update(**values):
        status.update(values,updated=now());save(run/'task.json',status);save(run/'run.json',status)
    def checkpoint(epoch):
        return dict(schema='table_visual_v1',config=cfg,dependencies=dependencies,architecture=model.architecture,
                    model=model.state_dict(),global_sha256=None,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),
                    epoch=epoch,history=history,patience=patience,best_scores=best_scores,stopped=stopped)
    update()
    if not (run/'last.pt').exists():
        atomic_checkpoint(run/'last.pt',checkpoint(0))
    positives=corpus.positives('train');keys=[(r['video_id'],r['frame_id']) for r in positives]
    for epoch in range(start,cfg['epochs']):
        if stopped:
            break
        order,records,audit=quota_order(keys,corpus,cfg,epoch,count=cfg['samples_per_epoch'])
        selected=[positives[int(i)] for i in order]
        save_jsonl(run/f'sampling/epoch_{epoch+1:03d}.jsonl',records);save(run/f'sampling/epoch_{epoch+1:03d}.json',audit)
        if epoch==0 and not (run/'augmentation_examples/manifest.json').exists():
            augmentation_examples(corpus,selected,cfg,run/'augmentation_examples')
        batches=loader(corpus,selected,cfg,epoch,True);accumulation=cfg['effective_batch']//cfg['micro_batch']
        model.train();optimizer.zero_grad(set_to_none=True);losses=[];parts=defaultdict(list);started=time.perf_counter();geometry_fallbacks=0
        encoder_before=state_digest(model.encoder) if epoch==0 else None;head_before=state_digest(model.head) if epoch==0 else None
        for step,batch in enumerate(batches):
            geometry_fallbacks+=sum(t.get('augmentation_geometry_fallback',False) for t in batch['transform'])
            progress=epoch+(step+1)/len(batches)
            factor=min(1.,progress/max(cfg['warmup_epochs'],1e-9))*.5*(1+math.cos(math.pi*progress/cfg['epochs']))
            for group in optimizer.param_groups:
                group['lr']=group['initial_lr']*factor
            with torch.autocast(device_type=device.type,enabled=cfg['amp'] and device.type=='cuda'):
                prediction,_=model(batch['images'].to(device))
                loss,terms=vertical_loss(prediction.float(),batch['target'].to(device).float(),cfg)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite independent MLP training loss')
            scaler.scale(loss/accumulation).backward()
            if (step+1)%accumulation==0:
                scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                if epoch==0 and not (run/'first_update.json').exists():
                    proof=dict(encoder_before=encoder_before,encoder_after=state_digest(model.encoder),head_before=head_before,head_after=state_digest(model.head),gradient_norm=float(norm))
                    if proof['encoder_before']!=proof['encoder_after'] and proof['head_before']!=proof['head_after']:
                        save(run/'first_update.json',proof)
            losses.append(float(loss.detach()))
            for name,value in terms.items():
                parts[name].append(float(value.detach()))
            if step%32==0 or step+1==len(batches):
                update(status='training',epoch=epoch+1,step=step+1,steps=len(batches),loss=losses[-1],
                       optimizer_steps=(step+1)//accumulation,elapsed_sec=time.perf_counter()-started,thin_exposure=audit)
                print({k:status[k] for k in ('status','epoch','step','steps','loss','elapsed_sec')},flush=True)
        if epoch==0 and not (run/'first_update.json').exists():
            raise ValueError('no optimizer update reached both independent encoder and MLP')
        update(status='validating',epoch=epoch+1)
        metrics=evaluate(model,None,corpus,cfg,device);save(run/f'validation/epoch_{epoch+1:03d}.json',metrics)
        scores=dict(video=metrics['video_mean_iou'],frame=metrics['mean_iou'],thin=metrics['by_bucket']['thin:all']['mean_iou'])
        improved={k:v>best_scores[k] for k,v in scores.items()};best_scores={k:max(v,best_scores[k]) for k,v in scores.items()}
        patience=0 if improved['video'] else patience+1;stopped=patience>=cfg['patience'] or epoch+1==cfg['epochs']
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),loss_components={k:float(np.mean(v)) for k,v in parts.items()},
                 val_frame_iou=scores['frame'],val_video_iou=scores['video'],val_thin_iou=scores['thin'],val_iou50=metrics['iou50'],
                 view_iou={k:v['mean_iou'] for k,v in metrics['by_bucket'].items() if k.startswith('view:')},
                 elapsed_sec=time.perf_counter()-started,thin_exposure=audit,geometry_fallbacks=geometry_fallbacks)
        history.append(row);state=checkpoint(epoch+1);atomic_checkpoint(run/'last.pt',state)
        for name,yes in improved.items():
            if yes:
                atomic_checkpoint(run/f'best_{name}.pt',state)
        if improved['video']:
            atomic_checkpoint(run/'best.pt',state)
        save(run/'history.json',history);update(status='trained_pending_evaluation' if stopped else 'training',epoch=epoch+1,best_scores=best_scores,latest_metrics=scores)
        print(row,flush=True)
    best=torch.load(run/'best.pt',map_location=device,weights_only=False);model.load_state_dict(best['model'])
    update(status='evaluating_best',best_epoch=best['epoch'])
    val=evaluate(model,None,corpus,cfg,device,run/'evaluation',overlay_mode='worst_per_video')
    train_metrics=evaluate(model,None,corpus,cfg,device,part='train')
    metrics=dict(train=train_metrics,val=val,best_epoch=best['epoch'],checkpoint_sha256=sha(run/'best.pt'),table_annotations=corpus.identity)
    save(run/'metrics.json',metrics)
    baselines=[('independent dense CNN',PROJECT/'out/experiments/20261104-095944_table_visual_comparison/independent_cnn/evaluation/metrics.json'),
               ('frozen B + 4x4 MLP',PROJECT/'out/experiments/20261104-142741_B_legacy_mlp_control/B/metrics.json'),
               ('frozen B + thin 16x4 MLP',PROJECT/'out/experiments/20261104-145747_B_mlp_thin/metrics.json')]
    comparisons={}
    for name,path in baselines:
        m=json.loads(path.read_text());v=m.get('val',m);buckets=v['by_bucket']
        comparisons[name]=dict(frame_mean_iou=v.get('frame_mean_iou',v.get('mean_iou')),video_mean_iou=v['video_mean_iou'],
                               thin_iou=buckets.get('thin',buckets.get('thin:all'))['mean_iou'],iou50=v['iou50'],metrics=str(path))
    comparisons['independent MLP + color']=dict(frame_mean_iou=val['mean_iou'],video_mean_iou=val['video_mean_iou'],thin_iou=val['by_bucket']['thin:all']['mean_iou'],iou50=val['iou50'],metrics=str(run/'metrics.json'))
    save(run/'comparison.json',comparisons)
    lines=['# Independent spatial encoder + MLP + color jitter','','Combined architecture/training recipe comparison; not an isolated color-jitter ablation.','',
           '| Recipe | Val frame IoU | Val video IoU | Thin IoU | IoU>=0.5 fraction |','|---|---:|---:|---:|---:|']
    for name,m in comparisons.items():
        lines.append(f'| {name} | {m["frame_mean_iou"]:.4f} | {m["video_mean_iou"]:.4f} | {m["thin_iou"]:.4f} | {m["iou50"]:.4f} |')
    (run/'comparison.md').write_text('\n'.join(lines)+'\n')
    update(status='completed',finished=now(),best_epoch=best['epoch'],metrics=str(run/'metrics.json'))
    print(json.dumps(comparisons['independent MLP + color']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=PROJECT/'configs/table/independent_mlp_color.json')
    p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    existed=args.run.exists()
    try:
        train(load_recipe(args.config),args.run,torch.device(args.device),args.resume)
    except BaseException as error:
        if args.run.exists() and (not existed or args.resume):
            save(args.run/'task.json',dict(status='failed',error=repr(error),updated=now()))
        raise
