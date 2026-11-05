"""Paired old-Global/B-Global MLP control; frozen encoders and identical heads."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch

from ttnet.common import DATA, PROJECT, SRC, identity, jsonl, save, save_jsonl, sha
from ttnet.data_tools.geometry import letterbox_transform, map_point, letterbox
from ttnet.data_tools.prepare_table import TableCorpus, is_thin
from ttnet.evaluation.inference import load_model
from ttnet.events.train import atomic_checkpoint, state_digest
from ttnet.scene.table import TableHead, table_loss, box_iou, decode_table
from ttnet.scene.train import extract, load_part


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def reference_features(corpus, source, dest, checkpoint):
    previous = json.loads((source/'identity.json').read_text())
    if previous['global_sha256']!=sha(checkpoint) or previous['splits']!=corpus.identity['split_hashes']:
        raise ValueError('reference cache does not match Global weights/frozen splits')
    ident = dict(source=str(source),source_identity_sha256=sha(source/'identity.json'),global_sha256=sha(checkpoint),
                 table_annotations=corpus.identity,target_version='v1_reencoded_from_original_pixels',input=[512,288],pool=[4,4])
    if (dest/'identity.json').exists() and json.loads((dest/'identity.json').read_text())!=ident:
        raise ValueError('reference derived-cache identity mismatch')
    save(dest/'identity.json',ident)
    for part in ('train','val'):
        for row in jsonl(corpus.root/f'splits/{part}.jsonl'):
            vid=row['video_id'];target=dest/f'{vid}.npz'
            if target.exists():
                continue
            with np.load(source/f'{vid}.npz',allow_pickle=False) as z:
                ids=z['frame_ids'].copy();features=z['features'].copy()
            if ids.tolist()!=corpus.frames[vid] or features.shape!=(len(ids),32,4,4) or not np.isfinite(features).all():
                raise ValueError('invalid reference feature rows')
            targets=[];valid=[]
            for fid in ids:
                label=corpus.by_key[(vid,int(fid))];box=label['table_bbox_xyxy'];valid.append(box is not None)
                if box is None:
                    targets.append([0.]*4)
                else:
                    r=label['resolution'];t=letterbox_transform(r['width'],r['height'])
                    targets.append(np.array([*map_point(*box[:2],t),*map_point(*box[2:],t)],np.float32)/[512,288,512,288])
            temporary=target.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,features=features,frame_ids=ids,targets=np.array(targets,np.float32),valid=np.array(valid))
            temporary.replace(target)
    return ident


def summarize(iou, keys, corpus, views):
    values=np.asarray(iou,dtype=float);videos=defaultdict(list);buckets=defaultdict(list)
    for value,key in zip(values,keys):
        vid,fid=key;row=corpus.by_key[key];videos[vid].append(float(value))
        buckets['view:'+views.get(vid,'unknown')].append(float(value))
        buckets['group:'+corpus.groups[vid]].append(float(value))
        if is_thin(row):
            buckets['thin'].append(float(value))
    return dict(frames=len(values),frame_mean_iou=float(values.mean()),
                video_mean_iou=float(np.mean([np.mean(v) for v in videos.values()])),
                iou50=float((values>=.5).mean()),iou75=float((values>=.75).mean()),
                per_video={k:dict(frames=len(v),mean_iou=float(np.mean(v))) for k,v in videos.items()},
                by_bucket={k:dict(frames=len(v),mean_iou=float(np.mean(v)),iou50=float(np.mean(np.array(v)>=.5))) for k,v in buckets.items()})


def run(task, checkpoint, device, feature_batch=8, resume=False):
    if task.exists() and not resume:
        raise FileExistsError(task)
    corpus=TableCorpus(PROJECT/'data/datasets/pp_table_reannotated_v1')
    old_run=PROJECT/'out/experiments/20261016_table_ar/table'
    old_cfg=json.loads((old_run/'config.yaml').read_text())
    old_checkpoint=Path(old_cfg['global_checkpoint'])
    if not old_checkpoint.is_absolute():
        old_checkpoint=PROJECT/old_checkpoint
    cfg=dict(seed=42,epochs=60,lr=3e-4,batch=128,patience=10,weight_decay=1e-4,loss='L1+GIoU',
             selection='val_frame_mean_iou',augmentation='none',training_samples='all17281_positive_frames_each_epoch',
             initialization='identical_fresh_TableHead_seed42',sampling='same_per_epoch_permutation_for_both_heads',
             table_dataset=str(corpus.root),table_annotations=corpus.identity,feature_batch=feature_batch,
             dependencies={name:sha(SRC/name) for name in ('scene/table_mlp_control.py','scene/train.py','scene/table.py',
                 'models/localizer.py','models/high_resolution.py','data_tools/geometry.py','data_tools/frame_store.py')})
    if resume and json.loads((task/'config.json').read_text())!=cfg:
        raise ValueError('paired MLP control config/code changed')
    task.mkdir(parents=True,exist_ok=True);save(task/'config.json',cfg)
    manifest=dict(status='extracting',updated=now(),B_global_checkpoint=str(checkpoint),B_global_sha256=sha(checkpoint),
                  reference_global_checkpoint=str(old_checkpoint),reference_global_sha256=sha(old_checkpoint),
                  selection=cfg['selection'],branches={name:dict(run=str(task/name),status='preparing') for name in ('reference','B')})
    save(task/'task.json',manifest)
    reference_features(corpus,old_run/'features',task/'features_reference',old_checkpoint)
    model,info=load_model(checkpoint,device)
    if info['config'].get('global_localizer_type',info['config'].get('localizer_type'))!='high_resolution_se':
        raise ValueError('control requires the mixed-ball B Global checkpoint')
    model.eval().requires_grad_(False)
    extract(model,checkpoint,task/'features_B',device,batch=feature_batch,table_dataset=corpus.root)
    del model
    if device.type=='cuda':
        torch.cuda.empty_cache()
    views={r['video_id']:r.get('view','unknown') for r in json.loads((PROJECT/'data/annotations/pp_bounce_confirmed_v1.json').read_text())['videos']}
    data={};keys={};branches={}
    torch.manual_seed(cfg['seed']);initial=TableHead();initial_weights=deepcopy(initial.state_dict());init_digest=state_digest(initial)
    for name,global_path in (('reference',old_checkpoint),('B',checkpoint)):
        cache=task/('features_reference' if name=='reference' else 'features_B')
        data[name]={};keys[name]={}
        for part in ('train','val'):
            x,y,k=load_part(cache,part,device);data[name][part]=(x,y);keys[name][part]=k
        root=task/name;root.mkdir(exist_ok=True)
        branch_cfg=dict(cfg,global_checkpoint=str(global_path),global_sha256=sha(global_path),initial_head_sha256=init_digest)
        save(root/'config.yaml',branch_cfg)
        head=TableHead().to(device);head.load_state_dict(initial_weights)
        opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        b=dict(head=head,optimizer=opt,epoch=0,best=-1.,patience=0,history=[],stopped=False,root=root,config=branch_cfg)
        if resume and (root/'last.pt').exists():
            s=torch.load(root/'last.pt',map_location=device,weights_only=False)
            if s['config']!=branch_cfg:
                raise ValueError('paired head dependency mismatch')
            head.load_state_dict(s['head']);opt.load_state_dict(s['optimizer'])
            b.update({k:s[k] for k in ('epoch','best','patience','history','stopped')})
        branches[name]=b
    for part in ('train','val'):
        if keys['reference'][part]!=keys['B'][part] or not torch.equal(data['reference'][part][1],data['B'][part][1]):
            raise ValueError('paired features have different sample order or targets')
    manifest.update(status='training',initial_head_sha256=init_digest,updated=now());save(task/'task.json',manifest)
    for epoch in range(cfg['epochs']):
        if all(b['stopped'] for b in branches.values()):
            break
        order=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch])).permutation(len(keys['B']['train']))
        save(task/f'sampling/epoch_{epoch+1:03d}.json',dict(count=len(order),order_sha256=identity(order.tolist()),seed=cfg['seed'],epoch=epoch+1))
        indexes=torch.from_numpy(order).to(device)
        for name,b in branches.items():
            if b['stopped'] or b['epoch']>epoch:
                continue
            head,opt=b['head'],b['optimizer'];x,y=data[name]['train'];vx,vy=data[name]['val']
            started=time.perf_counter();head.train();losses=[]
            for ix in indexes.split(cfg['batch']):
                opt.zero_grad(set_to_none=True);loss=table_loss(head(x[ix]),y[ix])
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite MLP loss')
                loss.backward();opt.step();losses.append(float(loss.detach()))
            head.eval()
            with torch.inference_mode():
                pred=torch.cat([head(a) for a in vx.split(256)]);iou=box_iou(pred,vy).float().cpu().numpy()
            video=summarize(iou,keys[name]['val'],corpus,views)['video_mean_iou'];score=float(iou.mean());improved=score>b['best']
            b.update(epoch=epoch+1,best=max(b['best'],score),patience=0 if improved else b['patience']+1)
            b['stopped']=b['patience']>=cfg['patience'] or epoch+1==cfg['epochs']
            b['history'].append(dict(epoch=epoch+1,loss=float(np.mean(losses)),val_frame_mean_iou=score,
                                     val_video_mean_iou=video,elapsed_sec=time.perf_counter()-started))
            saved={k:b[k] for k in ('epoch','best','patience','history','stopped','config')}
            saved.update(head=head.state_dict(),optimizer=opt.state_dict(),feature_identity=json.loads((task/('features_reference' if name=='reference' else 'features_B')/'identity.json').read_text()))
            atomic_checkpoint(b['root']/'last.pt',saved)
            if improved:
                atomic_checkpoint(b['root']/'best.pt',saved)
            save(b['root']/'history.json',b['history']);save(b['root']/'run.json',dict(status='trained_pending_evaluation' if b['stopped'] else 'training',epoch=epoch+1,best_val_iou=b['best'],updated=now()))
            manifest['branches'][name].update(epoch=epoch+1,best_val_iou=b['best'],status='trained' if b['stopped'] else 'training')
            manifest['updated']=now();save(task/'task.json',manifest);print(name,b['history'][-1],flush=True)
    comparison={}
    for name,b in branches.items():
        s=torch.load(b['root']/'best.pt',map_location=device,weights_only=False);head=b['head'];head.load_state_dict(s['head']);head.eval();metrics={}
        for part in ('train','val'):
            x,y=data[name][part]
            with torch.inference_mode():
                pred=torch.cat([head(a) for a in x.split(256)]);iou=box_iou(pred,y).float().cpu().numpy()
            metrics[part]=summarize(iou,keys[name][part],corpus,views)
            if part=='val':
                rows=[dict(video_id=k[0],frame_id=k[1],iou=float(v),predicted_box=p.tolist(),label_box=g.tolist()) for k,v,p,g in zip(keys[name][part],iou,pred.cpu(),y.cpu())]
                save(b['root']/'val_predictions.json',rows)
                save(b['root']/'error_index.json',[r for r in rows if r['iou']<.5])
                cases={}
                for r in rows:
                    if r['video_id'] not in cases or r['iou']<cases[r['video_id']]['iou']:
                        cases[r['video_id']]=r
                for vid,r in cases.items():
                    label=corpus.by_key[(vid,r['frame_id'])];res=label['resolution'];t=letterbox_transform(res['width'],res['height'])
                    image=corpus.image(vid,r['frame_id']).copy()[:,:,::-1].copy();box=decode_table(r['predicted_box'],t,res)['bbox_xyxy']
                    for coords,color in ((label['table_bbox_xyxy'],(0,255,0)),(box,(0,0,255))):
                        cv2.rectangle(image,tuple(round(z) for z in coords[:2]),tuple(round(z) for z in coords[2:]),color,2)
                    dest=b['root']/'examples'/f'{vid}_{r["frame_id"]:06d}.jpg';dest.parent.mkdir(exist_ok=True)
                    if not cv2.imwrite(str(dest),image):
                        raise OSError(dest)
        metrics.update(epoch=s['epoch'],checkpoint_sha256=sha(b['root']/'best.pt'),table_annotations=corpus.identity)
        save(b['root']/'metrics.json',metrics);save(b['root']/'run.json',dict(status='completed',best_epoch=s['epoch'],metrics=str(b['root']/'metrics.json'),finished=now()))
        comparison[name]=dict(best_epoch=s['epoch'],train=metrics['train'],val=metrics['val'])
        manifest['branches'][name].update(status='completed',best_epoch=s['epoch'])
    save(task/'comparison.json',comparison)
    lines=['# Paired frozen Global + legacy MLP control','','Same seed/init, permutations, labels, optimizer, batch and selection. Both heads trained from scratch.','',
           '| Encoder | Best epoch | Train frame IoU | Val frame IoU | Val video IoU |','|---|---:|---:|---:|---:|']
    for name,m in comparison.items():
        lines.append(f'| {name} | {m["best_epoch"]} | {m["train"]["frame_mean_iou"]:.4f} | {m["val"]["frame_mean_iou"]:.4f} | {m["val"]["video_mean_iou"]:.4f} |')
    lines+=['','Comparison with B dense-CNN head also changes the training recipe; that comparison alone is not a pure head ablation.']
    (task/'comparison.md').write_text('\n'.join(lines)+'\n')
    manifest.update(status='completed',finished=now(),updated=now());save(task/'task.json',manifest)
    print(json.dumps({k:dict(best_epoch=v['best_epoch'],val_frame_iou=v['val']['frame_mean_iou'],val_video_iou=v['val']['video_mean_iou']) for k,v in comparison.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=PROJECT/'out/experiments/20261103-142106_mixed_ball_B_global/checkpoints/best.pt')
    p.add_argument('--device',default='cuda');p.add_argument('--feature-batch',type=int,default=8);p.add_argument('--resume',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    if args.feature_batch<1:
        p.error('feature batch must be positive')
    try:
        run(args.task,args.checkpoint,torch.device(args.device),args.feature_batch,args.resume)
    except BaseException as error:
        if args.task.exists():
            save(args.task/'task.json',dict(status='failed',error=repr(error),updated=now()))
        raise
