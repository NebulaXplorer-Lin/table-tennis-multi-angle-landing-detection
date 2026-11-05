"""Frozen B + table MLP with thin quotas, vertical loss and 16x4 pooling."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from ttnet.common import PROJECT, SRC, identity, jsonl, save, save_jsonl, sha
from ttnet.data_tools.geometry import letterbox_transform
from ttnet.data_tools.prepare_table import TableCorpus, is_thin
from ttnet.evaluation.inference import load_model
from ttnet.events.train import atomic_checkpoint, state_digest
from ttnet.scene.table import TableHead, box_iou, decode_table
from ttnet.scene.table_mlp_control import now, summarize
from ttnet.scene.train import extract, load_part


def quota_order(keys, corpus, cfg, epoch, count=None):
    """Exact thin exposure, <=4 uses/frame and <=30% thin slots/video."""
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch]))
    thin=defaultdict(list);ordinary=[]
    for i,key in enumerate(keys):
        row=corpus.by_key[key]
        if row['split']!='train' or row['table_status']!='positive':
            raise ValueError('quota sampler only accepts positive training labels')
        if is_thin(row):
            thin[key[0]].append(i)
        else:
            ordinary.append(i)
    budget=len(keys) if count is None else int(count)
    if budget<1:
        raise ValueError('sampling budget must be positive')
    target=round(budget*cfg['thin_fraction']);video_cap=int(np.ceil(target*cfg['max_thin_video_fraction']))
    queues={vid:np.concatenate([rng.permutation(ids) for _ in range(cfg['max_thin_frame_repeats'])]).tolist()
            for vid,ids in sorted(thin.items())}
    capacity=sum(min(len(q),video_cap) for q in queues.values())
    if target>capacity or budget-target>len(ordinary):
        raise ValueError(f'thin quota infeasible: requested {target}, bounded capacity {capacity}')
    chosen=[];counts=Counter()
    while len(chosen)<target:
        active=[vid for vid,q in queues.items() if q and counts[vid]<video_cap]
        if not active:
            raise ValueError('thin quota ran out of eligible frames')
        for vid in rng.permutation(active):
            chosen.append(queues[vid].pop(0));counts[vid]+=1
            if len(chosen)==target:
                break
    order=rng.permutation(chosen+rng.choice(ordinary,budget-target,replace=False).tolist())
    repetitions=Counter(int(i) for i in order)
    if len(order)!=budget or max(repetitions.values())>cfg['max_thin_frame_repeats']:
        raise ValueError('incorrect exposure budget or frame repetition limit')
    records=[dict(video_id=keys[int(i)][0],frame_id=keys[int(i)][1],
                  thin=is_thin(corpus.by_key[keys[int(i)]]),group=corpus.groups[keys[int(i)][0]]) for i in order]
    audit=dict(epoch=epoch+1,count=len(order),thin=target,thin_fraction=target/len(order),
               unique_frames=len(repetitions),unique_thin_frames=len(set(chosen)),thin_videos=len(counts),
               max_frame_repeats=max(repetitions.values()),thin_video_cap=video_cap,
               thin_by_video=dict(sorted(counts.items())),
               thin_by_group=dict(Counter(r['group'] for r in records if r['thin'])),
               order_sha256=identity(order.tolist()),seed=cfg['seed'])
    return order,records,audit


def vertical_loss(pred, target, cfg):
    l1=F.l1_loss(pred,target);giou=(1-box_iou(pred,target,True)).mean()
    ph=(pred[:,3]-pred[:,1]).clamp_min(cfg['height_epsilon'])
    gh=(target[:,3]-target[:,1]).clamp_min(cfg['height_epsilon'])
    pc=(pred[:,1]+pred[:,3])*.5;gc=(target[:,1]+target[:,3])*.5
    relative_y=F.smooth_l1_loss((pc-gc)/gh.clamp_min(cfg['height_floor']),torch.zeros_like(pc),beta=cfg['smooth_l1_beta'])
    log_height=F.smooth_l1_loss(ph.log()-gh.log(),torch.zeros_like(ph),beta=cfg['smooth_l1_beta'])
    loss=l1+giou+cfg['relative_y_weight']*relative_y+cfg['log_height_weight']*log_height
    return loss,dict(l1=l1,giou=giou,relative_y=relative_y,log_height=log_height)


def vertical_summary(pred, target, keys, corpus):
    a=np.asarray(pred);b=np.asarray(target);h=b[:,3]-b[:,1]
    values=dict(center_shift_gt_height=((a[:,1]+a[:,3])-(b[:,1]+b[:,3]))/(2*h),
                height_ratio=(a[:,3]-a[:,1])/h,
                top_abs_error_gt_height=np.abs(a[:,1]-b[:,1])/h,
                bottom_abs_error_gt_height=np.abs(a[:,3]-b[:,3])/h)
    masks={'all':np.ones(len(keys),dtype=bool),'thin':np.array([is_thin(corpus.by_key[k]) for k in keys])}
    for group in sorted({corpus.groups[k[0]] for k in keys}):
        masks['group:'+group]=np.array([corpus.groups[k[0]]==group for k in keys])
    out={}
    for name,mask in masks.items():
        if not mask.any():
            continue
        out[name]=dict(frames=int(mask.sum()),**{k:dict(zip(('p10','median','p90'),np.quantile(v[mask],[.1,.5,.9]).tolist())) for k,v in values.items()},
                       center_abs_error_gt_height_median=float(np.median(np.abs(values['center_shift_gt_height'][mask]))))
    return out


def predict(head, x):
    with torch.inference_mode():
        return torch.cat([head(a) for a in x.split(256)])


def run(root, config_path, device, resume=False):
    if root.exists() and not resume:
        raise FileExistsError(root)
    cfg=json.loads(config_path.read_text())
    if cfg['table_pool']!=[16,4] or cfg['selection']!='val_frame_mean_iou' or not 0<cfg['thin_fraction']<1:
        raise ValueError('unsupported table pooling, selection or thin quota')
    if cfg['max_thin_frame_repeats']<1 or not 0<cfg['max_thin_video_fraction']<=1 or min(cfg['batch'],cfg['feature_batch'])<1:
        raise ValueError('invalid sampling or batch limits')
    if min(cfg['height_floor'],cfg['height_epsilon'],cfg['smooth_l1_beta'])<=0 or min(cfg['relative_y_weight'],cfg['log_height_weight'])<0:
        raise ValueError('invalid vertical loss weights or normalization')
    for name in ('global_checkpoint','table_dataset','baseline_run'):
        cfg[name]=str((PROJECT/cfg[name]).resolve())
    corpus=TableCorpus(Path(cfg['table_dataset']));checkpoint=Path(cfg['global_checkpoint']);baseline=Path(cfg['baseline_run'])
    cfg.update(global_sha256=sha(checkpoint),table_annotations=corpus.identity,
               samples_per_epoch=len(corpus.positives('train')),initialization='fresh_MLP_seed42',
               loss='L1+GIoU+0.25*relative_y_SmoothL1+0.25*log_height_SmoothL1',augmentation='none',
               changes='joint_thin_quota_vertical_loss_16x4_pooling',baseline_checkpoint_sha256=sha(baseline/'best.pt'),
               dependencies={name:sha(SRC/name) for name in ('scene/table_mlp_thin.py','scene/table.py','scene/train.py',
                   'scene/table_mlp_control.py','models/localizer.py','models/high_resolution.py',
                   'data_tools/prepare_table.py','data_tools/geometry.py','data_tools/frame_store.py')})
    prior=json.loads((baseline/'config.yaml').read_text())
    if prior['global_sha256']!=cfg['global_sha256'] or prior['table_annotations']!=corpus.identity:
        raise ValueError('baseline uses different B weights or table labels')
    if resume and json.loads((root/'config.json').read_text())!=cfg:
        raise ValueError('thin MLP config, code or dependencies changed')
    root.mkdir(parents=True,exist_ok=True);save(root/'config.json',cfg);save(root/'config.yaml',cfg)
    started=json.loads((root/'task.json').read_text()).get('started',now()) if resume and (root/'task.json').exists() else now()
    status=dict(status='extracting',started=started,updated=now(),run=str(root),config=str(root/'config.json'),
                global_sha256=cfg['global_sha256'],table_pool=cfg['table_pool'],baseline_run=str(baseline))
    save(root/'task.json',status)
    model,info=load_model(checkpoint,device)
    if info['config'].get('global_localizer_type',info['config'].get('localizer_type'))!='high_resolution_se':
        raise ValueError('thin MLP requires the B high-resolution Global locator')
    model.eval().requires_grad_(False)
    cache=root/'features'
    feature_identity=extract(model,checkpoint,cache,device,batch=cfg['feature_batch'],table_dataset=corpus.root,pool=cfg['table_pool'])
    del model
    if device.type=='cuda':
        torch.cuda.empty_cache()
    data={};keys={}
    for part in ('train','val'):
        x,y,k=load_part(cache,part,device);data[part]=(x,y);keys[part]=k
        if x.shape[1:]!=(32,16,4) or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('invalid high-resolution frozen table features')
        previous=json.loads((baseline/'val_predictions.json').read_text()) if part=='val' else None
        if previous is not None and ([(r['video_id'],r['frame_id']) for r in previous]!=k or
                                    not np.array_equal(np.array([r['label_box'] for r in previous],np.float32),y.cpu().numpy())):
            raise ValueError('baseline/new MLP validation rows or targets differ')
    torch.manual_seed(cfg['seed']);head=TableHead(cfg['table_pool']).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    head_init=state_digest(head);history=[];start=0;best=-1.;patience=0;stopped=False
    if resume and (root/'last.pt').exists():
        saved=torch.load(root/'last.pt',map_location=device,weights_only=False)
        if saved['config']!=cfg or saved['feature_identity']!=feature_identity:
            raise ValueError('thin MLP checkpoint dependencies differ')
        head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer'])
        start=saved['epoch'];best=saved['best'];patience=saved['patience'];history=saved['history'];stopped=saved['stopped']
        head_init=saved['initial_head_sha256']
    status.update(status='training',updated=now(),initial_head_sha256=head_init,parameters=sum(p.numel() for p in head.parameters()));save(root/'task.json',status)
    views={r['video_id']:r.get('view','unknown') for r in json.loads((PROJECT/'data/annotations/pp_bounce_confirmed_v1.json').read_text())['videos']}
    x,y=data['train'];vx,vy=data['val']
    for epoch in range(start,cfg['epochs']):
        if stopped:
            break
        started_epoch=time.perf_counter();order,records,audit=quota_order(keys['train'],corpus,cfg,epoch)
        save_jsonl(root/f'sampling/epoch_{epoch+1:03d}.jsonl',records);save(root/f'sampling/epoch_{epoch+1:03d}.json',audit)
        head.train();losses=[];components=defaultdict(list)
        for ix in torch.from_numpy(order).to(device).split(cfg['batch']):
            opt.zero_grad(set_to_none=True);loss,terms=vertical_loss(head(x[ix]),y[ix],cfg)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite thin MLP loss')
            loss.backward();opt.step();losses.append(float(loss.detach()))
            for key,value in terms.items():
                components[key].append(float(value.detach()))
        head.eval();pred=predict(head,vx);iou=box_iou(pred,vy).float().cpu().numpy()
        metric=summarize(iou,keys['val'],corpus,views);vertical=vertical_summary(pred.cpu().numpy(),vy.cpu().numpy(),keys['val'],corpus)
        score=metric['frame_mean_iou'];improved=score>best;best=max(best,score);patience=0 if improved else patience+1
        stopped=patience>=cfg['patience'] or epoch+1==cfg['epochs']
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),loss_components={k:float(np.mean(v)) for k,v in components.items()},
                 val_frame_mean_iou=score,val_video_mean_iou=metric['video_mean_iou'],val_thin_iou=metric['by_bucket']['thin']['mean_iou'],
                 val_vertical=vertical,thin_exposure=audit,elapsed_sec=time.perf_counter()-started_epoch)
        history.append(row)
        state=dict(head=head.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,best=best,patience=patience,stopped=stopped,
                   history=history,config=cfg,feature_identity=feature_identity,initial_head_sha256=head_init)
        atomic_checkpoint(root/'last.pt',state)
        if improved:
            atomic_checkpoint(root/'best.pt',state)
        save(root/'history.json',history);status.update(epoch=epoch+1,best_val_iou=best,latest_val_thin_iou=row['val_thin_iou'],updated=now());save(root/'task.json',status)
        print({k:row[k] for k in ('epoch','loss','loss_components','val_frame_mean_iou','val_video_mean_iou','val_thin_iou','elapsed_sec')},flush=True)
    status.update(status='evaluating',updated=now());save(root/'task.json',status)
    saved=torch.load(root/'best.pt',map_location=device,weights_only=False);head.load_state_dict(saved['head']);head.eval();metrics={}
    for part in ('train','val'):
        a,b=data[part];pred=predict(head,a);iou=box_iou(pred,b).float().cpu().numpy()
        metrics[part]=summarize(iou,keys[part],corpus,views)
        metrics[part]['vertical']=vertical_summary(pred.cpu().numpy(),b.cpu().numpy(),keys[part],corpus)
        if part=='val':
            rows=[dict(video_id=k[0],frame_id=k[1],iou=float(v),predicted_box=p.tolist(),label_box=g.tolist())
                  for k,v,p,g in zip(keys[part],iou,pred.cpu(),b.cpu())]
            save(root/'val_predictions.json',rows);save(root/'error_index.json',[r for r in rows if r['iou']<.5])
            cases={}
            for r in rows:
                if r['video_id'] not in cases or r['iou']<cases[r['video_id']]['iou']:
                    cases[r['video_id']]=r
            for vid,r in cases.items():
                label=corpus.by_key[(vid,r['frame_id'])];res=label['resolution'];t=letterbox_transform(res['width'],res['height'])
                image=corpus.image(vid,r['frame_id'])[:,:,::-1].copy();box=decode_table(r['predicted_box'],t,res)['bbox_xyxy']
                for coords,color in ((label['table_bbox_xyxy'],(0,255,0)),(box,(0,0,255))):
                    cv2.rectangle(image,tuple(round(z) for z in coords[:2]),tuple(round(z) for z in coords[2:]),color,2)
                dest=root/'examples'/f'{vid}_{r["frame_id"]:06d}.jpg';dest.parent.mkdir(exist_ok=True)
                if not cv2.imwrite(str(dest),image):
                    raise OSError(dest)
    metrics.update(epoch=saved['epoch'],checkpoint_sha256=sha(root/'best.pt'),table_annotations=corpus.identity)
    save(root/'metrics.json',metrics)
    old=json.loads((baseline/'metrics.json').read_text());old_rows=json.loads((baseline/'val_predictions.json').read_text())
    old['val']['vertical']=vertical_summary([r['predicted_box'] for r in old_rows],[r['label_box'] for r in old_rows],keys['val'],corpus)
    comparison=dict(baseline=old,thin_mlp=metrics,changes=cfg['changes'],interpretation='combined_recipe_comparison_not_individual_ablations')
    save(root/'comparison.json',comparison)
    lines=['# B MLP: thin quota + vertical loss + 16x4 pooling','','Both use frozen B and identical v1 labels/splits; three changes are combined.','',
           '| Recipe | Best epoch | Val frame IoU | Val video IoU | Thin IoU | Thin center abs error / GT height |',
           '|---|---:|---:|---:|---:|---:|']
    for name,m in (('B 4x4 baseline',old),('B 16x4 + thin quota + vertical loss',metrics)):
        v=m['val'];lines.append(f'| {name} | {m["epoch"]} | {v["frame_mean_iou"]:.4f} | {v["video_mean_iou"]:.4f} | {v["by_bucket"]["thin"]["mean_iou"]:.4f} | {v["vertical"]["thin"]["center_abs_error_gt_height_median"]:.4f} |')
    lines+=['','Thin target: 10% epoch exposure; <=4 uses/frame and <=30% thin slots/video. Source-scene diversity remains limited.','',
            'Best checkpoint is selected by val frame mean IoU, matching baseline; per-epoch video/thin/vertical metrics are retained.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    status.update(status='completed',best_epoch=saved['epoch'],finished=now(),updated=now());save(root/'task.json',status);save(root/'run.json',status)
    print(json.dumps({k:metrics['val'][k] for k in ('frame_mean_iou','video_mean_iou','iou50')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/table/B_mlp_thin.json')
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--device',default='cuda');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    try:
        run(args.run,args.config,torch.device(args.device),args.resume)
    except BaseException as error:
        if args.run.exists():
            save(args.run/'task.json',dict(status='failed',error=repr(error),updated=now()))
        raise
