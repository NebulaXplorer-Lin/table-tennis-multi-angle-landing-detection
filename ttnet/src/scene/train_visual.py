"""Prepare table experiments by default; training requires the explicit --train flag."""
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
from torch.utils.data import DataLoader

from ttnet.common import PROJECT, SRC, now, save, save_jsonl, sha
from ttnet.data_tools.prepare_table import TableCorpus, TableDataset, collate_table, sample_epoch, is_thin
from ttnet.evaluation.inference import load_model
from ttnet.models.table_encoder import IndependentTableEncoder, SharedGlobalTableHead, select_table
from ttnet.scene.table import box_iou


def load_recipe(path):
    cfg = json.loads(Path(path).read_text())
    required = {'schema','source','dataset','pretrained','global_checkpoint','input','channels','seed','epochs',
                'patience','micro_batch','effective_batch','workers','samples_per_epoch','thin_weight',
                'encoder_lr','head_lr','weight_decay','warmup_epochs','amp','confidence','nms_iou','augmentation'}
    if set(cfg) != required or cfg['schema'] != 'table_visual_recipe_v1':
        raise ValueError('invalid table recipe fields/schema')
    if cfg['source'] not in ('independent_yolo','shared_global'):
        raise ValueError('unsupported table source')
    for key in ('epochs','patience','micro_batch','effective_batch','samples_per_epoch'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'invalid {key}')
    if cfg['effective_batch']%cfg['micro_batch'] or cfg['workers'] < 0 or not 1 <= cfg['thin_weight'] <= 3:
        raise ValueError('invalid batch/sampling/workers')
    if len(cfg['input']) != 2 or any(type(i) is not int or i < 32 or i%32 for i in cfg['input']):
        raise ValueError('table input must be positive multiples of 32')
    if not 0 <= cfg['confidence'] <= 1 or not 0 < cfg['nms_iou'] <= 1:
        raise ValueError('invalid detection thresholds')
    a = cfg['augmentation']
    if set(a) != {'horizontal_flip','translate','scale'} or not 0 <= a['horizontal_flip'] <= 1 or not 0 <= a['translate'] <= .05 or not 0 < a['scale'][0] <= a['scale'][1]:
        raise ValueError('invalid explicit table augmentation')
    for key in ('dataset','pretrained','global_checkpoint'):
        if cfg[key] is not None:
            p = Path(cfg[key]); cfg[key] = str(p if p.is_absolute() else PROJECT/p)
    if cfg['source'] == 'shared_global' and (not cfg['global_checkpoint'] or cfg['pretrained'] is not None or cfg['channels'] != 32):
        raise ValueError('shared B route needs frozen Global and native32 channels')
    if cfg['source'] == 'independent_yolo' and (not cfg['pretrained'] or cfg['global_checkpoint'] is not None or cfg['input'] != [640,640]):
        raise ValueError('independent route requires explicit pretrained weights and 640 input')
    return cfg


def construct(cfg, device, restore=None):
    global_model = None
    if cfg['source'] == 'shared_global':
        global_model, info = load_model(Path(cfg['global_checkpoint']),device)
        if info.get('stage') != 'global' or info['config'].get('global_localizer_type',info['config'].get('localizer_type')) != 'high_resolution_se':
            raise ValueError('shared_B recipe requires a high_resolution_se Global checkpoint')
        if list(info['config']['global_input']) != cfg['input']:
            raise ValueError('shared Global input differs from the saved locator recipe')
        global_model.eval().requires_grad_(False)
        model = SharedGlobalTableHead(cfg['channels']).to(device)
    else:
        model = IndependentTableEncoder(cfg['pretrained'] if restore is None else None,
                                        None if restore is None else restore['architecture']).to(device)
        if model.channels != cfg['channels']:
            raise ValueError('table feature channel contract differs from instantiated model')
    if restore is not None:
        model.load_state_dict(restore['model'])
    return model, global_model


def forward(model, global_model, batch, device):
    images = batch['images'].to(device)
    if global_model is not None:
        with torch.no_grad():
            _,_,spatial = global_model(images,batch['frame_valid'].to(device),batch['offsets'].to(device),return_spatial=True)
        return model(spatial)
    return model(images)


def parameter_groups(model, cfg):
    detector_ids = set() if cfg['source']=='shared_global' else {id(p) for p in model.detector.model[-1].parameters()}
    buckets = defaultdict(list)
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        head = cfg['source']=='shared_global' or id(p) in detector_ids
        decay = p.ndim > 1
        buckets[(head,decay)].append(p)
    return [dict(params=values,lr=cfg['head_lr'] if head else cfg['encoder_lr'],
                 initial_lr=cfg['head_lr'] if head else cfg['encoder_lr'],
                 weight_decay=cfg['weight_decay'] if decay else 0.,name=('head' if head else 'encoder_neck')+('_decay' if decay else '_no_decay'))
            for (head,decay),values in buckets.items()]


def loader(corpus, rows, cfg, epoch=0, training=False):
    return DataLoader(TableDataset(corpus,rows,cfg,epoch,training),batch_size=cfg['micro_batch'],
                      shuffle=False,num_workers=cfg['workers'],collate_fn=collate_table,pin_memory=False)


def evaluate(model, global_model, corpus, cfg, device, output=None, overlay_mode='all', part='val'):
    if part not in ('train','val') or overlay_mode not in ('all','worst_per_video'):
        raise ValueError('invalid table evaluation split or overlay mode')
    model.eval()
    rows, buckets, edge_errors = [], defaultdict(list), []
    views_path = PROJECT/'data/annotations/pp_bounce_confirmed_v1.json'
    views = {r['video_id']:r.get('view','unknown') for r in json.loads(views_path.read_text())['videos']} if views_path.exists() else {}
    with torch.inference_mode():
        for step,batch in enumerate(loader(corpus,corpus.rows[part],cfg)):
            if step%128==0:
                print(f'{part} evaluation {step*cfg["micro_batch"]}/{len(corpus.rows[part])}',flush=True)
            with torch.autocast(device_type=device.type,enabled=cfg['amp'] and device.type=='cuda'):
                prediction,_ = forward(model,global_model,batch,device)
            kwargs = dict(confidence=cfg['confidence'],nms_iou=cfg['nms_iou'])
            if global_model is not None:
                kwargs['input_size'] = cfg['input']
            candidates = model.candidates(prediction,**kwargs)
            for r,t,c in zip(batch['row'],batch['transform'],candidates):
                predicted = select_table(c,t)
                gt = r['table_bbox_xyxy']
                item = dict(video_id=r['video_id'],frame_id=r['frame_id'],timestamp_sec=r['timestamp_sec'],
                            prediction=predicted,target=gt,iou=None)
                if gt is not None:
                    iou = float(box_iou(torch.tensor([predicted['bbox_xyxy']]),torch.tensor([gt]))[0]) if predicted['valid'] else 0.
                    item['iou'] = iou
                    buckets[('video',r['video_id'])].append(item)
                    buckets[('group',corpus.groups[r['video_id']])].append(item)
                    buckets[('view',views.get(r['video_id'],'unknown'))].append(item)
                    if is_thin(r):
                        buckets[('thin','all')].append(item)
                    if predicted['valid']:
                        errors = np.abs(np.array(predicted['bbox_xyxy'])-gt)
                        edge_errors.append(errors)
                        item['edge_errors_px'] = errors.tolist()
                        item['vertical_errors_gt_height'] = (errors[[1,3]]/(gt[3]-gt[1])).tolist()
                rows.append(item)
    def summarize(items):
        if not items:
            return dict(frames=0,mean_iou=None)
        values = np.array([r['iou'] for r in items])
        success = [r for r in items if r['prediction']['valid']]
        edge = np.array([r['edge_errors_px'] for r in success])
        vertical = np.array([r['vertical_errors_gt_height'] for r in success])
        return dict(frames=len(items),mean_iou=float(values.mean()),iou50=float((values>=.5).mean()),
                    iou75=float((values>=.75).mean()),missing=len(items)-len(success),coverage=len(success)/len(items),
                    edge_median_px=np.median(edge,axis=0).tolist() if success else None,
                    edge_p90_px=np.percentile(edge,90,axis=0).tolist() if success else None,
                    vertical_median_gt_height=np.median(vertical,axis=0).tolist() if success else None,
                    vertical_p90_gt_height=np.percentile(vertical,90,axis=0).tolist() if success else None)
    metrics = summarize([r for r in rows if r['iou'] is not None])
    metrics['video_mean_iou'] = float(np.mean([np.mean([r['iou'] for r in values]) for (kind,_),values in buckets.items() if kind=='video']))
    metrics['by_bucket'] = {f'{k}:{v}':summarize(items) for (k,v),items in sorted(buckets.items())}
    metrics['inferred_frames'] = len(rows)
    if output is not None:
        output = Path(output);output.mkdir(parents=True,exist_ok=True)
        save(output/'metrics.json',metrics); save_jsonl(output/'predictions.jsonl',rows)
        errors = [r for r in rows if r['iou'] is not None and r['iou'] < .5]
        save(output/'errors.json',errors)
        overlays=errors
        if overlay_mode=='worst_per_video':
            cases={}
            for item in errors:
                vid=item['video_id']
                if vid not in cases or item['iou']<cases[vid]['iou']:
                    cases[vid]=item
            overlays=list(cases.values())
        save(output/'overlay_manifest.json',dict(mode=overlay_mode,frames=[dict(video_id=r['video_id'],frame_id=r['frame_id'],iou=r['iou']) for r in overlays]))
        # All failures stay indexed; selected overlays use original-frame pixels.
        for item in overlays:
            image = corpus.image(item['video_id'],item['frame_id']).copy()[:,:,::-1].copy()
            for box,color in ((item['target'],(0,255,0)),(item['prediction']['bbox_xyxy'],(0,0,255))):
                if box:
                    cv2.rectangle(image,tuple(round(x) for x in box[:2]),tuple(round(x) for x in box[2:]),color,2)
            path = output/'errors'/item['video_id']/f'{item["frame_id"]:06d}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(path),image):
                raise OSError(path)
        (output/'index.html').write_text('<meta charset="utf-8"><h1>Table errors: green GT, red prediction</h1>'+''.join(
            f'<p>{r["video_id"]}:{r["frame_id"]} IoU={r["iou"]:.3f}</p><img loading="lazy" width="800" src="errors/{r["video_id"]}/{r["frame_id"]:06d}.jpg">' for r in overlays))
    return metrics


def train(cfg, corpus, run, device, resume=False):
    """Called only behind --train. No preparation command calls this function."""
    from ttnet.events.train import atomic_checkpoint, rng_state, set_rng
    if run.exists() and not resume:
        raise FileExistsError(run)
    if resume and not (run/'last.pt').is_file():
        raise FileNotFoundError(run/'last.pt')
    state = torch.load(run/'last.pt',map_location='cpu',weights_only=False) if resume else None
    hashes = dict(table_annotations=corpus.identity,global_sha256=sha(cfg['global_checkpoint']) if cfg['global_checkpoint'] else None,
                  pretrained_sha256=sha(cfg['pretrained']) if cfg['pretrained'] else None,
                  implementation={name:sha(SRC/name) for name in ('scene/train_visual.py','models/table_encoder.py','data_tools/prepare_table.py')})
    if state and (state['config']!=cfg or state['dependencies']!=hashes):
        raise ValueError('resume recipe or dataset/weights changed')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    model,global_model = construct(cfg,device,state)
    optimizer = torch.optim.AdamW(parameter_groups(model,cfg))
    scaler = torch.amp.GradScaler('cuda',enabled=cfg['amp'] and device.type=='cuda')
    run.mkdir(parents=True,exist_ok=True)
    save(run/'config.json',cfg); save(run/'dependencies.json',hashes)
    save(run/'parameter_groups.json',[{k:v for k,v in g.items() if k!='params'}|dict(parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups])
    start,best,patience,history = 0,-1.,0,[]
    if state:
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);set_rng(state['rng'])
        start,best,patience,history = state['epoch'],state['best'],state['patience'],state['history']
    def checkpoint_state(epoch):
        return dict(schema='table_visual_v1',config=cfg,model=model.state_dict(),architecture=getattr(model,'architecture',None),
                    global_sha256=hashes['global_sha256'],dependencies=hashes,epoch=epoch,best=best,patience=patience,
                    history=history,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state())
    if not (run/'last.pt').exists():
        atomic_checkpoint(run/'last.pt',checkpoint_state(start))
    save(run/'run.json',dict(status='training',epoch=start+1,step=0,updated=now()))
    for epoch in range(start,cfg['epochs']):
        selected = sample_epoch(corpus.positives('train'),corpus.groups,cfg['seed'],epoch,cfg['samples_per_epoch'],cfg['thin_weight'])
        save_jsonl(run/f'sampling/epoch_{epoch+1:03d}.jsonl',[dict(video_id=r['video_id'],frame_id=r['frame_id']) for r in selected])
        batches = loader(corpus,selected,cfg,epoch,True)
        model.train();optimizer.zero_grad(set_to_none=True);losses=[];started=time.perf_counter()
        accumulation = cfg['effective_batch']//cfg['micro_batch']
        for step,batch in enumerate(batches):
            group_start = (step//accumulation)*accumulation
            denominator = min(accumulation,len(batches)-group_start)
            progress = epoch+(step+1)/len(batches)
            factor = min(1.,progress/max(cfg['warmup_epochs'],1e-9))*.5*(1+math.cos(math.pi*progress/cfg['epochs']))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr']*factor
            with torch.autocast(device_type=device.type,enabled=cfg['amp'] and device.type=='cuda'):
                prediction,_ = forward(model,global_model,batch,device)
                loss,parts = model.loss(prediction,batch['target'].to(device))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite table loss')
            scaler.scale(loss/denominator).backward()
            if (step+1)%accumulation==0 or step+1==len(batches):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if step%50==0 or step+1==len(batches):
                progress_record = dict(status='training',epoch=epoch+1,step=step+1,steps=len(batches),
                                       loss=losses[-1],optimizer_steps=(step+1)//accumulation,
                                       elapsed_sec=time.perf_counter()-started,updated=now())
                save(run/'run.json',progress_record);print(progress_record,flush=True)
        save(run/'run.json',dict(status='validating',epoch=epoch+1,updated=now()))
        metrics = evaluate(model,global_model,corpus,cfg,device)
        score = metrics['video_mean_iou'];improved = score > best
        best = max(best,score);patience = 0 if improved else patience+1
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),video_mean_iou=score,elapsed_sec=time.perf_counter()-started))
        checkpoint = checkpoint_state(epoch+1)
        atomic_checkpoint(run/'last.pt',checkpoint)
        if improved:
            atomic_checkpoint(run/'best.pt',checkpoint)
        save(run/'history.json',history);save(run/'run.json',dict(status='training',epoch=epoch+1,best_video_iou=best))
        print(history[-1],flush=True)
        if patience>=cfg['patience']:
            break
    model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=False)['model'])
    evaluate(model,global_model,corpus,cfg,device,run/'evaluation')
    save(run/'run.json',dict(status='trained_evaluated',best_video_iou=best))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--run',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--train',action='store_true',help='explicit opt-in to optimizer updates')
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    if args.resume and not args.train:
        p.error('--resume requires --train')
    cfg = load_recipe(args.config);corpus = TableCorpus(cfg['dataset'])
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if args.train:
        if args.run is None:
            p.error('--train requires --run')
        try:
            train(cfg,corpus,args.run,torch.device(args.device),args.resume)
        except BaseException as error:
            if args.run.exists():
                save(args.run/'run.json',dict(status='failed',error=repr(error),updated=now()))
            raise
    else:
        report = dict(status='prepared_no_training',config=cfg,table_annotations=corpus.identity,stats=corpus.audit(),
                      required_weights={k:dict(path=cfg[k],available=Path(cfg[k]).is_file()) for k in ('pretrained','global_checkpoint') if cfg[k]},
                      recipe_sha256=sha(args.config))
        if args.output:
            if args.output.exists():
                raise FileExistsError(args.output)
            save(args.output,report)
        print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
