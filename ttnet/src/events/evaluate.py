"""Sparse sample evaluation, timing diagnostics, complete videos and PP transfer."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import shutil
import time

import av
import cv2
import numpy as np
import torch

from ttnet.common import PROJECT, jsonl, now, save, save_jsonl, sha
from ttnet.events.data import EventCorpus
from ttnet.events.engine import EventSystem, load_cache
from ttnet.events.inference import predict_video, predict_windows
from ttnet.events.metrics import classification, landing_match, peaks
from ttnet.events.train import infer_cache, validate
from ttnet.evaluation.metrics import aggregate


def error_page(dest,errors,title):
    save_jsonl(dest/'errors.jsonl',errors)
    cards=''.join(f'<figure><a href="{html.escape(e["visualization"])}"><img loading="lazy" src="{html.escape(e["visualization"])}"></a>'
                  f'<figcaption>{html.escape(e["description"])}</figcaption></figure>' for e in errors)
    (dest/'errors.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title>'
         '<style>body{background:#111;color:#eee;font:16px sans-serif}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}'
         'figure{margin:0}img{width:100%}figcaption{padding:8px}</style><h1>'+html.escape(title)+'</h1><main>'+cards+'</main>')


def paint(rgb,vid,fid,description,pred=None,gt=None,probabilities=None,label_note=''):
    im=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    if gt:
        for point in gt:
            cv2.drawMarker(im,(round(point['x']),round(point['y'])),(60,240,60),cv2.MARKER_CROSS,22,2)
    if pred and pred.get('valid',True):cv2.circle(im,(round(pred['x']),round(pred['y'])),12,(60,60,255),2)
    scale=min(1.,960/im.shape[1]);im=cv2.resize(im,None,fx=scale,fy=scale)
    cv2.rectangle(im,(0,0),(im.shape[1],110),(12,18,24),-1)
    lines=[f'{vid} frame {fid} | {description}', 'GREEN: GT | RED: prediction | '+label_note]
    if probabilities is not None:lines.append(f'Bounce={probabilities[0]:.4f} Net={probabilities[1]:.4f}')
    for i,line in enumerate(lines):cv2.putText(im,line,(10,25+i*27),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    return im


def sparse_evaluation(system,corpus,part,dest,thresholds,cache=None):
    dest.mkdir(parents=True,exist_ok=False)
    if cache is None:cache,loc_loss=infer_cache(system,corpus,part)
    else:loc_loss={}
    loss,anchors=validate(system,corpus,cache,part)
    save_jsonl(dest/'anchors.jsonl',anchors)
    metrics=classification(anchors,thresholds)
    metrics.update(event_masked_bce=loss,localization_loss=loc_loss,
                   protocol='classification only at annotated anchors; no full-video event F1 from sparse labels')
    by_video={vid:classification([r for r in anchors if r['video_id']==vid],thresholds) for vid in corpus.split[part]}
    save(dest/'per_video.json',by_video)
    diagnostics=[];errors=[];loc_rows=[];loc_preds=[];availability=[]
    (dest/'images').mkdir()
    for vid in corpus.split[part]:
        features,records=cache[vid];video=corpus.videos[vid]
        availability.extend(records.values())
        samples=[s for s in corpus.samples[part] if s['video_id']==vid]
        dense=predict_windows(system,corpus,vid,features,records,[s['frame_id'] for s in samples])
        dense_map={r['frame_id']:r for r in dense}
        save_jsonl(dest/f'{vid}_localizations.jsonl',[dict(frame_id=fid,**{k:v for k,v in r.items() if k!='frame_id'}) for fid,r in records.items()])
        for row in video['frames'].values():
            loc_rows.append(dict(video_id=vid,frame_id=row['frame_id'],label_status=row['label_status'],
                                 ball_centers=[row['xy']] if row['xy'] else []))
            loc_preds.append(records[row['frame_id']]['prediction'])
        for anchor in [s for s in anchors if s['video_id']==vid]:
            fid=anchor['frame_id'];probs=anchor['probabilities']
            wrong=[('false_negative' if anchor['target'][k]>.5 else 'false_positive')+'_'+kind for k,kind in enumerate(('bounce','net'))
                   if anchor.get('supervised',[True,True])[k] and (probs[k]>=thresholds[k])!=(anchor['target'][k]>.5)]
            for k,kind in enumerate(('bounce','net')):
                if anchor['target'][k]<.5:continue
                nearby=[r for f,r in dense_map.items() if abs(f-fid)<=2]
                peak=max(nearby,key=lambda r:(r['probabilities'][k],-abs(r['frame_id']-fid)))
                times=[e['timestamp_sec'] for e in video['events'] if e['frame_id']==fid and e['kind']==kind]
                error=min(abs(peak['timestamp_sec']-ts) for ts in times)
                passed=peak['probabilities'][k]>=thresholds[k]
                hit=passed and abs(peak['frame_id']-fid)<=1
                diagnostics.append(dict(video_id=vid,kind=kind,anchor=fid,peak_frame=peak['frame_id'],
                                        peak_timestamp_sec=peak['timestamp_sec'],gt_times_sec=times,
                                        error_sec=error,frame_offset=peak['frame_id']-fid,accepted=passed,hit_within_one_frame=hit))
                if not hit:wrong.append('time_or_score_'+kind)
            if wrong:
                description=','.join(wrong)
                ball=video['frames'][fid]['xy']
                image=paint(corpus.image(vid,fid),vid,fid,description,records[fid]['prediction'],
                            [dict(x=ball[0],y=ball[1])] if ball else [],probs,
                            'ball label '+video['frames'][fid]['label_status']+f'; event GT={anchor["target"]}')
                path=f'images/{vid}_{fid:06d}.jpg';cv2.imwrite(str(dest/path),image)
                tiles=[]
                for frame_id in range(fid-2,fid+3):
                    if frame_id not in video['frames']:continue
                    point=video['frames'][frame_id]['xy']
                    tile=paint(corpus.image(vid,frame_id),vid,frame_id,f'event anchor={fid}; GT={anchor["target"]}',
                               records[frame_id]['prediction'],[dict(x=point[0],y=point[1])] if point else [],
                               dense_map[frame_id]['probabilities'] if frame_id in dense_map else None,
                               label_note='sampled-frame ball label; event time may be between images')
                    tiles.append(cv2.resize(tile,(480,270)))
                while len(tiles)<6:tiles.append(np.zeros((270,480,3),np.uint8))
                context=f'images/{vid}_{fid:06d}_context.jpg'
                cv2.imwrite(str(dest/context),np.concatenate([np.concatenate(tiles[:3],axis=1),np.concatenate(tiles[3:6],axis=1)],axis=0))
                errors.append(dict(video_id=vid,frame_id=fid,kinds=wrong,gt_times_sec=anchor['event_times_sec'],
                                   description=f'{vid}:{fid} {description}',visualization=context,center_visualization=path))
        # All localization errors at scored ball frames are also covered.
        for fid,row in video['frames'].items():
            if row['label_status'] not in ('positive','negative'):continue
            p=records[fid]['prediction'];accepted=bool(p and p['valid']);xy=row['xy']
            bad=(bool(xy) and (not accepted or np.hypot(p['x']-xy[0],p['y']-xy[1])>10)) or (not xy and accepted)
            if bad:
                kind='localization_error' if accepted and xy else ('ball_miss' if xy else 'ball_false_positive')
                path=f'images/{vid}_{fid:06d}_ball.jpg'
                im=paint(corpus.image(vid,fid),vid,fid,kind,p,[dict(x=xy[0],y=xy[1])] if xy else [],
                         label_note='ball GT '+row['label_status'])
                cv2.imwrite(str(dest/path),im)
                errors.append(dict(video_id=vid,frame_id=fid,kinds=[kind],description=f'{vid}:{fid} {kind}',visualization=path))
    metrics['localization']={str(t):aggregate(loc_rows,loc_preds,t) for t in (10,50)}
    metrics['feature_availability']=dict(frames=len(availability),
          local_valid_fraction=sum(bool(r['local_valid']) for r in availability)/max(1,len(availability)),
          prediction_valid_fraction=sum(bool(r['prediction'] and r['prediction']['valid']) for r in availability)/max(1,len(availability)),
          note='all cached context frames; distinct from recall on labeled ball frames')
    metrics['local_timing']={kind:dict(events=len(sub),hit_within_one_frame=sum(r['hit_within_one_frame'] for r in sub)/max(1,len(sub)),
                         median_error_ms=float(np.median([r['error_sec'] for r in sub])*1000) if sub else None,
                         note='scan only around known events; not continuous event recall')
                          for kind in ('bounce','net') for sub in [[r for r in diagnostics if r['kind']==kind]]}
    save(dest/'metrics.json',metrics);save_jsonl(dest/'timing.jsonl',diagnostics)
    error_page(dest,errors,f'{part}: all scored classification, timing and localization errors')
    return metrics


def public_evaluation(system,root,dest,thresholds):
    dest.mkdir(parents=True,exist_ok=False);(dest/'images').mkdir()
    manifest=json.loads((root/'manifest.json').read_text());gt=jsonl(root/'gt_public.jsonl')
    all_predictions=[];by_angle=defaultdict(list);errors=[];performance=[];per_video={};diagnostics=Counter()
    for video in manifest['videos']:
        vid=video['video_id'];source=root/video['path']
        rows,records,speed=predict_video(system,source,vid)
        if len(rows)!=video['n_frames']:raise ValueError('manifest/decode frame mismatch')
        performance.append(speed)
        spotted=peaks(rows,thresholds,system.config['nms_sec'])
        bounces=[dict(video_id=vid,frame_id=r['frame_id'],x=r['prediction']['x'],y=r['prediction']['y'],confidence=r['confidence'])
                 for r in spotted if r['kind']=='bounce' and r['prediction'] and r['prediction']['valid']]
        missing=[r for r in spotted if r['kind']=='bounce' and not (r['prediction'] and r['prediction']['valid'])]
        truth=[g for g in gt if g['video_id']==vid]
        match=landing_match(bounces,truth);match['events_without_coordinates']=len(missing)
        per_video[vid]=match;by_angle[video['angle']].append(match);all_predictions.extend(bounces)
        save_jsonl(dest/f'{vid}_frames.jsonl',rows)
        save_jsonl(dest/f'{vid}_events_without_coordinates.jsonl',missing)
        requested=defaultdict(list);valid_truth=[g for g in truth if g.get('valid',True)]
        for index in match['false_positives']:
            p=bounces[index];requested[p['frame_id']].append(('event_false_positive',p))
            close=[g for g in valid_truth if abs(g['frame_id']-p['frame_id'])<=1]
            if close:
                d=min(np.hypot(p['x']-g['x'],p['y']-g['y']) for g in close)
                diagnostics['position_error_at_event_time' if d>50 else 'duplicate_near_matched_gt']+=1
            elif any(abs(g['frame_id']-p['frame_id'])<=3 and np.hypot(p['x']-g['x'],p['y']-g['y'])<=50 for g in valid_truth):
                diagnostics['nearby_time_error_2_to_3_frames']+=1
            else:diagnostics['other_unmatched_event']+=1
            if p['frame_id']<7 or p['frame_id']>=video['n_frames']-7:diagnostics['boundary_false_positives']+=1
        for index in match['false_negatives']:
            g=valid_truth[index];requested[g['frame_id']].append(('event_false_negative',g))
            diagnostics['fn_with_prediction_at_wrong_position' if any(abs(p['frame_id']-g['frame_id'])<=1 for p in bounces)
                        else 'fn_without_event_prediction_within_one_frame']+=1
        for r in missing:requested[r['frame_id']].append(('event_without_coordinate',r))
        with av.open(str(source)) as container:
            for fid,frame in enumerate(container.decode(video=0)):
                if fid not in requested:continue
                kinds=','.join(sorted({kind for kind,_ in requested[fid]}))
                current=[p for p in bounces if abs(p['frame_id']-fid)<=1]
                labels=[g for g in truth if abs(g['frame_id']-fid)<=1 and g.get('valid',True)]
                im=paint(frame.to_ndarray(format='rgb24'),vid,fid,kinds,current[0] if current else records[fid]['prediction'],labels,
                         rows[fid]['probabilities'],label_note='event GT within +/-1 frame')
                path=f'images/{vid}_{fid:06d}.jpg';cv2.imwrite(str(dest/path),im)
                errors.append(dict(video_id=vid,frame_id=fid,kinds=kinds,description=f'{vid}:{fid} {kinds}',visualization=path))
    angle_metrics={}
    for angle,items in by_angle.items():
        tp=sum(m['tp'] for m in items);fp=sum(m['fp'] for m in items);fn=sum(m['fn'] for m in items)
        p=tp/max(1,tp+fp);r=tp/max(1,tp+fn)
        angle_metrics[angle]=dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(1e-12,p+r))
    metric=dict(by_angle=angle_metrics,macro_f1=sum(m['f1'] for m in angle_metrics.values())/len(angle_metrics),
                protocol='transfer only; no tuning. +/-1 frame, 50px, maximum one-to-one valid GT matching; ignored GT absorbs unmatched predictions',
                events_without_coordinates=sum(m['events_without_coordinates'] for m in per_video.values()),
                error_diagnostics=dict(diagnostics),
                e2e_fps=sum(m['frames'] for m in performance)/sum(m['elapsed_sec'] for m in performance))
    save(dest/'metrics.json',metric);save(dest/'per_video.json',per_video);save(dest/'performance.json',performance)
    save_jsonl(dest/'predictions.jsonl',all_predictions);error_page(dest,errors,'PP public transfer: all errors')
    return metric


def head_benchmark(system,dest):
    device=system.device
    f=torch.zeros(1,13,256,device=device);m=torch.zeros(1,13,9,device=device);v=torch.ones(1,13,dtype=torch.bool,device=device)
    samples=[];macs={}
    def count(name,module,inputs):
        total=[0];handles=[]
        def hook(layer,inp,out):
            if isinstance(layer,torch.nn.Linear):total[0]+=out.numel()*layer.in_features
            else:total[0]+=out.numel()*(layer.in_channels//layer.groups)*int(np.prod(layer.kernel_size))
        for layer in module.modules():
            if isinstance(layer,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Linear)):handles.append(layer.register_forward_hook(hook))
        module(*inputs)
        for h in handles:h.remove()
        macs[name]=total[0]
    with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=system.config['amp'] and device.type=='cuda'):
        count('head',system.head,(f,m,v))
        for name,module in [('global',system.global_model),('local',system.local_model)]:
            w,h=system.config[name+'_input']
            count(name,module,(torch.zeros(1,3,3,h,w,device=device),torch.ones(1,3,dtype=torch.bool,device=device),
                               torch.tensor([[-.04,0.,.04]],device=device)))
        for _ in range(50):system.head(f,m,v)
        for _ in range(3):
            for _ in range(200):
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();system.head(f,m,v)
                if device.type=='cuda':torch.cuda.synchronize()
                samples.append((time.perf_counter()-start)*1000)
    stats=dict(parameters={name:sum(p.numel() for p in module.parameters()) for name,module in
                           [('global',system.global_model),('local',system.local_model),('head',system.head)]},
               head_forward_p50_ms=float(np.percentile(samples,50)),head_forward_p95_ms=float(np.percentile(samples,95)),
               input_shapes=[[1,13,256],[1,13,9],[1,13]],batch=1,device=str(device),amp=system.config['amp'],
               hardware=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',warmup=50,measured=600)
    stats.update(macs=macs,flops={k:2*v for k,v in macs.items()},
                 compute_coverage='Partial Conv1d/Conv2d/Linear hooks; 1 MAC=2 FLOPs; excludes functional/fused attention and projections, normalization, activation, pooling, resize and decoding; local assumes one ROI (association uses up to two)')
    save(dest/'benchmark.json',stats);save(dest/'benchmark_samples_ms.json',samples)
    return stats


def evaluate(args):
    state=torch.load(args.run/'checkpoints/best.pt',map_location='cpu',weights_only=False);cfg=state['config']
    torch.set_num_threads(cfg['torch_threads']);cv2.setNumThreads(1)
    system=EventSystem(cfg,torch.device(args.device),initialize=False);system.load_state_dict(state['system']);system.configure(cfg['mode'],False)
    corpus=EventCorpus(Path(cfg['dataset']))
    if cfg.get('best_metric')=='pp_landing_macro_f1':
        # This entry point is the final external/sparse evaluation. Continuous
        # PP selection has its own command and cannot tune on these results.
        from ttnet.events.continuous import attach_pp
        attach_pp(corpus,cfg['pp_annotations'])
        if corpus.identity!=cfg['dataset_id']:raise ValueError('evaluation dataset mismatch')
        pp_ids=set(corpus.pp_annotations)
        for part in ('train','val','test'):
            corpus.split[part]=[v for v in corpus.split[part] if v not in pp_ids]
            corpus.samples[part]=[s for s in corpus.samples[part] if s['video_id'] not in pp_ids]
        for vid in pp_ids:del corpus.videos[vid]
    elif corpus.identity!=cfg['dataset_id']:raise ValueError('evaluation dataset mismatch')
    dest=args.run/'evaluations'/args.eval_id
    if dest.exists() and not args.resume:raise FileExistsError(f'{dest}; use --resume or another --eval-id')
    dest.mkdir(parents=True,exist_ok=True)
    selection=json.loads((args.run/'best_selection.json').read_text());thresholds=selection['thresholds']
    ident=dict(checkpoint_sha256=sha(args.run/'checkpoints/best.pt'),dataset_id=corpus.identity,
               thresholds=thresholds,best_epoch=state['epoch'],protocol_sha256=sha(__file__),started=now())
    if (dest/'identity.json').exists():
        previous=json.loads((dest/'identity.json').read_text())
        if any(previous[k]!=ident[k] for k in ('checkpoint_sha256','dataset_id','thresholds','best_epoch')):
            raise ValueError('evaluation resume identity mismatch')
    else:save(dest/'identity.json',ident)
    attempts=dest/'attempts';attempts.mkdir(exist_ok=True)
    save(attempts/f'{len(list(attempts.iterdir()))+1:03d}.json',ident)
    cached=None
    if cfg['mode']=='frozen':
        from ttnet.events.engine import build_cache,cache_identity,feature_cache_path
        if args.cache is None:args.cache=feature_cache_path(system,corpus)
        build_cache(system,corpus,args.cache,parts=('val',))
        cached=load_cache(corpus,args.cache,parts=('val',),expected=cache_identity(system,corpus))
    metrics={}
    for part in ('val','test'):
        path=dest/part
        if (path/'metrics.json').exists():metrics[part]=json.loads((path/'metrics.json').read_text());continue
        if path.exists():path.rename(dest/f'{part}_partial_{len(list(dest.glob(part+"_partial_*")))+1:03d}')
        metrics[part]=sparse_evaluation(system,corpus,part,path,thresholds,cached if part=='val' else None)
    save(dest/'metrics_progress.json',metrics)
    if (dest/'public/metrics.json').exists():metrics['public']=json.loads((dest/'public/metrics.json').read_text())
    else:
        path=dest/'public'
        if path.exists():path.rename(dest/f'public_partial_{len(list(dest.glob("public_partial_*")))+1:03d}')
        metrics['public']=public_evaluation(system,args.public,path,thresholds)
    save(dest/'metrics_progress.json',metrics)
    # The entire held-out match is inferred, but unreviewed timestamps are not counted as FP.
    full=[]
    for vid in corpus.split['val']:
        path=dest/f'{vid}_complete_predictions.jsonl'
        if args.resume and path.exists() and (dest/f'{vid}_unscored_events.jsonl').exists():
            rows=jsonl(path)
            if len(rows)!=corpus.videos[vid]['meta']['n_frames']:raise ValueError('incomplete resumed full-video result')
            full.append(dict(frames=len(rows),elapsed_sec=None,e2e_fps=None,
                             note='reused successful complete-video predictions from previous evaluation attempt; timing was not persisted'))
            continue
        rows,_,speed=predict_video(system,corpus.videos[vid]['meta']['source'],vid)
        save_jsonl(dest/f'{vid}_complete_predictions.jsonl',rows)
        save_jsonl(dest/f'{vid}_unscored_events.jsonl',peaks(rows,thresholds,cfg['nms_sec']))
        full.append(speed)
    metrics['full_validation_inference']=full
    metrics['benchmark']=head_benchmark(system,dest)
    save(dest/'metrics.json',metrics)
    save(args.run/'run.json',dict(status='completed',stage='event',mode=cfg['mode'],best_epoch=state['epoch'],
                                 checkpoint_sha256=sha(args.run/'checkpoints/best.pt'),eval_id=args.eval_id,finished=now()))
    from ttnet.events.report import publish
    publish(args.run)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--eval-id',default='best');p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--cache',type=Path)
    p.add_argument('--public',type=Path,default=PROJECT/'data/pp_public_data')
    a=p.parse_args()
    try:evaluate(a)
    except Exception as exc:
        save(a.run/'evaluation_failure.json',dict(error=repr(exc),finished=now()))
        if (a.run/'run.json').exists():
            status=json.loads((a.run/'run.json').read_text());status.update(status='trained_pending_eval',evaluation_error=repr(exc),updated=now())
            save(a.run/'run.json',status)
        raise


if __name__=='__main__':main()
