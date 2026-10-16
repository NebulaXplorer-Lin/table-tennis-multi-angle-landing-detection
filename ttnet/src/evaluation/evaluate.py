from __future__ import annotations

from ttnet.data_tools.frame_store import FrameStore

import argparse
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from ttnet.common import DATA,PROJECT,jsonl,now,save,save_jsonl,sha
from ttnet.models.features import FeatureWindowCache
from ttnet.training.scales import input_size
from ttnet.temporal import temporal_mode,slots_for_mode
from .inference import load_model,predict,prediction_map
from .metrics import accept,aggregate


def overlay(image,row,pred,tolerance=10):
    im=image.copy()
    for b in row['ball_boxes']:
        cv2.rectangle(im,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(60,240,60),2)
    for c in row['ball_centers']:
        cv2.drawMarker(im,tuple(map(round,c)),(60,240,60),cv2.MARKER_CROSS,16,2)
    if pred and pred['valid']:
        cv2.circle(im,(round(pred['x']),round(pred['y'])),9,(60,60,255),2)
    positive=row['label_status']=='positive'
    accepted=bool(pred and pred['valid'])
    error=float(np.linalg.norm(np.asarray(row['ball_centers'][0])-[pred['x'],pred['y']])) if positive and accepted else None
    if row['label_status'] in ['unknown','ignore']:kind=row['label_status']
    elif positive:kind='tp' if accepted and error<=tolerance else ('location_error' if accepted else 'miss')
    else:kind='false_positive' if accepted else 'true_negative'
    t=pred.get('crop_transform') if pred else None
    if t:
        cv2.rectangle(im,(t['left'],t['top']),(t['left']+t['side'],t['top']+t['side']),(255,190,60),1)
    cv2.rectangle(im,(0,0),(im.shape[1],85),(12,22,30),-1)
    conf=f" conf={pred['confidence']:.3f} {pred['source']}" if accepted else ' no accepted prediction'
    label=f"{row['video_id']} frame {row['frame_id']} | {kind}{conf}"
    if error is not None:label+=f" error={error:.2f}px"
    cv2.putText(im,label,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    r=row['resolution']
    cv2.putText(im,f"GREEN: GT | RED: prediction | BLUE: ROI | {r['width']}x{r['height']} {row['nominal_fps']:.2f}fps",
                (10,60),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    return im,kind,error


def baseline_comparison(dataset,metrics):
    # Read-only comparison to the saved, fixed-pilot baseline; never edit YOLO records.
    root=PROJECT/'out/experiments/20261013-212318_baseline_ball-position'
    files=list(root.rglob('summary.json'))+list(root.rglob('metrics.json'))
    source=PROJECT/'experiments/comparison.csv'
    with source.open() as f:
        records=list(csv.DictReader(f))
    selected=[r for r in records if r['experiment_id']=='20261013-212318_baseline_ball-position' and r['eval_id']=='onnx_fp32_ball']
    if len(selected)!=1:raise RuntimeError('historical baseline record not uniquely available')
    reference=float(selected[0]['center_f1'])
    return dict(reference_run='20261013-212318_baseline_ball-position',source_files=[str(p.relative_to(PROJECT)) for p in files]+[str(source.relative_to(PROJECT))],
                source_sha256=sha(source),reference_center_f1_10px=reference,
                note='historical result from experiments/comparison.csv; same held-out video identities; OpenCV vs PyAV decoding and precision differ',
                current_center_f1_10px=metrics['10']['f1'],delta_f1_10px=metrics['10']['f1']-reference)


def evaluate(args):
    run=args.run;out=run/'evaluations'/args.eval_id;out.mkdir(parents=True,exist_ok=False)
    device=torch.device(args.device);torch.set_num_threads(4);cv2.setNumThreads(1)
    global_model,ginfo=load_model(args.global_checkpoint,device)
    meta=json.loads((args.dataset/'metadata/dataset.json').read_text())
    if ginfo['dataset_id']!=meta['dataset_id']:raise ValueError('global model dataset differs')
    local_model,linfo=(load_model(args.local_checkpoint,device) if args.local_checkpoint else (None,None))
    if linfo and (linfo['dataset_id']!=meta['dataset_id'] or linfo['global_checkpoint_sha256']!=sha(args.global_checkpoint)):
        raise ValueError('local model/global checkpoint identity mismatch')
    started=time.perf_counter()
    rows,gc,gf=predict(global_model,args.dataset,'val',device,batch=2,workers=4,amp=args.amp,features=True,
                       input_size=input_size(ginfo.get('config'),'global'),window_half_span_sec=ginfo.get('config',{}).get('window_half_span_sec',1/30))
    gmap=prediction_map(rows,gc,ginfo['threshold'])
    lc,lf=None,None
    if local_model:
        lr,lc,lf=predict(local_model,args.dataset,'val',device,'local',gmap,batch=2,workers=4,amp=args.amp,features=True,
                        input_size=input_size(linfo.get('config'),'local'),crop_side_at_1080=linfo.get('config',{}).get('local_crop_side_1080',256),
                        window_half_span_sec=linfo.get('config',{}).get('window_half_span_sec',1/30))
        if [(r['video_id'],r['frame_id']) for r in rows]!=[(r['video_id'],r['frame_id']) for r in lr]:raise RuntimeError('prediction frame mismatch')
    predictions=[]
    for i,row in enumerate(rows):
        g=gmap[(row['video_id'],row['frame_id'])]
        l=accept(lc[i],linfo['threshold'],row['resolution']) if lc is not None else None
        p=l if l and l['valid'] else g
        if p and lc is not None:p=dict(p,crop_transform=lc[i].get('crop_transform') if lc[i] else None)
        predictions.append(p)
    elapsed=time.perf_counter()-started
    records=[dict(video_id=r['video_id'],frame_id=r['frame_id'],timestamp_sec=r['timestamp_sec'],
                  resolution=r['resolution'],fps=r['fps'],label_status=r['label_status'],ball_centers=r['ball_centers'],
                  global_candidate=gc[i],local_candidate=lc[i] if lc else None,prediction=p) for i,(r,p) in enumerate(zip(rows,predictions))]
    save_jsonl(run/'predictions'/f'{args.eval_id}.jsonl',records)
    metrics={str(t):aggregate(rows,predictions,t) for t in [10,50]}
    global_metrics={str(t):aggregate(rows,[gmap[(r['video_id'],r['frame_id'])] for r in rows],t) for t in [10,50]}
    roi_total=roi_hits=local_tp=local_fp=local_fn=0
    if lc:
        for r,l in zip(rows,lc):
            if r['label_status']!='positive':continue
            roi_total+=1
            if l and l['crop_transform']:
                t=l['crop_transform'];x,y=r['ball_centers'][0]
                inside=t['left']<=x<t['left']+t['side'] and t['top']<=y<t['top']+t['side']
                roi_hits+=int(inside)
                if inside:
                    accepted=accept(l,linfo['threshold'],r['resolution'])
                    one=aggregate([r],[accepted],10)
                    local_tp+=one['tp'];local_fp+=one['fp'];local_fn+=one['fn']
    lp=local_tp/max(1,local_tp+local_fp);lr=local_tp/max(1,local_tp+local_fn)
    conditional=dict(tp=local_tp,fp=local_fp,fn=local_fn,precision=lp,recall=lr,f1=2*lp*lr/max(1e-12,lp+lr)) if lc else None
    by_video=defaultdict(list)
    for i,row in enumerate(rows):by_video[row['video_id']].append(i)
    per_video={k:{str(t):aggregate([rows[i] for i in ids],[predictions[i] for i in ids],t) for t in [10,50]} for k,ids in by_video.items()}
    save(out/'per_video.json',per_video)
    feature_dir=run/'feature_cache'/args.eval_id;feature_dir.mkdir(parents=True,exist_ok=True)
    mode=temporal_mode(ginfo.get('config'))
    feature_mode=mode if mode!='single' else 'fps_mask9'
    cache=FeatureWindowCache(meta['dataset_id'],sha(args.global_checkpoint),sha(args.local_checkpoint) if args.local_checkpoint else None,
                             temporal_mode=feature_mode,window_half_span_sec=ginfo.get('config',{}).get('window_half_span_sec',1/30))
    frame_store=FrameStore(args.dataset)
    feature_checks=[]
    errors,examples=[],[]
    visual=run/'visualizations'/args.eval_id;visual.mkdir(parents=True,exist_ok=True)
    for k,ids in by_video.items():
        cache.reset(k)
        for i in ids:
            r=rows[i];local_valid=bool(lc and lc[i] and accept(lc[i],linfo['threshold'],r['resolution'])['valid'])
            cache.append(r['frame_id'],r['timestamp_sec'],gf[i],lf[i] if lf is not None else None,local_valid,predictions[i],
                         lc[i]['crop_transform'] if lc and lc[i] else None)
        arrays=np.array([f['features'] for f in cache.frames],np.float32)
        np.savez_compressed(feature_dir/f'{k}.npz',features=arrays,
                            timestamps=np.array([f['timestamp_sec'] for f in cache.frames],np.float64),
                            local_valid=np.array([f['local_valid'] for f in cache.frames],bool))
        save_jsonl(feature_dir/f'{k}_localizations.jsonl',[dict(frame_id=f['frame_id'],localization=f['localization'],crop_transform=f['crop_transform']) for f in cache.frames])
        for center in {0,len(ids)//2,len(ids)-1}:
            w=cache.window(center)
            assert w['features'].shape==(1,slots_for_mode(feature_mode),256)
            assert not w['features'][~w['frame_valid']].any()
            assert not w['features'][:,:,128:][~w['local_valid']].any()
            feature_checks.append(dict(video_id=k,center_frame_id=center,valid_frames=int(w['frame_valid'].sum())))
        sample_kinds=set()
        for i in ids:
            r,p=rows[i],predictions[i]
            positive=r['label_status']=='positive'
            accepted=bool(p and p['valid'])
            err=float(np.linalg.norm(np.asarray(r['ball_centers'][0])-[p['x'],p['y']])) if positive and accepted else None
            kind='tp' if positive and accepted and err<=10 else ('location_error' if positive and accepted else ('miss' if positive else r['label_status']))
            needs_error=(positive and kind!='tp') or (r['label_status']=='negative' and accepted)
            sample=kind not in sample_kinds
            if needs_error or sample:
                im=frame_store.row_rgb(r)[:,:,::-1].copy()
                painted,kind,error=overlay(im,r,p)
                dest=visual/k/f"{r['frame_id']:06d}_{kind}.jpg";dest.parent.mkdir(exist_ok=True)
                if not cv2.imwrite(str(dest),painted,[cv2.IMWRITE_JPEG_QUALITY,90]):raise RuntimeError('visualization write failed')
                relative=str(dest.relative_to(run))
                if needs_error:errors.append(dict(video_id=k,frame_id=r['frame_id'],kind=kind,error_px=error,visualization=relative))
                if sample:examples.append(dict(video_id=k,frame_id=r['frame_id'],kind=kind,visualization=relative));sample_kinds.add(kind)
        print(f"evaluated {k}: F1@10={per_video[k]['10']['f1']:.4f}",flush=True)
    save(feature_dir/'identity.json',cache.identity)
    save(out/'feature_interface_checks.json',feature_checks)
    save(out/'errors.json',errors);save(out/'examples.json',examples)
    tiles=''.join(f'<figure><a href="../../../{html.escape(e["visualization"])}"><img loading="lazy" src="../../../{html.escape(e["visualization"])}"></a><figcaption>{html.escape(e["video_id"])} frame {e["frame_id"]}: {e["kind"]}</figcaption></figure>' for e in examples)
    # evaluations/<id>/index.html is two levels below run root.
    tiles=tiles.replace('../../../','../../')
    (out/'index.html').write_text('<meta charset="utf-8"><style>body{background:#111;color:#eee;font:16px sans-serif}figure{display:inline-block;width:440px}img{width:100%}</style><h1>Ball localization: prediction and label</h1><p>Green: label; red: prediction; blue: ROI. Every scored error has an overlay in errors.json.</p>'+tiles)
    summary=dict(status='evaluated',evaluated=now(),dataset_id=meta['dataset_id'],frames=len(rows),val_videos=len(by_video),
                 global_checkpoint_sha256=sha(args.global_checkpoint),local_checkpoint_sha256=sha(args.local_checkpoint) if args.local_checkpoint else None,
                 global_threshold=ginfo['threshold'],local_threshold=linfo['threshold'] if linfo else None,
                 global_input=list(input_size(ginfo.get('config'),'global')),
                 local_input=list(input_size(linfo.get('config'),'local')) if linfo else None,
                 temporal_mode=mode,feature_window_slots=slots_for_mode(feature_mode),
                 protocol='one predicted center, original pixels; 10/50px; unknown/ignore excluded; AP unavailable for heatmaps',
                 precision='PyTorch AMP FP16' if args.amp else 'PyTorch FP32',center=metrics,global_center=global_metrics,
                 roi_coverage=roi_hits/max(1,roi_total) if lc else None,local_conditional_10px=conditional,
                 saved_frame_pipeline_fps=len(rows)/elapsed,saved_frame_pipeline_sec=elapsed,
                 error_count=len(errors),baseline=baseline_comparison(args.dataset,metrics))
    save(out/'metrics.json',summary)
    state=json.loads((run/'run.json').read_text())
    state.update(status='evaluated_pending_export',evaluation=str(out.relative_to(run)));save(run/'run.json',state)
    from ttnet.results import write_results
    write_results()
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=DATA);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--global-checkpoint',type=Path,required=True);p.add_argument('--local-checkpoint',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--amp',action='store_true');p.add_argument('--eval-id',default='best_val')
    evaluate(p.parse_args())


if __name__=='__main__':main()
