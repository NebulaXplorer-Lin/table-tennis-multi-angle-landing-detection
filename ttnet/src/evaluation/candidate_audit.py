"""Quick, no-training candidate-recall diagnostic. GT is used only to measure ceilings."""
from __future__ import annotations

from ttnet.data_tools.frame_store import FrameStore

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch

from ttnet.common import DATA,ROOT,OUTPUT,jsonl,now,save,save_jsonl,sha
from ttnet.data_tools.geometry import crop_transform,letterbox,map_point
from ttnet.training.dataset import normalize,read_rgb


from ttnet.tracking.candidates import candidates

def summarize(records):
    tables={}
    for threshold in [.7,.1,.01]:
        rows=[]
        for k in [1,3,5]:
            hit10=hit50=roi=0
            for r in records:
                selected=[p for p in r['candidates'][:k] if p['confidence']>=threshold]
                hit10+=any(p['distance_px']<=10 for p in selected)
                hit50+=any(p['distance_px']<=50 for p in selected)
                roi+=any(p['roi_contains_gt'] for p in selected)
            rows.append(dict(k=k,frames=len(records),hit10=hit10,hit50=hit50,roi_hits=roi,
                             coverage_10px=hit10/max(1,len(records)),coverage_50px=hit50/max(1,len(records)),
                             roi_coverage=roi/max(1,len(records))))
        tables[str(threshold)]=rows
    base=tables['0.7'][0]
    base_misses=len(records)-base['hit10']
    failure_modes={}
    for threshold in [.7,.1,.01]:
        recovered=sum(not any(p['confidence']>=.7 and p['distance_px']<=10 for p in r['candidates'][:1])
                      and any(p['confidence']>=threshold and p['distance_px']<=10 for p in r['candidates'][:5]) for r in records)
        failure_modes[str(threshold)]=dict(top1_misses=base_misses,recoverable_by_top5_oracle=recovered,
                                          recoverable_fraction_of_top1_misses=recovered/max(1,base_misses))
    ranks={str(k):sum(next((i+1 for i,p in enumerate(r['candidates']) if p['distance_px']<=10),0)==k for r in records) for k in range(6)}
    return dict(threshold_tables=tables,recovery_ceiling=failure_modes,true_ball_first_rank_10px=ranks)


def in_roi(crop, xy):
    return bool(crop and crop['left']<=xy[0]<crop['left']+crop['side'] and crop['top']<=xy[1]<crop['top']+crop['side'])


def classify_error(row, record, tolerance=10):
    p=record['prediction'];accepted=bool(p and p['valid'])
    if row['label_status'] not in ('positive','negative'):return []
    if row['label_status']=='negative':return ['false_positive'] if accepted else []
    xy=row['ball_centers'][0]
    if accepted and np.hypot(p['x']-xy[0],p['y']-xy[1])<=tolerance:return []
    peaks=record.get('global_candidates',[]);regions=record.get('roi_transforms',[])
    present=any(np.hypot(c['x']-xy[0],c['y']-xy[1])<=tolerance for c in peaks)
    roi=any(in_roi(c,xy) for c in regions)
    return [kind for kind,flag in [('candidate_absent',not present),('ranking_or_selection',present),
                                  ('search_region_wrong',not roi),('local_error_in_correct_roi',roi)] if flag]


def overlay(rgb, row, record, kinds):
    im=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for i,p in enumerate(record.get('global_candidates',[])):
        xy=(round(p['x']),round(p['y']));cv2.circle(im,xy,7,(255,200,0),1)
        cv2.putText(im,str(i+1),xy,cv2.FONT_HERSHEY_SIMPLEX,.5,(255,200,0),1)
    for crop in record.get('roi_transforms',[]):
        l,t,s=crop['left'],crop['top'],crop['side'];cv2.rectangle(im,(l,t),(l+s,t+s),(255,80,180),1)
    for xy in row['ball_centers']:cv2.drawMarker(im,tuple(round(v) for v in xy),(0,255,0),cv2.MARKER_CROSS,18,2)
    p=record['prediction']
    if p and p['valid']:cv2.circle(im,(round(p['x']),round(p['y'])),11,(0,0,255),2)
    text=f"{row['video_id']}:{row['frame_id']} GT={row['label_status']} conf={p['confidence'] if p else 0:.3f} {' '.join(kinds)}"
    # Text occupies a separate strip: no scoring errors are hidden by the legend.
    im=cv2.copyMakeBorder(im,55,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    for y,line in [(20,text),(43,'GREEN GT | RED observation | CYAN candidates | PINK ROI')]:
        cv2.putText(im,line,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    return im


def audit(config, dataset, dest, device='cpu', video_ids=None):
    from collections import Counter
    from ttnet.events.engine import EventSystem
    from ttnet.evaluation.ball_corpus import BallCorpus
    from ttnet.evaluation.metrics import aggregate
    from ttnet.common import identity
    cfg=dict(config,candidate_diagnostics=True)
    if cfg['temporal_mode']!='fixed3':raise ValueError('candidate audit requires current fixed3 localization')
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(cfg.get('torch_threads',4));cv2.setNumThreads(1)
    system=EventSystem(cfg,torch.device(device));system.configure('frozen',False)
    corpus=BallCorpus(dataset)
    for path in (cfg['global_checkpoint'],cfg['local_checkpoint']):
        trained=torch.load(path,map_location='cpu',weights_only=False)['config']
        if trained.get('split_hashes')!=corpus.metadata['split_hashes']:
            raise ValueError('locator checkpoint and frozen PP split differ')
    all_records=[];per_video={};error_index=[];timings=[]
    selected=video_ids or corpus.split['val']
    if set(selected)-set(corpus.split['val']):raise ValueError('audit only accepts frozen PP validation videos')
    started=time.perf_counter()
    for vid in selected:
        system.reset_tracking(vid);rows=corpus.rows[vid];records=[];predictions=[];counts=Counter();writer=None
        path=dest/f'{vid}_overlay.mp4'
        if system.device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        try:
            with torch.inference_mode(),torch.autocast(device_type=system.device.type,enabled=cfg['amp'] and system.device.type=='cuda'):
                for row in rows:
                    fid=row['frame_id'];_,result,_=system.forward_frames(corpus,vid,[fid]);r=result[fid]
                    kinds=classify_error(row,r);counts.update(kinds)
                    if kinds:error_index.append(dict(video_id=vid,frame_id=fid,kinds=kinds,visualization=path.name,timestamp_sec=row['timestamp_sec']))
                    pred=r['prediction'];predictions.append(dict(pred,crop_transform=r['crop_transform']) if pred else None)
                    observed=dict(video_id=vid,frame_id=fid,timestamp_sec=row['timestamp_sec'],label=row,**{k:v for k,v in r.items() if k!='frame_id'})
                    records.append(observed)
                    if row['label_status']=='positive':
                        gt=row['ball_centers'][0];peaks=[]
                        for c in r['global_candidates']:
                            crop=crop_transform(c['x'],c['y'],row['resolution']['width'],row['resolution']['height'],cfg['local_input'][0],cfg['local_crop_side_1080'])
                            peaks.append(dict(c,distance_px=float(np.hypot(c['x']-gt[0],c['y']-gt[1])),roi_contains_gt=in_roi(crop,gt)))
                        all_records.append(dict(video_id=vid,frame_id=fid,candidates=peaks))
                    im=overlay(corpus.image(vid,fid),row,r,kinds)
                    if writer is None:
                        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),corpus.videos[vid]['meta']['fps'],(im.shape[1],im.shape[0]))
                        if not writer.isOpened():raise OSError(f'cannot encode {path}')
                    writer.write(im)
        finally:
            if writer is not None:writer.release()
        if system.device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        timings.append(dict(video_id=vid,frames=len(rows),elapsed_sec=elapsed,fps=len(rows)/elapsed,
                            protocol='array read + locators + candidates + scoring + overlay encoding; not inference-only FPS'))
        save_jsonl(dest/f'{vid}_predictions.jsonl',records)
        per_video[vid]=dict(candidate=summarize([r for r in all_records if r['video_id']==vid]),
                            localization={str(t):aggregate(rows,predictions,t) for t in (10,50)},failure_modes=dict(counts),overlay=path.name)
        print(f'audited {vid}: {len(rows)} frames',flush=True)
    summary=dict(status='completed' if set(selected)==set(corpus.split['val']) else 'partial_video_subset',
        dataset_id=corpus.identity,config=cfg,config_hash=identity(cfg),global_sha256=sha(cfg['global_checkpoint']),local_sha256=sha(cfg['local_checkpoint']),
        implementation_sha256=sha(__file__),backend='PyTorch',device=device,elapsed_sec=time.perf_counter()-started,
        videos=len(selected),scored_positive_frames=len(all_records),candidate=summarize(all_records),
        failure_modes=dict(sum((Counter(v['failure_modes']) for v in per_video.values()),Counter())),
        error_frames=len(error_index),interpretation='GT only measures candidate ceilings and errors; categories overlap; all scored errors have full video overlays')
    save(dest/'summary.json',summary);save(dest/'per_video.json',per_video);save(dest/'timings.json',timings)
    save_jsonl(dest/'error_index.jsonl',error_index);save_jsonl(dest/'candidate_samples.jsonl',all_records)
    links=''.join(f'<li><a href="{v["overlay"]}">{vid}</a></li>' for vid,v in per_video.items())
    (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Fixed3 candidate audit</h1><p>All scored errors are covered in the full videos.</p><ul>'+links+'</ul>')
    return summary


def main():
    from ttnet.configuration import load_config
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DATA)
    p.add_argument('--config',type=Path,default=Path('configs/experiments/event_comparison_A.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--video',action='append',help='explicit subset smoke check; omit for full validation')
    args=p.parse_args();cfg,_=load_config(args.config)
    result=audit(cfg,args.dataset,args.output,args.device,args.video)
    print(json.dumps({k:result[k] for k in ('status','videos','scored_positive_frames','error_frames')},indent=2))


if __name__=='__main__':main()
