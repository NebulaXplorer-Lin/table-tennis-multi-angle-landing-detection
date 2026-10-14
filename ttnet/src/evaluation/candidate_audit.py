"""Quick, no-training candidate-recall diagnostic. GT is used only to measure ceilings."""
from __future__ import annotations

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
import onnxruntime as ort

from ttnet.common import DATA,ROOT,jsonl,now,save,save_jsonl,sha
from ttnet.data_tools.geometry import crop_transform,letterbox,map_point
from ttnet.training.dataset import normalize,read_rgb


def candidates(heat,transform,resolution,count=5,min_separation=6):
    heat=np.asarray(heat).squeeze()
    maximum=cv2.dilate(heat,np.ones((3,3),np.uint8))
    ys,xs=np.nonzero(heat==maximum)
    order=np.argsort(-heat[ys,xs],kind='stable')
    chosen=[];anchors=[]
    for idx in order:
        iy,ix=int(ys[idx]),int(xs[idx])
        if any((ix-px)**2+(iy-py)**2<min_separation**2 for px,py in anchors):continue
        y1,y2,x1,x2=max(0,iy-3),min(heat.shape[0],iy+4),max(0,ix-3),min(heat.shape[1],ix+4)
        patch=heat[y1:y2,x1:x2].astype(np.float64)
        weights=np.maximum(patch-patch.min(),0);total=weights.sum()
        if total>1e-12:
            xx,yy=np.meshgrid(np.arange(x1,x2),np.arange(y1,y2))
            x,y=float((xx*weights).sum()/total)*2,float((yy*weights).sum()/total)*2
        else:x,y=ix*2.,iy*2.
        x,y=map_point(x,y,transform,inverse=True)
        if not (0<=x<resolution['width'] and 0<=y<resolution['height']):continue
        anchors.append((ix,iy))
        chosen.append(dict(x=x,y=y,confidence=float(heat[iy,ix])))
        if len(chosen)==count:break
    return chosen


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


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=DATA)
    p.add_argument('--model-run',type=Path,default=ROOT/'runs/20261013-234947_global_ball')
    p.add_argument('--per-video',type=int,default=24,help='0 evaluates all labelled val frames')
    p.add_argument('--batch',type=int,default=8);args=p.parse_args()
    cv2.setNumThreads(1)
    export=json.loads((args.model_run/'exports/manifest.json').read_text())
    config=json.loads((args.model_run/'config.yaml').read_text());w,h=config['global_input']
    model=args.model_run/'exports/global.onnx'
    if sha(model)!=export['models'][0]['onnx_sha256']:raise ValueError('ONNX identity changed')
    dm=json.loads((args.dataset/'metadata/dataset.json').read_text())
    if dm['split_hashes']!=config['split_hashes']:raise ValueError('split identity changed')
    grouped=defaultdict(list)
    for r in jsonl(args.dataset/'labels/val.jsonl'):
        if r['label_status']=='positive':grouped[r['video_id']].append(r)
    samples=[]
    for vid,rows in sorted(grouped.items()):
        n=min(len(rows),args.per_video) if args.per_video else len(rows)
        samples.extend(rows[i] for i in np.linspace(0,len(rows)-1,n).round().astype(int))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
    def prepare(row):
        rgb,t=letterbox(read_rgb(args.dataset/row['frame_path']),w,h)
        return normalize(rgb),t
    records=[];start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as workers:
        for pos in range(0,len(samples),args.batch):
            batch=samples[pos:pos+args.batch]
            prepared=list(workers.map(prepare,batch))
            heat=session.run(['heatmap'],{'input':np.stack([x for x,t in prepared])})[0]
            for row,hm,(_,t) in zip(batch,heat,prepared):
                peaks=candidates(hm,t,row['resolution'])
                gt=np.array(row['ball_centers'][0]);r=row['resolution']
                for peak in peaks:
                    peak['distance_px']=float(np.linalg.norm(gt-[peak['x'],peak['y']]))
                    crop=crop_transform(peak['x'],peak['y'],r['width'],r['height'])
                    peak['roi_contains_gt']=bool(crop['left']<=gt[0]<crop['left']+crop['side'] and crop['top']<=gt[1]<crop['top']+crop['side'])
                records.append(dict(video_id=row['video_id'],frame_id=row['frame_id'],timestamp_sec=row['timestamp_sec'],
                                    resolution=r,gt_center=gt.tolist(),candidates=peaks))
            if len(records)%160==0 or len(records)==len(samples):print(f'candidate diagnostic {len(records)}/{len(samples)}',flush=True)
    summary=dict(measured=now(),dataset_id=dm['dataset_id'],checkpoint_dataset_id=export['dataset_id'],onnx_sha256=sha(model),
                 backend='ONNX Runtime CPU FP32',model_input=[w,h],val_videos=len(jsonl(args.dataset/'splits/val.jsonl')),
                 labelled_val_videos=len(grouped),sample_frames=len(records),
                 total_positive_val_frames=sum(map(len,grouped.values())),sampling='evenly spaced positive frames within every labelled val video; videos with only unknown labels excluded',
                 per_video_cap=args.per_video,candidate_method='3x3 local maxima, descending heat score, 6 heatmap-pixel NMS radius, 7x7 peak centroid',
                 roi='same default Local ROI: 256 original px at 1080p, scaled by source height',
                 elapsed_sec=time.perf_counter()-start,**summarize(records),
                 interpretation='GT used only for oracle candidate coverage and ROI inclusion; these are upper bounds, not tracking accuracy or F1; no model updates')
    stamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S')
    dest=ROOT/'output/diagnostics'/f'{stamp}_global_candidates';dest.mkdir(parents=True)
    save(dest/'summary.json',summary);save_jsonl(dest/'samples.jsonl',records)
    per_video={vid:summarize([r for r in records if r['video_id']==vid]) for vid in grouped}
    save(dest/'per_video.json',per_video)
    lines=['# Global 多候选诊断','',f"样本：{len(records)} 帧，覆盖 {len(grouped)} 个验证视频；CPU FP32，不训练。",
           '', 'GT 仅计算候选覆盖率上限；不代表实际轨迹关联精度。','', '| 阈值 | K | 10px覆盖率 | 50px覆盖率 | ROI覆盖率 |','|---:|---:|---:|---:|---:|']
    for threshold,table in summary['threshold_tables'].items():
        for r in table:lines.append(f"| {threshold} | {r['k']} | {r['coverage_10px']:.2%} | {r['coverage_50px']:.2%} | {r['roi_coverage']:.2%} |")
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True);print('report',dest,flush=True)


if __name__=='__main__':main()
