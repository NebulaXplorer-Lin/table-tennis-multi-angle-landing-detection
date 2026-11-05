"""Whole-source native OpenTT inference, GT-free selection and sparse scoring."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import av
import cv2
import numpy as np
import torch

from ttnet.common import PROJECT, jsonl, save, save_jsonl, sha
from ttnet.data_tools.prepare_mixed_ball import ball_label
from ttnet.data_tools.video_store import frame_rgb
from ttnet.events.engine import EventSystem
from ttnet.events.inference import StreamingCorpus
from ttnet.evaluation.candidate_audit import classify_error,in_roi,summarize
from ttnet.evaluation.evaluate import overlay
from ttnet.evaluation.trajectory_comparison import measure


def system_config(global_checkpoint,local_checkpoint,tracker='independent',checkpoint_thresholds=False):
    g=torch.load(global_checkpoint,map_location='cpu',weights_only=False)
    l=torch.load(local_checkpoint,map_location='cpu',weights_only=False)
    cfg=dict(g['config'],global_checkpoint=str(global_checkpoint),local_checkpoint=str(local_checkpoint),
             event_head_type='tcn',tracker_type=tracker,localization_threshold=.7,
             candidate_diagnostics=True,mode='frozen',amp=True)
    if checkpoint_thresholds:
        cfg.update(global_threshold=g['threshold'],local_threshold=l['threshold'])
    else:
        cfg.pop('global_threshold',None);cfg.pop('local_threshold',None)
    return cfg


def compact_prediction(p):
    return {k:p[k] for k in ('x','y','confidence','valid','source') if k in p} if p else None


def evaluate_native(dataset,global_checkpoint,local_checkpoint,output,part='val',tracker='independent',checkpoint_thresholds=False):
    dataset,output=Path(dataset),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    cfg=system_config(global_checkpoint,local_checkpoint,tracker,checkpoint_thresholds)
    system=EventSystem(cfg,torch.device('cuda'));system.configure('frozen',False)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    splits=[e for e in jsonl(dataset/f'splits/{part}.jsonl') if e['source_domain']=='opentt']
    per_video={};all_rows=[];all_records=[];candidate_rows=[];errors=[]
    full_frames=0;elapsed=0.;inference_seconds=0.
    for entry in splits:
        vid=entry['video_id'];meta=json.loads((dataset/f'metadata/videos/{vid}.json').read_text())
        if meta.get('source_hashes'):
            if sha(PROJECT/meta['video'])!=meta['source_hashes']['video_sha256'] or sha(PROJECT/meta['annotation'])!=meta['source_hashes']['annotation_sha256']:
                raise ValueError(f'native source identity changed: {vid}')
        raw=json.loads((PROJECT/meta['annotation']).read_text())
        times=np.load(dataset/f'native_store/{vid}/all_timestamps.npy',mmap_mode='r')
        corpus=StreamingCorpus(vid,dict(n_frames=meta['frames'],fps=meta['fps'],width=meta['width'],height=meta['height']))
        frames=corpus.videos[vid]['frames'];system.reset_tracking(vid)
        rows=[];records=[];start=time.perf_counter();seen=0
        def process(fid,stream):
            nonlocal inference_seconds
            before=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda'):
                _,result,_=system.forward_frames(corpus,vid,[fid])
            rec=result[fid]
            inference_seconds+=time.perf_counter()-before
            status,centers=ball_label(raw.get(str(fid)),meta['width'],meta['height'])
            stream.write(json.dumps(dict(video_id=vid,frame_id=fid,timestamp_sec=float(times[fid]),
                            label_status=status,prediction=compact_prediction(rec.get('prediction'))))+'\n')
            if status not in ('positive','negative'):return
            row=dict(video_id=vid,frame_id=fid,timestamp_sec=float(times[fid]),fps=meta['fps'],nominal_fps=meta['fps'],
                     resolution=dict(width=meta['width'],height=meta['height']),label_status=status,ball_centers=centers,ball_boxes=[])
            rec=dict(rec,video_id=vid,prediction=compact_prediction(rec.get('prediction')))
            rows.append(row);records.append(rec)
            if status=='positive':
                gt=centers[0];peaks=[]
                from ttnet.data_tools.geometry import crop_transform
                for c in rec.get('global_candidates',[]):
                    crop=crop_transform(c['x'],c['y'],meta['width'],meta['height'],256,256)
                    peaks.append(dict(distance_px=float(np.hypot(c['x']-gt[0],c['y']-gt[1])),
                                      confidence=c['confidence'],roi_contains_gt=in_roi(crop,gt)))
                candidate_rows.append(dict(video_id=vid,frame_id=fid,candidates=peaks))
            kinds=classify_error(row,rec,10)
            if kinds:
                image,kind,error=overlay(cv2.cvtColor(corpus.image(vid,fid),cv2.COLOR_RGB2BGR),row,rec['prediction'])
                path=output/'errors'/vid/f'{fid:06d}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,85]):raise OSError(path)
                errors.append(dict(video_id=vid,frame_id=fid,kinds=kinds,error_px=error,visualization=str(path.relative_to(output))))
        with (output/f'{vid}_predictions.jsonl').open('w') as stream,av.open(str(PROJECT/meta['video'])) as container:
            source=container.streams.video[0];source.codec_context.thread_count=2
            first=None
            for fid,frame in enumerate(container.decode(source)):
                if frame.pts is None:raise ValueError('missing native PTS')
                ts=float(frame.pts*frame.time_base);first=ts if first is None else first
                if abs(ts-first-times[fid])>1e-7:raise ValueError('native PTS identity changed')
                frames[fid]=dict(timestamp_sec=ts-first,label_status='unknown',xy=None)
                corpus.images[fid]=frame_rgb(frame,'lossless_bgr_to_rgb');seen+=1
                if fid:
                    process(fid-1,stream)
                    for old in list(corpus.images):
                        if old<fid-1:corpus.images.pop(old);frames.pop(old,None)
                if fid%10000==0:print(f'native {tracker} {vid}: {fid}/{meta["frames"]}',flush=True)
            if seen!=meta['frames']:raise ValueError('incomplete native source inference')
            process(seen-1,stream)
        seconds=time.perf_counter()-start;elapsed+=seconds;full_frames+=seen
        metric,jumps=measure(rows,records,{})
        per_video[vid]=dict(metric,frames=seen,scored_frames=len(rows),elapsed_sec=seconds)
        all_rows.extend(rows);all_records.extend(records)
        save(output/'progress.json',dict(completed_videos=list(per_video),full_frames=full_frames))
    totals,_=measure(all_rows,all_records,{})
    result=dict(status='evaluated',part=part,tracker=tracker,training_dataset_id=system.config.get('locator_training_dataset_id'),
                global_sha256=sha(global_checkpoint),local_sha256=sha(local_checkpoint),
                thresholds=dict(global_=cfg.get('global_threshold',.7),local=cfg.get('local_threshold',.7)),
                metric=totals,candidate=summarize(candidate_rows),per_video=per_video,
                full_frames=full_frames,scored_frames=len(all_rows),error_frames=len(errors),
                elapsed_sec=elapsed,inference_calls_sec=inference_seconds,
                timing_scope='whole source decode, GPU inference, sparse scoring and error JPEGs; not an inference-only FPS benchmark')
    save(output/'metrics.json',result);save(output/'errors.json',errors)
    (output/'index.html').write_text('<meta charset="utf-8"><h1>Native OpenTT ball errors: green GT / red prediction</h1>'+''.join(
        f'<figure><img loading="lazy" width="480" src="{e["visualization"]}"><figcaption>{e["video_id"]}:{e["frame_id"]} {e["kinds"]}</figcaption></figure>' for e in errors))
    return result
