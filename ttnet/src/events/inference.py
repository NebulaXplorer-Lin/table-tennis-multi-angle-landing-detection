"""Offline full-video inference. No event GT is required for decoding or prediction."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import time

import av
import numpy as np
import torch

from ttnet.events.data import assemble_window
from ttnet.events.prepare import nearest_index


class StreamingCorpus:
    """Only the raw triple is retained; per-frame metadata is kept for event windows."""
    def __init__(self, vid, meta):
        self.videos={vid:dict(meta=meta,frames={})}
        self.images={}

    def image(self, vid, fid):return self.images[fid]

    def window_ids(self, vid, center):
        n=self.videos[vid]['meta']['n_frames']
        return [i if 0<=i<n else None for i in range(center-6,center+7)]


def source_mapping(n_source, rate, target_fps):
    rate=Fraction(rate)
    # Preserve native frame IDs for nominal 25fps clips with rounded/container
    # average-rate discrepancies; the per-frame PTS check still applies below.
    if abs(float(rate)-target_fps)/target_fps<=.001:
        return n_source,{i:i for i in range(n_source)}
    n_target=int((n_source-1)*target_fps/rate)+1
    mapping={nearest_index(Fraction(i)*rate/target_fps):i for i in range(n_target)}
    if len(mapping)!=n_target or max(mapping)>=n_source:
        raise ValueError('requested sampling would duplicate frames or exceed source')
    return n_target,mapping


def predict_windows(system,corpus,vid,features,records,centers=None,batch=64):
    centers=sorted(features) if centers is None else sorted(set(centers))
    output=[]
    if not hasattr(corpus,'_prediction_windows'):corpus._prediction_windows={}
    signature=tuple((k,system.config.get(k)) for k in ('context_head','scene_mode','event_window_mode'))
    entry=corpus._prediction_windows.get(vid)
    if entry is None or entry[0] is not features or entry[1] is not records or entry[2]!=signature:
        entry=(features,records,signature,{})
        corpus._prediction_windows[vid]=entry
    prepared=entry[3]
    with torch.inference_mode(),torch.autocast(device_type=system.device.type,enabled=system.config['amp'] and system.device.type=='cuda'):
        for start in range(0,len(centers),batch):
            chosen=centers[start:start+batch]
            for c in chosen:
                if c not in prepared:prepared[c]=assemble_window(corpus,vid,c,features,records,torch.device('cpu'),system.config)
            inputs=[prepared[c] for c in chosen]
            tensors=[torch.cat([item[k] for item in inputs]).to(system.device) for k in range(3)]
            probabilities=system.head(*tensors).float().sigmoid().cpu().numpy()
            for fid,p in zip(chosen,probabilities):
                output.append(dict(frame_id=fid,timestamp_sec=corpus.videos[vid]['frames'][fid]['timestamp_sec'],
                                   source_frame_id=corpus.videos[vid]['frames'][fid].get('source_frame_id'),
                                   probabilities=p.tolist(),prediction=records[fid]['prediction']))
    return output


def predict_video(system,source,vid,target_fps=25,progress=True):
    if system.device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    system.reset_tracking(vid)
    started=time.perf_counter();features={};records={}
    with av.open(str(source)) as container:
        stream=container.streams.video[0];rate=Fraction(stream.average_rate)
        n_source=stream.frames
        if not n_source:raise ValueError('source frame count is required')
        native=target_fps is None
        n_target,target_sources=(n_source,{i:i for i in range(n_source)}) if native else source_mapping(n_source,rate,target_fps)
        effective_fps=float(rate) if native else target_fps
        meta=dict(n_frames=n_target,width=stream.width,height=stream.height,fps=effective_fps)
        corpus=StreamingCorpus(vid,meta);frames=corpus.videos[vid]['frames']
        first_pts=None;seen=0;stream.thread_type='AUTO'
        with torch.inference_mode(),torch.autocast(device_type=system.device.type,enabled=system.config['amp'] and system.device.type=='cuda'):
            for source_id,frame in enumerate(container.decode(stream)):
                if frame.pts is None:raise ValueError('missing PTS')
                ts=float(frame.pts*frame.time_base)
                if first_pts is None:first_pts=ts
                if source_id not in target_sources:continue
                fid=target_sources[source_id];seen+=1
                if not native and abs(ts-first_pts-fid/target_fps)>.51/float(rate):raise ValueError('non-CFR sampling mismatch')
                if frames and ts-first_pts<=frames[max(frames)]['timestamp_sec']:raise ValueError('non-monotonic video PTS')
                corpus.images[fid]=frame.to_ndarray(format='rgb24')
                frames[fid]=dict(timestamp_sec=ts-first_pts,source_frame_id=source_id,label_status='unknown',xy=None)
                if fid>=1:
                    center=fid-1
                    d,r,_=system.forward_frames(corpus,vid,[center])
                    features[center]=d[center].float().cpu();records.update(r)
                    corpus.images.pop(center-1,None)
                if progress and fid%1000==0:print(f'full video {vid} {fid}/{n_target}',flush=True)
            if seen!=n_target:raise ValueError(f'incomplete video decode {seen}/{n_target}')
            d,r,_=system.forward_frames(corpus,vid,[n_target-1])
            features[n_target-1]=d[n_target-1].float().cpu();records.update(r)
        corpus.images.clear()
    rows=predict_windows(system,corpus,vid,features,records)
    # Include JSON serialization, exclude actual disk I/O and model initialization.
    import json
    json.dumps(rows,allow_nan=False)
    if system.device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    from ttnet.events.data import event_window_ids
    waits=[]
    for i in range(n_target):
        ids=[j for j in event_window_ids(corpus,vid,i,system.config) if j is not None]
        last=min(n_target-1,max(ids)+1)
        waits.append(frames[last]['timestamp_sec']-frames[i]['timestamp_sec'])
    performance=dict(frames=len(rows),elapsed_sec=elapsed,e2e_fps=len(rows)/elapsed,
                     peak_cuda_bytes=torch.cuda.max_memory_allocated() if system.device.type=='cuda' else None,
                     input_fps=float(rate),output_fps=effective_fps,native_frame_ids=native,source=str(Path(source)),
                     hardware=str(system.device),backend='PyTorch',precision='AMP' if system.config['amp'] and system.device.type=='cuda' else 'FP32',
                     tracker_future_frames=0,event_future_frames=6,localizer_future_frames=1,
                     nominal_minimum_future_wait_sec=(.24+1/effective_fps if system.config.get('event_window_mode')=='seconds' else 7/effective_fps),
                     event_window_mode=system.config.get('event_window_mode','frames'),
                     actual_future_wait_sec=dict(min=min(waits),max=max(waits)) if waits else None,execution='offline_full_video',
                     protocol='CPU PyAV decode + sampling + preprocessing + both locators + head + JSON serialization; excludes disk writes/init')
    return rows,records,performance
