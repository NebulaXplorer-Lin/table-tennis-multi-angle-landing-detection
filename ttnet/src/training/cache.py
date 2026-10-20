"""Rebuildable NPY input stores; source pixels remain immutable."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import math
from pathlib import Path

import cv2

from ttnet.common import CACHE,ROOT,identity,jsonl,save,sha
from ttnet.data_tools.frame_store import FrameStore,pack_video
from ttnet.data_tools.geometry import letterbox,crop_image,crop_transform
from ttnet.temporal import sample_window,temporal_mode
from .scales import input_size


def _publish(root,key,dataset_id,videos):
    root=Path(root)
    save(root/'manifest.json',dict(schema=1,status='completed',storage_id=key,dataset_id=dataset_id,
                                   videos=sorted(videos),format='npy_rgb_u8_v1'))
    save(root/'frame_store.json',dict(schema=1,path='.',storage_id=key))


def cache_video(args):
    dataset,cache,vid,rows,sizes,dataset_id=args
    cv2.setNumThreads(1)
    store=FrameStore(dataset)
    for w,h in sizes:
        def reader(row):return letterbox(store.row_rgb(row),w,h)[0]
        pack_video(Path(cache)/f'{w}x{h}',vid,rows,reader,
                   source_identity=dict(dataset_id=dataset_id,size=[w,h]))
    return vid


def ensure_global_cache(dataset,sizes,workers=4):
    dataset=Path(dataset)
    sizes=sorted(set(tuple(s) for s in sizes))
    dm=json.loads((dataset/'metadata/dataset.json').read_text())
    policy=dict(dataset_id=dm['dataset_id'],sizes=sizes,geometry_sha256=sha(ROOT/'src/data_tools/geometry.py'),
                policy='npy_rgb_u8_v1; letterbox before augmentation and normalization')
    backend=FrameStore(dataset).backend_identity()
    if backend:policy['pixel_backend']=backend
    key=identity(policy)
    root=CACHE/'inputs/global'/key[:20]
    groups={}
    for split in ('train','val'):
        for row in jsonl(dataset/'labels'/f'{split}.jsonl'):groups.setdefault(row['video_id'],[]).append(row)
    print(f'Preparing NPY input cache: {len(groups)} videos, sizes={sizes}',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(cache_video,(str(dataset),str(root),vid,rows,sizes,dm['dataset_id'])) for vid,rows in groups.items()]
        for done,future in enumerate(as_completed(futures),1):
            future.result()
            if done%25==0 or done==len(groups):print(f'input cache {done}/{len(groups)}',flush=True)
    for w,h in sizes:_publish(root/f'{w}x{h}',identity(dict(policy,size=[w,h])),dm['dataset_id'],groups)
    save(root/'identity.json',dict(**policy,cache_key=key,complete=True))
    return root


def gt_carrier_transform(row,side_at_1080=256,jitter_fraction=.2):
    x,y=row['ball_centers'][0];r=row['resolution']
    native=max(2,round(side_at_1080*r['height']/1080));side=math.ceil(native*max(1.5,1+2*jitter_fraction+.1))
    return dict(left=round(x-side/2),top=round(y-side/2),side=side,output_size=side,scale=1.,
                original_width=r['width'],original_height=r['height'])


def local_cache_video(args):
    dataset,root,vid,rows,predictions,config,policy=args
    cv2.setNumThreads(1)
    root=Path(root);store=FrameStore(dataset)
    times=[r['timestamp_sec'] for r in rows]
    width,_=input_size(config,'local');span=config.get('local_crop_side_1080',256)
    grouped={'gt_context':[],'predicted':[]};pairs={'gt_context':{},'predicted':{}}
    for row,pred in zip(rows,predictions):
        transforms=[]
        if row['label_status']=='positive':transforms.append(('gt_context',gt_carrier_transform(row,span,config.get('augmentation',{}).get('gt_crop_jitter_fraction',.2))))
        if pred and pred['valid']:
            r=row['resolution'];transforms.append(('predicted',crop_transform(pred['x'],pred['y'],r['width'],r['height'],width,span)))
        window=sample_window(times,row['frame_id'],temporal_mode(config),config.get('window_half_span_sec',1/30))
        for fid in window['frame_ids']:
            if fid is None:continue
            for kind,transform in transforms:
                pair=f"{row['frame_id']}:{fid}"
                if pair in pairs[kind]:continue
                ordinal=len(grouped[kind]);pairs[kind][pair]=ordinal
                grouped[kind].append(dict(frame_id=ordinal,source_frame_id=fid,
                                           timestamp_sec=times[fid],row=rows[fid],transform=transform))
    for kind,items in grouped.items():
        if items:
            def reader(item):return crop_image(store.row_rgb(item['row']),item['transform'])
            pack_video(root/kind,vid,items,reader,source_identity=dict(policy,kind=kind))
        save(root/kind/vid/'pairs.json',pairs[kind])
    return vid


def ensure_local_cache(dataset,prediction_maps,global_sha256,config):
    dataset=Path(dataset);dm=json.loads((dataset/'metadata/dataset.json').read_text())
    policy=dict(dataset_id=dm['dataset_id'],global_sha256=global_sha256,temporal_mode=temporal_mode(config),
                local_input=list(input_size(config,'local')),side_at_1080=config.get('local_crop_side_1080',256),
                half_span=config.get('window_half_span_sec',1/30),gt_jitter_limit=config.get('augmentation',{}).get('gt_crop_jitter_fraction',.2),
                source_sha256=sha(Path(__file__)),geometry_sha256=sha(ROOT/'src/data_tools/geometry.py'),
                prediction_sha256=identity([{str(k):v for k,v in sorted(mapping.items())} for mapping in prediction_maps.values()]))
    backend=FrameStore(dataset).backend_identity()
    if backend:policy['pixel_backend']=backend
    root=CACHE/'inputs/local'/identity(policy)[:20]
    groups={};all_predictions={}
    for split in ('train','val'):
        for row in jsonl(dataset/'labels'/f'{split}.jsonl'):groups.setdefault(row['video_id'],[]).append(row)
        all_predictions.update(prediction_maps[split])
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        futures=[pool.submit(local_cache_video,(str(dataset),str(root),vid,rows,
                    [all_predictions.get((vid,r['frame_id'])) for r in rows],config,policy)) for vid,rows in groups.items()]
        for done,future in enumerate(as_completed(futures),1):
            future.result()
            if done%25==0 or done==len(groups):print(f'Local cache {done}/{len(groups)}',flush=True)
    for kind in ('gt_context','predicted'):
        videos=[v for v in groups if (root/kind/v/'manifest.json').is_file()]
        _publish(root/kind,identity(dict(policy,kind=kind)),dm['dataset_id'],videos)
    save(root/'identity.json',dict(**policy,complete=True))
    return root
