"""Run-time resized-frame cache, independent of the immutable source dataset."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import cv2
import math
from collections import OrderedDict

from ttnet.common import ROOT,identity,jsonl,save,sha
from ttnet.data_tools.geometry import letterbox,crop_image,crop_transform
from ttnet.temporal import sample_window,temporal_mode
from .scales import input_size


def cache_video(args):
    dataset,cache,vid,rows,sizes=args
    cv2.setNumThreads(1)
    dataset,cache=Path(dataset),Path(cache)
    marker=cache/'complete'/f'{vid}.json'
    if marker.exists():
        if all((cache/f'{w}x{h}'/vid/f"{r['frame_id']:06d}.png").is_file() for w,h in sizes for r in rows):
            return vid,True
    for r in rows:
        bgr=cv2.imread(str(dataset/r['frame_path']))
        if bgr is None:raise RuntimeError('unreadable original frame')
        for w,h in sizes:
            image,_=letterbox(bgr,w,h)
            path=cache/f'{w}x{h}'/vid/f"{r['frame_id']:06d}.png";path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_name(path.stem+'.tmp.png')
            if not cv2.imwrite(str(tmp),image,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise RuntimeError('cache write failed')
            tmp.replace(path)
    save(marker,dict(video_id=vid,frames=len(rows),sizes=sizes))
    return vid,False


def ensure_global_cache(dataset,sizes,workers=4):
    sizes=sorted(set(tuple(s) for s in sizes))
    dm=json.loads((dataset/'metadata/dataset.json').read_text())
    key=identity(dict(dataset_id=dm['dataset_id'],sizes=sizes,geometry_sha256=sha(ROOT/'src/data_tools/geometry.py'),
                      policy='letterbox uint8 PNG; augmentation and normalization applied at read time'))
    root=ROOT/'data/cache/global_frames'/key[:20]
    groups={}
    for split in ['train','val']:
        for r in jsonl(dataset/'labels'/f'{split}.jsonl'):groups.setdefault(r['video_id'],[]).append(r)
    print(f'Preparing run-time resize cache: {len(groups)} videos, sizes={sizes}',flush=True)
    done=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(cache_video,(str(dataset),str(root),vid,rows,sizes)) for vid,rows in groups.items()]
        for f in as_completed(futures):
            vid,reused=f.result();done+=1
            if done%25==0 or done==len(groups):print(f'input cache {done}/{len(groups)}; {vid}',flush=True)
    save(root/'identity.json',dict(dataset_id=dm['dataset_id'],sizes=sizes,cache_key=key,complete=True))
    return root


def gt_carrier_transform(row,side_at_1080=256):
    x,y=row['ball_centers'][0];r=row['resolution']
    native=max(2,round(side_at_1080*r['height']/1080));side=math.ceil(native*1.5)
    return dict(left=round(x-side/2),top=round(y-side/2),side=side,output_size=side,scale=1.,
                original_width=r['width'],original_height=r['height'])


def local_cache_video(args):
    dataset,root,vid,rows,predictions,config=args
    cv2.setNumThreads(1);dataset,root=Path(dataset),Path(root)
    marker=root/'complete'/f'{vid}.json'
    if marker.exists():return vid,True
    mode=temporal_mode(config);times=[r['timestamp_sec'] for r in rows]
    width,height=input_size(config,'local');span=config.get('local_crop_side_1080',256)
    frames=OrderedDict()
    def raw(fid):
        if fid not in frames:
            image=cv2.imread(str(dataset/rows[fid]['frame_path']))
            if image is None:raise RuntimeError('unreadable source image')
            frames[fid]=image
            while len(frames)>12:frames.popitem(last=False)
        return frames[fid]
    gt_count=pred_count=0
    for row,pred in zip(rows,predictions):
        center=row['frame_id'];index=sample_window(times,center,mode,config.get('window_half_span_sec',1/30))
        transforms=[]
        if row['label_status']=='positive':transforms.append(('gt_context',gt_carrier_transform(row,span)))
        if pred and pred['valid']:
            r=row['resolution'];transforms.append(('predicted',crop_transform(pred['x'],pred['y'],r['width'],r['height'],width,span)))
        for fid in index['frame_ids']:
            if fid is None:continue
            image=raw(fid)
            for kind,t in transforms:
                out=crop_image(image,t)
                path=root/kind/vid/f'{center:06d}_{fid:06d}.png';path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_name(path.stem+'.tmp.png')
                if not cv2.imwrite(str(tmp),out,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise RuntimeError('crop cache write failed')
                tmp.replace(path)
                if kind=='gt_context':gt_count+=1
                else:pred_count+=1
    save(marker,dict(video_id=vid,frames=len(rows),gt_context_images=gt_count,predicted_images=pred_count))
    return vid,False


def ensure_local_cache(dataset,prediction_maps,global_sha256,config):
    dm=json.loads((dataset/'metadata/dataset.json').read_text())
    policy=dict(dataset_id=dm['dataset_id'],global_sha256=global_sha256,temporal_mode=temporal_mode(config),
                local_input=list(input_size(config,'local')),side_at_1080=config.get('local_crop_side_1080',256),
                half_span=config.get('window_half_span_sec',1/30),gt_jitter_limit=.2,carrier_factor=1.5,
                source_sha256=sha(Path(__file__)),geometry_sha256=sha(ROOT/'src/data_tools/geometry.py'))
    root=ROOT/'data/cache/local_crops'/identity(policy)[:20]
    groups={};all_predictions={}
    for split in ['train','val']:
        for r in jsonl(dataset/'labels'/f'{split}.jsonl'):groups.setdefault(r['video_id'],[]).append(r)
        all_predictions.update(prediction_maps[split])
    print('Preparing exact Local crop cache (GT jitter carrier + frozen predicted crops)',flush=True)
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        futures=[pool.submit(local_cache_video,(str(dataset),str(root),vid,rows,
                                                [all_predictions.get((vid,r['frame_id'])) for r in rows],config)) for vid,rows in groups.items()]
        for done,f in enumerate(as_completed(futures),1):
            f.result()
            if done%25==0 or done==len(groups):print(f'Local crop cache {done}/{len(groups)}',flush=True)
    save(root/'identity.json',dict(**policy,complete=True))
    return root
