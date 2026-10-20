from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ttnet.common import identity, jsonl
from ttnet.data_tools.frame_store import FrameStore


class EventCorpus:
    def __init__(self, root):
        self.context_stores = {}
        self.root = Path(root)
        self.split = json.loads((self.root/'split.json').read_text())
        self.metadata = json.loads((self.root/'dataset.json').read_text())
        self.videos = {}
        self.samples = {}
        for part in ('train','val','test'):
            self.samples[part] = []
            for vid in self.split[part]:
                path = self.root/vid
                if not (path/'metadata.json').exists():
                    continue
                meta = json.loads((path/'metadata.json').read_text())
                self.videos[vid] = dict(meta=meta, frames={r['frame_id']:r for r in jsonl(path/'frames.jsonl')},
                                        events=jsonl(path/'events.jsonl'))
                self.samples[part].extend(dict(s, video_id=vid) for s in jsonl(path/'samples.jsonl'))
        self.identity = self.metadata['dataset_id']
        self.frame_store = FrameStore(self.root)
        if self.frame_store.is_video:
            for vid,video in self.videos.items():
                video['meta']['source_original']=video['meta']['source']
                video['meta']['source']=str(self.frame_store.source_path(vid))

    @lru_cache(maxsize=32)
    def image(self, vid, fid):
        if vid in self.context_stores:
            return self.context_stores[vid].get_rgb(vid,fid)
        row = self.videos[vid]['frames'][fid]
        return self.frame_store.get_rgb(vid,fid,path=str(Path(vid)/row['path']))

    def window_ids(self, vid, center):
        video = self.videos[vid]
        ids = [i if 0<=i<video['meta']['n_frames'] else None for i in range(center-6,center+7)]
        if any(i is not None and i not in video['frames'] for i in ids):
            raise ValueError(f"window crosses uncached interval: {vid}:{center}")
        return ids


def balanced_samples(samples, seed, epoch):
    """Balance anchors, then choose one of its soft-label shifts. Paired across modes."""
    rng = np.random.default_rng(np.random.SeedSequence([seed,epoch]))
    groups = {}
    for sample in samples:
        groups.setdefault(sample['group'],{}).setdefault((sample['video_id'],sample['anchor']),[]).append(sample)
    if set(groups) != {'bounce','net','empty_event'}:
        raise ValueError('all three groups are required')
    count = max(len(g) for g in groups.values())
    result=[]
    for group in ('bounce','net','empty_event'):
        anchors=list(groups[group].values())
        indexes=rng.permutation(len(anchors)).tolist()
        indexes.extend(rng.integers(len(anchors),size=count-len(indexes)).tolist())
        result.extend(anchors[i][int(rng.integers(len(anchors[i])))] for i in indexes)
    return [result[i] for i in rng.permutation(len(result))]


def window_signature(config):
    return tuple((k,config.get(k)) for k in ('context_head','scene_mode','event_window_mode',
                                            'table_smoothing','table_smoothing_tau_sec'))


def assemble_window(corpus, vid, center, descriptors, records, device, config=None):
    config=config or {}
    ids=event_window_ids(corpus,vid,center,config)
    features=[]; motion=[]; valid=[]
    frames=corpus.videos[vid]['frames'];meta=corpus.videos[vid]['meta']
    from ttnet.events.refinement import smooth_window_tables
    tables=smooth_window_tables(ids,frames,records,config)
    zero=next(iter(descriptors.values())).new_zeros(256)
    for fid in ids:
        frame_valid=fid is not None and frames[fid].get('frame_valid',True)
        valid.append(frame_valid)
        if not frame_valid:
            features.append(zero);motion.append([0.]*(18 if config.get('context_head') else 9));continue
        features.append(descriptors[fid])
        rec=records[fid]; p=rec['prediction']; pv=bool(p and p['valid'])
        x,y=(p['x']/meta['width'],p['y']/meta['height']) if pv else (0.,0.)
        prev=records.get(fid-1)
        previous=prev['prediction'] if prev else None
        vv=bool(pv and previous and previous['valid'] and fid-1 in frames)
        dt=frames[fid]['timestamp_sec']-frames[fid-1]['timestamp_sec'] if vv else 0.
        vv=bool(vv and dt>0)
        vx=(x-previous['x']/meta['width'])/dt if vv else 0.
        vy=(y-previous['y']/meta['height'])/dt if vv else 0.
        motion.append([x,y,vx,vy,p['confidence'] if pv else 0.,float(pv),float(vv),float(rec['local_valid']),
                       frames[fid]['timestamp_sec']-frames[center]['timestamp_sec']])
        if config.get('context_head'):
            from ttnet.scene.table import geometry
            table=rec.get('table') if tables is None else tables.get(fid)
            motion[-1].extend(geometry(table,p,meta['width'],meta['height'],config.get('scene_mode','none')).tolist())
    return (torch.stack(features).unsqueeze(0),torch.tensor([motion],device=device,dtype=torch.float32),
            torch.tensor([valid],device=device,dtype=torch.bool))


def sample_identity(samples):
    return identity([(s['video_id'],s['anchor'],s['frame_id'],s['target'],s['supervised']) for s in samples])


def event_window_ids(corpus,vid,center,config):
    if config.get('event_window_mode','frames')=='frames':return corpus.window_ids(vid,center)
    frames=corpus.videos[vid]['frames'];ordered=sorted(frames)
    times=np.array([frames[i]['timestamp_sec'] for i in ordered]);tc=frames[center]['timestamp_sec']
    result=[];used=set()
    for offset in np.linspace(-.24,.24,13):
        desired=tc+offset
        if desired<times[0]-1e-8 or desired>times[-1]+1e-8:result.append(None);continue
        j=int(np.argmin(abs(times-desired)));fid=ordered[j]
        # Missing intervals and repeated source frames must not become synthetic observations.
        if abs(times[j]-desired)>.021 or fid in used:result.append(None);continue
        result.append(fid);used.add(fid)
    result[6]=center
    return result
