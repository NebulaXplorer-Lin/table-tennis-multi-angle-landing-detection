from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from ttnet.common import DATA,ROOT,identity,jsonl,save,sha
from .geometry import window_indices
from .frame_store import FrameStore


def audit(dataset,full_images=False):
    store=FrameStore(dataset)
    meta=json.loads((dataset/'metadata/dataset.json').read_text())
    frozen=json.loads((ROOT/'src/configs/frozen_split.json').read_text())
    assert meta['status']=='completed'
    splits={s:jsonl(dataset/'splits'/f'{s}.jsonl') for s in ['train','val']}
    assert len(splits['train'])==114 and len(splits['val'])==40
    assert not {x['video_id'] for x in splits['train']} & {x['video_id'] for x in splits['val']}
    assert sorted(splits['train']+splits['val'],key=lambda x:x['video_id'])==sorted(frozen['videos'],key=lambda x:x['video_id'])
    total=Counter(); fps_groups=Counter();resolutions=Counter()
    for s,entries in splits.items():
        assert sha(dataset/'splits'/f'{s}.jsonl')==meta['split_hashes'][s]
        expected=[]
        for e in entries:
            vm=json.loads((dataset/'metadata/videos'/f"{e['video_id']}.json").read_text())
            rows=jsonl(dataset/'labels'/f"{e['video_id']}.jsonl")
            windows=jsonl(dataset/'windows'/f"{e['video_id']}.jsonl")
            assert vm['identity']==identity(e) and vm['complete']
            assert sha(dataset/'labels'/f"{e['video_id']}.jsonl")==vm['labels_sha256']
            assert sha(dataset/'windows'/f"{e['video_id']}.jsonl")==vm['windows_sha256']
            assert len(rows)==vm['actual_frames']==len(windows)
            times=[r['timestamp_sec'] for r in rows]
            assert not len(times)>1 or np.all(np.diff(times)>0)
            fps_groups[min([25,30,60,120],key=lambda f:abs(f-vm['nominal_fps']))]+=1
            resolutions[f"{vm['resolution']['width']}x{vm['resolution']['height']}"]+=1
            for i,(r,w) in enumerate(zip(rows,windows)):
                assert r['frame_id']==i and r['video_id']==e['video_id'] and r['split']==s
                for field in ['resolution','fps','fps_num','fps_den','nominal_fps']:assert r[field]==vm[field]
                assert np.isclose(r['fps'],r['fps_num']/r['fps_den'])
                assert np.isclose(r['timestamp_sec'],r['pts']*r['time_base_num']/r['time_base_den']-vm['pts_audit']['first_pts_sec'])
                for field,value in window_indices(times,i).items():assert w[field]==value
                if r['label_status']=='unknown':assert not r['ball_boxes']
                if r['label_status']=='positive':
                    assert len(r['ball_centers'])==1
                    c=r['ball_centers'][0]
                    assert 0<=c[0]<r['resolution']['width'] and 0<=c[1]<r['resolution']['height']
                image_fields=[('frame_path',(r['resolution']['width'],r['resolution']['height']))]
                if meta['schema'].startswith('ttnet_ball_v2'):
                    assert not {'global_resolution','global_transform','global_input_path'} & r.keys()
                elif 'global_input_path' in r:
                    image_fields.append(('global_input_path',(512,288)))
                for field,dims in image_fields:
                    if store.root is not None and field=='frame_path':
                        image=store.row_rgb(r)
                        assert image.shape[1::-1]==dims and image.dtype==np.uint8
                        continue
                    p=dataset/r[field];assert p.is_file() and p.stat().st_size>0
                    if full_images:
                        image=cv2.imread(str(p));assert image is not None and image.shape[1::-1]==dims
                total[r['label_status']]+=1
            expected.extend(rows)
            if len(expected)%1000<len(rows):
                print(f"{s} PNG audit: {e['video_id']}, {len(expected)} frames verified",flush=True)
        combined=jsonl(dataset/'labels'/f'{s}.jsonl')
        assert [(r['video_id'],r['frame_id']) for r in combined]==[(r['video_id'],r['frame_id']) for r in sorted(expected,key=lambda x:(x['video_id'],x['frame_id']))]
        print(f'audited {s}: {len(entries)} videos, {len(combined)} frames',flush=True)
    assert sum(total.values())==meta['frames']==23690
    report=dict(passed=True,dataset_id=meta['dataset_id'],full_png_decode=full_images,
                frames=sum(total.values()),label_counts=dict(total),fps_groups=dict(fps_groups),resolutions=dict(resolutions))
    save(dataset/'metadata/audit.json',report)
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=DATA);p.add_argument('--full-images',action='store_true')
    a=p.parse_args();audit(a.dataset,a.full_images)
