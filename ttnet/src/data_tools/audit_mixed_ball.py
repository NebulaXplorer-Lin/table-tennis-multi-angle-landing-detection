"""Cross-domain near-duplicate screening, followed by temporal verification."""
from collections import defaultdict
from pathlib import Path
import argparse
import json

import cv2
import numpy as np

from ttnet.common import jsonl, save, save_jsonl, sha
from ttnet.data_tools.frame_store import FrameStore
from ttnet.training.cache import ensure_global_cache


def fingerprint(image):
    gray=cv2.cvtColor(cv2.resize(image,(32,32)),cv2.COLOR_RGB2GRAY).astype(np.float32)
    dct=cv2.dct(gray)[:8,:8].reshape(-1)[1:]
    bits=dct>np.median(dct)
    value=sum(int(b)<<i for i,b in enumerate(bits))
    return value,gray


def audit(dataset,output):
    dataset,output=Path(dataset),Path(output)
    cache=ensure_global_cache(dataset,[(512,288)],2)
    store=FrameStore(cache/'512x288')
    rows={part:jsonl(dataset/f'labels/{part}.jsonl') for part in ('train','val','test')}
    # Test is consulted only to prevent copied training content, never for tuning.
    pp=[r for part in ('train','val') for r in rows[part] if r['source_domain']=='pp' and r['frame_id']%5==0]
    ot=[r for part in ('train','val','test') for r in rows[part] if r['source_domain']=='opentt']
    original_store=FrameStore(dataset)
    buckets=defaultdict(set);indexed=[]
    for r in pp:
        h,g=fingerprint(store.row_rgb(r));index=len(indexed);indexed.append((r,h,g))
        for band in range(8):buckets[(band,(h>>(band*8))&255)].add(index)
    candidates=[]
    for done,r in enumerate(ot):
        h,g=fingerprint((original_store if r['split']=='test' else store).row_rgb(r));possible=set()
        for band in range(8):possible.update(buckets[(band,(h>>(band*8))&255)])
        for i in possible:
            p,other,pg=indexed[i]
            if p['split']==r['split'] or (h^other).bit_count()>3:continue
            # Quantify full-image agreement, then leave candidates reviewable.
            mae=float(np.abs(g-pg).mean())
            if mae<=5:
                candidates.append(dict(pp_video=p['video_id'],pp_frame=p['frame_id'],pp_split=p['split'],
                                       opentt_video=r['video_id'],opentt_frame=r['frame_id'],opentt_split=r['split'],
                                       phash_distance=(h^other).bit_count(),gray_mae=mae))
        if done%5000==0:print(f'overlap screen {done}/{len(ot)}, candidates={len(candidates)}',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    save_jsonl(output/'candidates.jsonl',candidates)
    result=dict(status='review_required' if candidates else 'passed_screening',
                pp_sampled_frames=len(pp),opentt_scored_frames=len(ot),candidates=len(candidates),
                source_manifest_sha256=sha(dataset/'metadata/dataset.json'),
                policy='PP every fifth frame vs native train/val/test ball centers; pHash<=3 and gray32 MAE<=5; exact source hashes checked during preparation',
                limitations='A screen for copied content, not proof of absence of all crops/overlays. Test content is consulted solely for training-leakage screening, not model selection.',
                exclusions=[])
    save(output/'audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.dataset,a.output)),flush=True)


if __name__=='__main__':main()
