"""Create input-size-independent v2 labels using existing, immutable original PNGs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

from ttnet.common import DATA,ROOT,identity,jsonl,now,save,save_jsonl,sha
from .prepare import POLICY,SCHEMA

INPUT_FIELDS={"global_resolution","global_transform","global_input_path"}


def migrate(source,dest):
    sm=json.loads((source/'metadata/dataset.json').read_text())
    if sm['status']!='completed':raise ValueError('source dataset incomplete')
    entries=sm['source_videos'];policy_hash=identity(POLICY)
    dataset_id=identity(dict(policy_hash=policy_hash,videos=entries))
    manifest=dest/'metadata/dataset.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['dataset_id']!=dataset_id:raise ValueError('destination identity differs')
        if old['status']=='completed':return old
    dest.mkdir(parents=True,exist_ok=True)
    shared=dest/'frames'
    if shared.is_symlink():
        if shared.resolve()!=(source/'frames').resolve():raise ValueError('wrong shared frame directory')
    elif shared.exists():raise ValueError('destination frames already exists')
    else:shared.symlink_to(os.path.relpath(source/'frames',dest),target_is_directory=True)
    save(manifest,dict(schema=SCHEMA,dataset_id=dataset_id,status='preparing',started=now(),policy_hash=policy_hash))
    save(dest/'metadata/processing_config.json',POLICY)
    for s in ['train','val']:
        save_jsonl(dest/'splits'/f'{s}.jsonl',jsonl(source/'splits'/f'{s}.jsonl'))
    by_split={'train':[],'val':[]};metas=[]
    for entry in entries:
        key=entry['video_id'];path=source/'labels'/f'{key}.jsonl'
        vm=json.loads((source/'metadata/videos'/f'{key}.json').read_text())
        if sha(path)!=vm['labels_sha256']:raise ValueError(f'source labels changed: {key}')
        rows=[{k:v for k,v in r.items() if k not in INPUT_FIELDS} for r in jsonl(path)]
        save_jsonl(dest/'labels'/f'{key}.jsonl',rows)
        windows=source/'windows'/f'{key}.jsonl'
        if sha(windows)!=vm['windows_sha256']:raise ValueError('source windows changed')
        (dest/'windows').mkdir(exist_ok=True)
        shutil.copyfile(windows,dest/'windows'/windows.name)
        vm.pop('resize_version',None)
        vm.update(policy_hash=policy_hash,png_bytes=sum((dest/r['frame_path']).stat().st_size for r in rows),
                  labels_sha256=sha(dest/'labels'/f'{key}.jsonl'),image_encoder_version=sm.get('resize_version'),
                  shared_original_frames=True)
        save(dest/'metadata/videos'/f'{key}.json',vm)
        by_split[entry['split']].extend(rows);metas.append(vm)
    for s,rows in by_split.items():
        save_jsonl(dest/'labels'/f'{s}.jsonl',sorted(rows,key=lambda r:(r['video_id'],r['frame_id'])))
    report=dict(schema=SCHEMA,dataset_id=dataset_id,policy_hash=policy_hash,status='completed',finished=now(),
                videos=len(metas),train_videos=sm['train_videos'],val_videos=sm['val_videos'],frames=sm['frames'],
                label_counts=sm['label_counts'],png_bytes=sum(m['png_bytes'] for m in metas),
                split_hashes={s:sha(dest/'splits'/f'{s}.jsonl') for s in ['train','val']},source_videos=entries,
                source_audit_sha256=sm.get('source_audit_sha256') or json.loads((ROOT/'src/configs/frozen_split.json').read_text())['source_audit_sha256'],
                decode_version=sm['decode_version'],
                image_encoder_version=sm.get('resize_version'),
                migration=dict(source_dataset_id=sm['dataset_id'],source_dataset=os.path.relpath(source.resolve(),ROOT),
                               frame_storage='relative symbolic link to existing original PNG directory; no image copies',
                               source_metadata_sha256=sha(source/'metadata/dataset.json')))
    save(manifest,report)
    save(dest/'metadata/code_provenance.json',dict(schema=SCHEMA,files={f:sha(ROOT/"src"/f) for f in ['data_tools/prepare.py','data_tools/geometry.py','data_tools/migrate_raw.py','common.py']}))
    print(json.dumps({k:v for k,v in report.items() if k!='source_videos'}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'data/datasets/ball_v1');p.add_argument('--dataset',type=Path,default=DATA)
    a=p.parse_args();migrate(a.source,a.dataset)
