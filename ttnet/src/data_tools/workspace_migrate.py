"""Migrate current image datasets to exact-pixel, resumable NPY frame stores."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time

import cv2

from ttnet.common import PROJECT, identity, jsonl, save, sha
from .frame_store import FrameStore, image_rgb, pack_video

DATASETS = {'pp_ball_v2': ('ttnet/data/datasets/ball_v2','ball'),
            'openttgames_event25_v3': ('ttnet/data/datasets/event25_v3','event')}


def source_rows(source, kind):
    grouped = {}
    if kind == 'ball':
        for split in ('train','val'):
            for row in jsonl(source/'labels'/f'{split}.jsonl'):
                grouped.setdefault(row['video_id'],[]).append(row)
    else:
        for path in sorted(source.glob('*/frames.jsonl')):
            grouped[path.parent.name] = [dict(r, video_id=path.parent.name) for r in jsonl(path)]
    return grouped


def migrate_video(args):
    source, dest, kind, video_id, rows, dataset_id, chunk_frames = args
    source, dest = Path(source), Path(dest)
    cv2.setNumThreads(1)
    frame_store=FrameStore(source)
    def read(row):
        if frame_store.root is not None:
            return frame_store.get_rgb(video_id,row['frame_id'])
        path = source/row['frame_path'] if kind == 'ball' else source/video_id/row['path']
        return image_rgb(path)
    started = time.perf_counter()
    result = pack_video(dest, video_id, rows, read, chunk_frames=chunk_frames,
                        source_identity=dict(dataset_id=dataset_id, video_id=video_id))
    return video_id, result['frames'], time.perf_counter()-started


def plan():
    datasets = {}
    for name, (relative,kind) in DATASETS.items():
        source = PROJECT/relative
        grouped = source_rows(source,kind)
        metadata_path = source/('metadata/dataset.json' if kind=='ball' else 'dataset.json')
        meta = json.loads(metadata_path.read_text())
        total = 0
        for video_id,rows in grouped.items():
            if kind=='ball':
                total += sum(r['resolution']['width']*r['resolution']['height']*3 for r in rows)
            else:
                vm = json.loads((source/video_id/'metadata.json').read_text())
                total += len(rows)*vm['width']*vm['height']*3
        datasets[name] = dict(source=relative, kind=kind, dataset_id=meta['dataset_id'],
                              videos=len(grouped), frames=sum(map(len,grouped.values())),
                              estimated_bytes=total)
    return dict(schema=1,datasets=datasets,total_estimated_bytes=sum(d['estimated_bytes'] for d in datasets.values()),
                policy='copy existing decoded RGB pixels; keep all original labels, split files and image archives')


def migrate(workers=2, chunk_frames=32):
    if workers < 1 or chunk_frames < 1:raise ValueError('workers and chunk_frames must be positive')
    description = plan()
    required = description['total_estimated_bytes']
    # Account for chunks already committed when resuming an interrupted migration.
    existing=0
    for info in description['datasets'].values():
        key=identity(dict(dataset_id=info['dataset_id'],format='npy_rgb_u8_v1',chunk_frames=chunk_frames))
        existing+=sum(p.stat().st_size for p in (PROJECT/'data/cache/frames'/key[:20]).glob('*/*.npy'))
    if shutil.disk_usage(PROJECT).free < max(0,required-existing)+(20<<30):
        raise OSError('insufficient disk space; keep at least 20 GiB free')
    report_path = PROJECT/'out/workspace/frame_migration.json'
    save(report_path, dict(description,status='running',completed_videos=0))
    started = time.perf_counter()
    completed = 0
    stores = {}
    for name, info in description['datasets'].items():
        source = PROJECT/info['source']
        storage_id = identity(dict(dataset_id=info['dataset_id'],format='npy_rgb_u8_v1',chunk_frames=chunk_frames))
        store = PROJECT/'data/cache/frames'/storage_id[:20]
        grouped = source_rows(source,info['kind'])
        tasks = [(str(source),str(store),info['kind'],video_id,rows,info['dataset_id'],chunk_frames)
                 for video_id,rows in grouped.items()]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(migrate_video,t) for t in tasks]):
                video_id,frames,elapsed = future.result()
                completed += 1
                print(f'{name}: {video_id} {frames} frames verified, {elapsed:.1f}s; completed videos={completed}',flush=True)
                save(report_path, dict(description,status='running',completed_videos=completed,
                                       current_dataset=name,elapsed_sec=time.perf_counter()-started))
        manifests = {vid:json.loads((store/vid/'manifest.json').read_text()) for vid in grouped}
        save(store/'manifest.json',dict(schema=1,status='completed',storage_id=storage_id,
              dataset_id=info['dataset_id'],videos={v:dict(frames=m['frames'],source_key=m['source_key']) for v,m in manifests.items()},
              format='npy_rgb_u8_v1',chunk_frames=chunk_frames))
        target = PROJECT/'data/datasets'/name
        target.mkdir(parents=True,exist_ok=True)
        # Metadata remains byte-identical. Pixel paths are legacy provenance;
        # the frame-store adapter resolves actual pixels by video and frame IDs.
        for parent,dirs,files in os.walk(source,followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('frames','global_inputs','previews') and not (Path(parent)/d).is_symlink()]
            for filename in files:
                path = Path(parent)/filename
                if path.suffix not in ('.json','.jsonl') or path.name=='frame_store.json':continue
                relative = path.relative_to(source)
                destination = target/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists() and sha(destination) != sha(path):
                    raise ValueError(f'existing descriptor differs: {destination}')
                if not destination.exists():shutil.copy2(path,destination)
        save(target/'frame_store.json',dict(schema=1,path=os.path.relpath(store,target),storage_id=storage_id,
                                           logical_dataset_id=info['dataset_id'],legacy_source=info['source']))
        stores[name] = dict(dataset=str(target.relative_to(PROJECT)),store=str(store.relative_to(PROJECT)),
                            storage_id=storage_id,frames=info['frames'],videos=info['videos'])
    result = dict(description,status='completed',stores=stores,completed_videos=completed,
                  elapsed_sec=time.perf_counter()-started,pixel_verification='all frames exact',
                  array_bytes=sum(p.stat().st_size for p in (PROJECT/'data/cache/frames').glob('*/*/*.npy')))
    save(report_path,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--chunk-frames',type=int,default=32)
    args = parser.parse_args()
    if args.workers < 1:parser.error('workers must be positive')
    print(json.dumps(plan() if args.dry_run else migrate(args.workers,args.chunk_frames),indent=2))


if __name__ == '__main__':main()
