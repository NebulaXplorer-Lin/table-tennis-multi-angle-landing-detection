"""Publish PP + native OpenTT ball supervision and exact-PTS sparse context."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import av
import numpy as np

from ttnet.common import PROJECT, identity, jsonl, now, save, save_jsonl, sha


def ball_label(value, width, height):
    if value is None:
        return 'unknown', []
    x, y = float(value['x']), float(value['y'])
    if (x, y) == (-1., -1.):
        return 'negative', []
    if x == -1 or y == -1:
        return 'unknown', []
    if not np.isfinite([x, y]).all() or not (0 <= x < width and 0 <= y < height):
        raise ValueError(f'invalid OpenTT coordinate: {value}')
    return 'positive', [[x, y]]


def native_pts(video):
    """H264 one-packet-per-frame PTS index, checked against decoded samples."""
    with av.open(str(video)) as container:
        stream=container.streams.video[0]
        rate=float(stream.average_rate); count=stream.frames
        packets=[(p.pts,p.time_base) for p in container.demux(stream) if p.size and p.pts is not None]
        packets.sort(key=lambda p:p[0]*p[1])
        if len(packets)!=count or len({p[0] for p in packets})!=count:
            raise ValueError(f'packet/frame cardinality mismatch: {video}')
        absolute=np.array([float(pts*base) for pts,base in packets],np.float64)
        times=absolute-absolute[0]
        if np.any(np.diff(times)<=0) or not np.allclose(np.diff(times),1/rate,atol=1e-7,rtol=0):
            raise ValueError(f'non-CFR source needs decoded-frame indexing: {video}')
        # Seek by exact source PTS, then verify decoded timestamps at spread indices.
        for ordinal in np.linspace(0,count-1,9).round().astype(int):
            wanted=absolute[ordinal]
            container.seek(int(round(wanted/stream.time_base)),stream=stream,backward=True)
            found=False
            for frame in container.decode(stream):
                actual=float(frame.pts*frame.time_base)
                if actual < wanted-1e-7:continue
                if abs(actual-wanted)>1e-7:raise ValueError('decoded/packet PTS mismatch')
                found=True;break
            if not found:raise ValueError('missing decoded PTS sample')
        return times,packets


def prepare(output, pp=PROJECT/'data/datasets/pp_ball_v2', source=PROJECT/'data/ttnet'):
    output, pp, source = Path(output).resolve(), Path(pp).resolve(), Path(source).resolve()
    if (output/'metadata/dataset.json').exists():
        old = json.loads((output/'metadata/dataset.json').read_text())
        if old['status'] == 'completed':
            raise FileExistsError(f'published dataset is immutable: {output}')
    output.mkdir(parents=True, exist_ok=True)
    ppmeta = json.loads((pp/'metadata/dataset.json').read_text())
    split = json.loads((PROJECT/'ttnet/src/configs/event_split.json').read_text())
    all_games = [g for p in ('train', 'val', 'test') for g in split[p]]
    if len(all_games) != len(set(all_games)):
        raise ValueError('OpenTT split overlap')
    labels, splits, videos, source_hashes = {}, {}, {}, []
    routes = {}
    for part in ('train', 'val', 'test'):
        labels[part] = [dict(r, source_domain='pp') for r in jsonl(pp/f'labels/{part}.jsonl')] if part != 'test' else []
        splits[part] = [dict(r, source_domain='pp') for r in jsonl(pp/f'splits/{part}.jsonl')] if part != 'test' else []
        for entry in splits[part]:
            routes[entry['video_id']] = 'pp'
        origin = 'test' if part == 'test' else 'training'
        for game in split[part]:
            vid = 'ott_' + game
            video = source/origin/'videos'/f'{game}.mp4'
            annotation = source/origin/'annotations'/game/'ball_markup.json'
            raw = json.loads(annotation.read_text())
            hashes = dict(video_sha256=sha(video), annotation_sha256=sha(annotation))
            source_hashes.append(dict(video_id=vid, **hashes))
            store = output/'native_store'/vid
            index = output/'metadata/videos'/f'{vid}.json'
            context_file = output/'context'/f'{vid}.jsonl'
            if index.is_file():
                meta = json.loads(index.read_text())
                if meta['source_hashes'] != hashes:
                    raise ValueError(f'changed native source: {vid}')
                context = jsonl(context_file)
            else:
                with av.open(str(video)) as container:
                    stream = container.streams.video[0]
                    stream.codec_context.thread_count = 2
                    w, h = stream.width, stream.height
                    supervision = {int(k) for k,v in raw.items() if ball_label(v,w,h)[0] != 'unknown'}
                    needed = {fid+d for fid in supervision for d in (-1,0,1) if 0 <= fid+d < stream.frames}
                    timestamps,packets=native_pts(video)
                    context=[]
                    for fid in sorted(needed):
                        pts,base=packets[fid]
                        ts=float(timestamps[fid])
                        status, centers = ball_label(raw.get(str(fid)), w, h)
                        context.append(dict(video_id=vid, split=part, source_domain='opentt',
                            native_sparse=True, frame_id=fid, source_frame_id=fid, timestamp_sec=ts,
                            pts=pts, time_base_num=base.numerator,
                            time_base_den=base.denominator, fps=float(stream.average_rate),
                            nominal_fps=float(stream.average_rate), resolution=dict(width=w,height=h),
                            ball_boxes=[], ball_centers=centers, label_status=status,
                            frame_path=f'native/{vid}/{fid:06d}.rgb'))
                    if len(timestamps) != stream.frames or {r['frame_id'] for r in context} != needed:
                        raise ValueError(f'incomplete source decode: {vid}')
                    if any(fid < 0 or fid >= len(timestamps) for fid in map(int,raw)):
                        raise ValueError(f'annotation outside source: {vid}')
                    meta = dict(video_id=vid, frames=len(timestamps), width=w, height=h,
                                fps=float(stream.average_rate), source_hashes=hashes,
                                video=str(video.relative_to(PROJECT)), annotation=str(annotation.relative_to(PROJECT)))
                for row in context:
                    row['video_frame_count'] = len(timestamps)
                store.mkdir(parents=True, exist_ok=True)
                np.save(store/'all_timestamps.npy', np.array(timestamps,np.float64))
                save_jsonl(context_file, context)
                save(index, meta)
            store.mkdir(parents=True, exist_ok=True)
            np.save(store/'frame_ids.npy', np.array([r['frame_id'] for r in context],np.int64))
            np.save(store/'timestamps.npy', np.array([r['timestamp_sec'] for r in context],np.float64))
            save(store/'manifest.json', dict(schema=1,status='completed',video_id=vid,frames=len(context),
                source=os.path.relpath(video,output/'native_store'), source_bytes=video.stat().st_size,
                shape=[meta['height'],meta['width'],3], pixel_policy='lossless_bgr_to_rgb'))
            centers = [r for r in context if r['label_status'] in ('positive','negative')]
            labels[part].extend(centers)
            entry = dict(video_id=vid, source_domain='opentt', split=part, video=meta['video'],
                         annotation=meta['annotation'], **hashes)
            splits[part].append(entry)
            routes[vid] = 'opentt'
            videos[vid] = meta
            print(f'{part} {vid}: {len(centers)} supervision centers, {len(context)} native context frames', flush=True)
    duplicates = []
    all_entries = [e for part in splits.values() for e in part]
    by_hash = {}
    for entry in all_entries:
        previous = by_hash.get(entry['video_sha256'])
        if previous and previous['split'] != entry['split']:
            duplicates.append([previous['video_id'],entry['video_id']])
        by_hash[entry['video_sha256']] = entry
    if duplicates:
        raise ValueError(f'cross-split duplicate source videos: {duplicates}')
    for part in ('train','val','test'):
        save_jsonl(output/f'labels/{part}.jsonl',labels[part])
        save_jsonl(output/f'splits/{part}.jsonl',splits[part])
    split_hashes = {part:sha(output/f'splits/{part}.jsonl') for part in splits}
    provenance = dict(pp_dataset_id=ppmeta['dataset_id'],pp_split_hashes=ppmeta['split_hashes'],
                      opentt_split=split, sources=source_hashes, policy='native120_fixed3_unknown_masked_v1',
                      preparation_sha256=sha(__file__))
    dataset_id = identity(provenance)
    save(output/'native_store/manifest.json',dict(status='completed',format='source_video_rgb_v1',storage_id=dataset_id))
    save(output/'native_descriptor/frame_store.json',dict(path='../native_store',storage_id=dataset_id))
    save(output/'frame_store.json',dict(schema=1,format='routed_rgb_v1',storage_id=dataset_id,
        sources=dict(pp=os.path.relpath(pp,output),opentt='native_descriptor'),video_sources=routes))
    counts={part:dict(Counter(r['source_domain'] for r in labels[part] if r['label_status'] in ('positive','negative'))) for part in labels}
    save(output/'metadata/dataset.json',dict(schema=1,status='completed',dataset_id=dataset_id,
         split_id='pp_opentt_ball_mixed_v1',split_hashes=split_hashes,
         train_videos=len(splits['train']),val_videos=len(splits['val']),counts=counts,
         source_datasets=dict(pp=dict(dataset_id=ppmeta['dataset_id'],split_hashes=ppmeta['split_hashes'])),
         provenance=provenance, created=now(), source_overlap_audit='pending perceptual audit before training'))
    (output/'README.md').write_text('# PP + OpenTTGames native ball v1\n\nPP fixed 114/40; OpenTT game_1–4 train, game_5 val, test_1–7 test.\nNative three-frame contexts use exact source IDs/PTS; missing annotations stay unknown.\nPP labels and source pixels are referenced, not replaced. Training sources are balanced 50/50.\n')
    return dataset_id


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'data/datasets/pp_opentt_ball_v1')
    args=parser.parse_args()
    print(prepare(args.output),flush=True)


if __name__=='__main__':main()
