"""Immutable 25fps sparse event dataset; decode originals without inventing negatives."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

import av
import cv2

from ttnet.common import CONFIGS, PROJECT, identity, jsonl, now, save, save_jsonl, sha

CLASSES = ("bounce", "net", "empty_event")


def nearest_index(value):
    # Explicit half-up tie breaking; independent of Python's round-to-even.
    return int(value + .5)


def target_samples(events, frame_times=None, soft_radius_sec=.04, fps=25):
    """Preserve original event times; only explicitly annotated anchors are hard labels."""
    by_center = {}
    for e in events:
        by_center.setdefault(e['frame_id'], set()).add(e['kind'])
    samples = []
    for center, kinds in sorted(by_center.items()):
        group = next((k for k in CLASSES[:2] if k in kinds), 'empty_event')
        if 'empty_event' in kinds and len(kinds) > 1:
            # Distinct 120fps instants may collide at 25fps: never label a positive negative.
            kinds = kinds - {'empty_event'}
        for shift in range(-2, 3):
            if (group == 'empty_event' and shift != 0) or center + shift < 0:
                continue
            fid=center+shift
            if frame_times is not None and fid not in frame_times:continue
            timestamp=frame_times[fid] if frame_times is not None else fid/fps
            target, mask = [], []
            for kind in CLASSES[:2]:
                relevant=[e for e in events if e['kind']==kind and abs(e['timestamp_sec']-timestamp)<=soft_radius_sec+1e-9]
                target.append(max([max(0.,1-abs(e['timestamp_sec']-timestamp)/soft_radius_sec) for e in relevant], default=0.))
                mask.append(shift==0 or kind in kinds or bool(relevant))
            samples.append(dict(frame_id=fid, anchor=center, group=group, target=target,
                                hard_target=[float(k in kinds) for k in CLASSES[:2]],
                                event_times_sec=[e['timestamp_sec'] for e in events if e['frame_id']==center and e['kind']!='empty_event'],
                                supervised=mask, original=shift==0))
    return samples


def prepare_video(source, annotations, dest, fps=25, full=False, limit=None, soft_radius_sec=.04):
    source, annotations, dest = Path(source), Path(annotations), Path(dest)
    source_identity = dict(video_sha256=sha(source), events_sha256=sha(annotations/'events_markup.json'),
                           ball_sha256=sha(annotations/'ball_markup.json'), target_fps=fps,
                           full=full, limit=limit, soft_radius_sec=soft_radius_sec, schema=3, code_sha256=sha(__file__))
    if (dest/'metadata.json').exists():
        meta = json.loads((dest/'metadata.json').read_text())
        if meta['source_identity'] != source_identity:
            raise ValueError(f"dataset identity changed: {dest}; choose a new output directory")
        rows = jsonl(dest/'frames.jsonl')
        if all((dest/r['path']).is_file() for r in rows):
            print(f"reuse {dest.name}: {len(rows)} frames", flush=True)
            return meta
        raise ValueError(f"incomplete previously published dataset: {dest}")
    raw_events = json.loads((annotations/'events_markup.json').read_text())
    ball = json.loads((annotations/'ball_markup.json').read_text())
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        rate = Fraction(stream.average_rate)
        n_source = stream.frames
        if not n_source or not stream.duration:
            raise ValueError("frame count and duration required")
        n_target = int((n_source-1)*fps/rate)+1
        events = []
        for fid, kind in sorted(raw_events.items(), key=lambda x: int(x[0])):
            if kind not in CLASSES or not 0 <= int(fid) < n_source:
                raise ValueError(f"invalid annotation: {fid} {kind}")
            target_id=min(n_target-1,nearest_index(int(fid)*fps/rate))
            events.append(dict(source_frame_id=int(fid), frame_id=target_id, timestamp_sec=float(int(fid)/rate),
                               kind=kind,contact_xy=ball.get(fid)))
        if limit:
            # Smoke/pilot data has a different identity and is never a formal split.
            events = events[:limit]
        samples = [s for s in target_samples(events,soft_radius_sec=soft_radius_sec,fps=fps) if s['frame_id'] < n_target]
        # ±2 anchor shifts + event radius6 + localizer radius1 = ±9 frames.
        needed = set(range(n_target)) if full else {i for s in samples for i in range(max(0,s['frame_id']-7), min(n_target,s['frame_id']+8))}
        mapping = {nearest_index(Fraction(i)*rate/fps): i for i in sorted(needed)}
        if len(mapping) != len(needed):
            raise ValueError("upsampling/repeated frames is not supported")
        dest.mkdir(parents=True, exist_ok=True)
        (dest/'frames').mkdir(exist_ok=True)
        rows, first_pts, max_error = [], None, 0.
        last_source = max(mapping)
        stream.thread_type = 'AUTO'
        event_by_source={}
        for event in events:event_by_source.setdefault(event['source_frame_id'],[]).append(event)
        for source_id, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError("missing PTS")
            timestamp = float(frame.pts*frame.time_base)
            if first_pts is None:
                first_pts = timestamp
            for event in event_by_source.get(source_id,[]):event['timestamp_sec']=timestamp-first_pts
            if source_id not in mapping:
                if source_id > last_source:
                    break
                continue
            fid = mapping[source_id]
            actual = timestamp-first_pts
            error = abs(actual-fid/fps)
            max_error = max(max_error,error)
            if error > .51/float(rate):
                raise ValueError(f"PTS/CFR mismatch at {source_id}: {error}")
            path = f'frames/{fid:06d}.jpg'
            bgr = frame.to_ndarray(format='bgr24')
            if not cv2.imwrite(str(dest/path),bgr,[cv2.IMWRITE_JPEG_QUALITY,95]):
                raise OSError(f"cannot save {path}")
            b = ball.get(str(source_id))
            status, xy = 'unknown', None
            if b is not None:
                x,y = float(b['x']),float(b['y'])
                if (x,y)==(-1.,-1.):status='negative'
                elif 0<=x<frame.width and 0<=y<frame.height:status,xy='positive',[x,y]
                elif x == -1 or y == -1:
                    # Partially missing coordinates are not certain no-ball labels.
                    status = 'unknown'
                else:raise ValueError(f"invalid ball point {source_id}: {b}")
            rows.append(dict(frame_id=fid, source_frame_id=source_id, timestamp_sec=actual,
                             path=path, label_status=status, xy=xy))
        if {r['frame_id'] for r in rows} != needed:
            raise ValueError("incomplete decode")
        save_jsonl(dest/'frames.jsonl', rows)
        save_jsonl(dest/'events.jsonl', events)
        samples=target_samples(events,{r['frame_id']:r['timestamp_sec'] for r in rows},soft_radius_sec,fps)
        save_jsonl(dest/'samples.jsonl', samples)
        centers={e['frame_id'] for e in events}
        collisions=[dict(frame_id=f,events=[e for e in events if e['frame_id']==f]) for f in sorted(centers)
                    if len([e for e in events if e['frame_id']==f])>1]
        meta = dict(source_identity=source_identity, source=str(source.resolve()), fps=fps,
                    source_fps=str(rate), n_frames=n_target, source_frames=n_source,
                    width=stream.width,height=stream.height, decoded_cached_frames=len(rows),
                    events=dict(Counter(e['kind'] for e in events)), max_pts_error_sec=max_error,
                    ball_labels=dict(Counter(r['label_status'] for r in rows)),
                    frame_index_base=0, jpeg_quality=95, full=full, mapping_collisions=collisions)
        save(dest/'metadata.json',meta)
        print(f"prepared {dest.name}: {len(rows)} frames, events={meta['events']}",flush=True)
        return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=PROJECT/'data/ttnet')
    p.add_argument('--output',type=Path,default=PROJECT/'ttnet/data/datasets/event25_v3')
    p.add_argument('--split',type=Path,default=CONFIGS/'event_split.json')
    p.add_argument('--part',choices=['train','val','test','all'],default='all')
    p.add_argument('--limit',type=int)
    a=p.parse_args();cv2.setNumThreads(1)
    split=json.loads(a.split.read_text()); parts=['train','val','test'] if a.part=='all' else [a.part]
    all_games=[g for k in ['train','val','test'] for g in split[k]]
    if len(all_games)!=len(set(all_games)):raise ValueError('split leakage')
    a.output.mkdir(parents=True,exist_ok=True)
    save(a.output/'split.json',split)
    for part in parts:
        for game in split[part]:
            origin='test' if part=='test' else 'training'
            prepare_video(a.source/origin/'videos'/f'{game}.mp4',a.source/origin/'annotations'/game,
                          a.output/game,limit=a.limit)
    # Dataset identity is complete only when every declared match is prepared.
    metas={g:json.loads((a.output/g/'metadata.json').read_text()) for g in all_games if (a.output/g/'metadata.json').exists()}
    save(a.output/'dataset.json',dict(dataset_id=identity(dict(split=split,videos=metas)),split_id=split['split_id'],
                                    prepared=list(metas),complete=len(metas)==len(all_games),pilot=bool(a.limit),updated=now()))


if __name__=='__main__':main()
