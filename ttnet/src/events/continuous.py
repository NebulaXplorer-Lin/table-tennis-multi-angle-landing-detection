"""Reviewed PP annotations, source-level mixing, and continuous landing selection.

All coordinates and frame tolerances are in the original PP video. Unknown
intervals override reviewed intervals. Net remains masked unless reviewed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from ttnet.common import DATA, PROJECT, identity, jsonl, save, sha
from ttnet.data_tools.frame_store import FrameStore
from ttnet.events.metrics import landing_match, peaks


def contains(intervals, fid):
    return any(a <= fid <= b for a,b in intervals)


def known(annotation, fid):
    return contains(annotation['reviewed_intervals'],fid) and not contains(annotation['unknown_intervals'],fid)


def validate_annotations(document, ball_root, require_review=True):
    ball_root=Path(ball_root)
    if document.get('schema') != 1:raise ValueError('unsupported continuous annotation schema')
    meta=json.loads((ball_root/'metadata/dataset.json').read_text())
    splits={p:jsonl(ball_root/f'splits/{p}.jsonl') for p in ('train','val')}
    actual={p:sha(ball_root/f'splits/{p}.jsonl') for p in splits}
    if document['ball_dataset_id']!=meta['dataset_id'] or document['ball_split_hashes']!=actual:
        raise ValueError('PP ball dataset/split identity changed')
    allowed={r['video_id']:dict(r,split=p) for p,rows in splits.items() for r in rows}
    label_rows={p:defaultdict(list) for p in ('train','val')}
    for p in label_rows:
        for row in jsonl(ball_root/f'labels/{p}.jsonl'):label_rows[p][row['video_id']].append(row)
    groups={};seen=set();counts=Counter()
    for v in document['videos']:
        vid=v['video_id']
        if vid in seen or vid not in allowed:raise ValueError('duplicate or non-training PP video')
        seen.add(vid);source=allowed[vid]
        if v['split']!=source['split'] or v['source_sha256']!=source['video_sha256']:
            raise ValueError('PP video split or source mismatch')
        if sha(PROJECT/source['video']) != v['source_sha256']:
            raise ValueError('PP source video content changed')
        group=v.get('match_group')
        if not group or group != source['group']:
            raise ValueError('match_group must preserve the frozen ball split group')
        if group in groups and groups[group]!=v['split']:raise ValueError('match group crosses train/val')
        groups[group]=v['split']
        if require_review and (not v.get('reviewed_by') or not v.get('view') or v.get('status')!='reviewed'):
            raise ValueError(f'{vid}: human review and view are required')
        rows=label_rows[v['split']][vid]
        n=len(rows)
        if [r['frame_id'] for r in rows]!=list(range(n)):raise ValueError('PP dataset must contain every original frame')
        for intervals in (v['reviewed_intervals'],v['unknown_intervals']):
            for a,b in intervals:
                if type(a) is not int or type(b) is not int or not 0<=a<=b<n:raise ValueError('invalid annotation interval')
        if require_review and any(not contains(v['reviewed_intervals'],i) and not contains(v['unknown_intervals'],i) for i in range(n)):
            raise ValueError(f'{vid}: full rally must be reviewed or explicitly unknown')
        event_keys=set()
        for e in v['events']:
            if e['kind'] not in ('bounce','net') or type(e['frame_id']) is not int or not 0<=e['frame_id']<n:
                raise ValueError('invalid event kind/frame')
            key=(e['kind'],e['frame_id'])
            if key in event_keys:raise ValueError('duplicate event')
            event_keys.add(key)
            if not known(v,e['frame_id']):raise ValueError('confirmed event overlaps unknown/unreviewed interval')
            if e['kind']=='net' and not v.get('net_reviewed',False):raise ValueError('unreviewed Net event')
            if e['kind']=='bounce' and not e.get('spatial_valid',True):
                if e.get('coordinate_source')!='missing_ball_position_gt' or e.get('x') is not None or e.get('y') is not None:
                    raise ValueError('missing spatial labels must be explicit and may not invent coordinates')
                continue
            if e['kind']=='bounce':
                w=rows[e['frame_id']]['resolution']['width'];h=rows[e['frame_id']]['resolution']['height']
                if not np.isfinite([e['x'],e['y']]).all() or not (0<=e['x']<w and 0<=e['y']<h):
                    raise ValueError('invalid original-image contact coordinate')
        counts[(v['split'],v.get('view'))]+=1
    if require_review:
        for part in ('train','val'):
            if not any(v['split']==part for v in document['videos']):raise ValueError(f'no PP {part} rallies')
    return dict(annotation_id=identity(document),counts={f'{p}/{v}':n for (p,v),n in counts.items()},
                train=sum(v['split']=='train' for v in document['videos']),val=sum(v['split']=='val' for v in document['videos']),
                views=sorted({v['view'] for v in document['videos'] if v.get('view')}))


def attach_pp(corpus, annotation_path, ball_root=DATA):
    document=json.loads(Path(annotation_path).read_text())
    audit=validate_annotations(document,ball_root)
    store=FrameStore(ball_root);base_image=corpus.image
    pp_ids=set();corpus.pp_annotations={}
    labels={p:jsonl(Path(ball_root)/f'labels/{p}.jsonl') for p in ('train','val')}
    for annotation in document['videos']:
        original=annotation['video_id'];vid='pp_'+original;part=annotation['split']
        if vid in corpus.videos:raise ValueError('duplicate PP identity')
        rows=[r for r in labels[part] if r['video_id']==original]
        frames={r['frame_id']:dict(frame_id=r['frame_id'],source_frame_id=r['frame_id'],timestamp_sec=r['timestamp_sec'],
                    label_status=r['label_status'],xy=r['ball_centers'][0] if r['ball_centers'] else None,path=r['frame_path']) for r in rows}
        resolution=rows[0]['resolution']
        corpus.videos[vid]=dict(meta=dict(n_frames=len(rows),fps=rows[0]['fps'],**resolution,source_identity=dict(soft_radius_sec=.04)),
                                frames=frames,events=annotation['events'])
        corpus.split[part].append(vid);pp_ids.add(vid);corpus.pp_annotations[vid]=annotation
        for r in rows:
            fid=r['frame_id']
            if not known(annotation,fid):continue
            # PP is sampled uniformly over verified times, including difficult negatives.
            target=[float(any(e['kind']==kind and e['frame_id']==fid for e in annotation['events'])) for kind in ('bounce','net')]
            corpus.samples[part].append(dict(video_id=vid,frame_id=fid,anchor=fid,group='pp_continuous',source='pp',
                original=True,target=target,hard_target=target,supervised=[True,bool(annotation.get('net_reviewed'))]))
    def image(vid,fid):
        if vid in pp_ids:return store.get_rgb(vid[3:],fid,path=corpus.videos[vid]['frames'][fid]['path'])
        return base_image(vid,fid)
    image.cache_clear=base_image.cache_clear
    corpus.image=image
    corpus.identity=identity(dict(openttgames=corpus.identity,pp=audit['annotation_id']))
    corpus.metadata=dict(corpus.metadata,pp_annotation_id=audit['annotation_id'],pp_review=audit)
    return audit


def mixed_samples(samples, seed, epoch):
    from ttnet.events.data import balanced_samples
    public=[s for s in samples if s.get('source')!='pp']
    pp=[s for s in samples if s.get('source')=='pp' and any(s['supervised'])]
    if not pp:raise ValueError('50/50 sampling requires reviewed PP training times')
    public=balanced_samples(public,seed,epoch)
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,2026]))
    indexes=[]
    while len(indexes)<len(public):indexes.extend(rng.permutation(len(pp)).tolist())
    paired=public+[pp[i] for i in indexes[:len(public)]]
    return [paired[i] for i in rng.permutation(len(paired))]


def _aggregate(results):
    tp=sum(r['tp'] for r in results);fp=sum(r['fp'] for r in results);fn=sum(r['fn'] for r in results)
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn)
    return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(1e-12,p+r))


def score_continuous(rows_by_video, annotations, threshold, nms_sec=.08):
    per_video={};by_view=defaultdict(list);by_resolution=defaultdict(list)
    for vid,rows in rows_by_video.items():
        a=annotations[vid]
        temporal_gt=[e for e in a['events'] if e['kind']=='bounce' and known(a,e['frame_id'])]
        gt=[e for e in temporal_gt if e.get('spatial_valid',True)]
        missing_xy=[e for e in temporal_gt if not e.get('spatial_valid',True)]
        # Apply postprocessing to the full video, then ignore unknown decisions.
        detected=[r for r in peaks(rows,[threshold,2.],nms_sec) if r['kind']=='bounce' and known(a,r['frame_id'])]
        landed=[dict(frame_id=r['frame_id'],x=r['prediction']['x'],y=r['prediction']['y']) for r in detected if r.get('prediction') and r['prediction']['valid']]
        landing=landing_match(landed,gt)
        # Match spatially scored GT first; missing-coordinate GT only absorbs
        # otherwise-unmatched predictions near its known event time.
        ignored=[i for i in landing['false_positives'] if any(abs(landed[i]['frame_id']-e['frame_id'])<=1 for e in missing_xy)]
        landing['false_positives']=[i for i in landing['false_positives'] if i not in ignored]
        landing['fp']-=len(ignored);landing['ignored']+=len(ignored)
        landing['precision']=landing['tp']/max(1,landing['tp']+landing['fp'])
        pp,rr=landing['precision'],landing['recall'];landing['f1']=2*pp*rr/max(1e-12,pp+rr)
        temporal=landing_match([dict(frame_id=r['frame_id'],x=0,y=0) for r in detected],
                               [dict(e,x=0,y=0) for e in temporal_gt])
        result=dict(landing=landing,continuous_bounce=temporal,predictions=landed,ground_truth=gt,detected=detected,
                    temporal_ground_truth=temporal_gt,spatial_gt_missing=len(missing_xy),spatial_gt_scored=len(gt),
                    missing_location_events=len(detected)-len(landed))
        per_video[vid]=result;by_view[a['view']].append(landing)
        by_resolution[a.get('resolution','unknown')].append(landing)
    views={v:_aggregate(r) for v,r in by_view.items()}
    return dict(per_video=per_video,by_view=views,by_resolution={v:_aggregate(r) for v,r in by_resolution.items()},
                landing=_aggregate([r['landing'] for r in per_video.values()]),
                continuous_bounce=_aggregate([r['continuous_bounce'] for r in per_video.values()]),
                macro_f1=float(np.mean([v['f1'] for v in views.values()])) if views else None,
                missing_views=max(0,3-len(views)),threshold=threshold,
                protocol=dict(frame_tolerance=1,original_pixel_tolerance=50,matching='maximum cardinality one-to-one',net='masked unless reviewed'))


def select_continuous_threshold(rows, annotations, grid, nms_sec=.08):
    results=[score_continuous(rows,annotations,t,nms_sec) for t in grid]
    if not results or results[0]['macro_f1'] is None:raise ValueError('continuous PP validation is required for model selection')
    best=max(results,key=lambda r:(r['macro_f1'],-r['landing']['fp'],r['threshold']))
    return best,[dict(threshold=r['threshold'],macro_f1=r['macro_f1'],fp=r['landing']['fp']) for r in results]


def validate_continuous(system, corpus, cache, grid):
    from ttnet.events.inference import predict_windows
    rows={};annotations={}
    for vid in corpus.split['val']:
        if vid not in corpus.pp_annotations:continue
        features,records=cache[vid]
        rows[vid]=predict_windows(system,corpus,vid,features,records)
        a=dict(corpus.pp_annotations[vid]);m=corpus.videos[vid]['meta']
        a['resolution']=f"{m['width']}x{m['height']}";annotations[vid]=a
    best,sweep=select_continuous_threshold(rows,annotations,grid,system.config['nms_sec'])
    return best,sweep,rows


def create_template(ball_root, dest):
    """Deterministic review queue. Views stay unknown until a person confirms them."""
    ball_root=Path(ball_root);meta=json.loads((ball_root/'metadata/dataset.json').read_text())
    videos=[]
    for part,count in [('train',12),('val',6)]:
        groups=defaultdict(list)
        for row in jsonl(ball_root/f'splits/{part}.jsonl'):groups[row['group']].append(row)
        chosen=[]
        while len(chosen)<count and any(groups.values()):
            for group in sorted(groups):
                if groups[group] and len(chosen)<count:chosen.append(groups[group].pop(0))
        for r in chosen:
            videos.append(dict(video_id=r['video_id'],split=part,match_group=r['group'],view=None,
                source=r['video'],source_sha256=r['video_sha256'],status='pending_review',reviewed_by=None,
                net_reviewed=False,reviewed_intervals=[],unknown_intervals=[],events=[],
                review_tags=['racket','floor','occlusion','video_boundary']))
    result=dict(schema=1,version='pp_continuous_v1',ball_dataset_id=meta['dataset_id'],
                ball_split_hashes={p:sha(ball_root/f'splits/{p}.jsonl') for p in ('train','val')},videos=videos)
    if Path(dest).exists():raise FileExistsError(dest)
    save(dest,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['template','validate']);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--ball-dataset',type=Path,default=DATA)
    a=p.parse_args()
    if a.action=='template':create_template(a.ball_dataset,a.annotations)
    else:print(json.dumps(validate_annotations(json.loads(a.annotations.read_text()),a.ball_dataset),indent=2))


if __name__=='__main__':main()
