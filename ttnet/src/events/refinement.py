"""Small, opt-in sampling and table-feature refinements on immutable caches."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from ttnet.common import DATA, now, save, sha


def stratified_pp_samples(pp, count, rng, positive_fraction, hard_fraction, hard_keys):
    if not 0 <= positive_fraction <= 1 or not 0 <= hard_fraction <= 1 - positive_fraction:
        raise ValueError('invalid PP sampling fractions')
    positives = [s for s in pp if s['supervised'][0] and s['target'][0] > 0]
    negatives = [s for s in pp if s['supervised'][0] and s['target'][0] == 0]
    hard = [s for s in negatives if (s['video_id'], s['frame_id']) in hard_keys]
    ordinary = [s for s in negatives if (s['video_id'], s['frame_id']) not in hard_keys]
    n_positive = int(count * positive_fraction)
    n_hard = min(int(count * hard_fraction), len(hard))

    def draw(pool, n):
        if not n:
            return []
        if not pool:
            raise ValueError('empty required PP sampling stratum')
        indexes = []
        while len(indexes) < n:
            indexes.extend(rng.permutation(len(pool)).tolist())
        return [pool[i] for i in indexes[:n]]

    return draw(positives, n_positive) + draw(hard, n_hard) + draw(ordinary, count-n_positive-n_hard)


def load_hard_negative_keys(path, corpus):
    from ttnet.events.data import sample_identity
    manifest = json.loads(Path(path).read_text())
    identity = manifest['identity']
    expected = dict(dataset_id=corpus.identity,
                    annotation_id=corpus.metadata['pp_annotation_id'],
                    train_samples_sha256=sample_identity(corpus.samples['train']),
                    train_split_sha256=sha(DATA/'splits/train.jsonl'))
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError('hard-negative training data identity mismatch')
    allowed = {(s['video_id'], s['frame_id']) for s in corpus.samples['train']
               if s.get('source') == 'pp' and s['supervised'][0] and s['target'][0] == 0}
    keys = {(r['video_id'], r['frame_id']) for r in manifest['samples']}
    if not keys <= allowed:
        raise ValueError('hard negatives must be reviewed training backgrounds')
    for vid, fid in keys:
        if any(e['kind'] == 'bounce' and abs(e['frame_id']-fid) <= 2
               for e in corpus.pp_annotations[vid]['events']):
            raise ValueError('hard negative overlaps excluded bounce neighborhood')
    return keys


def mine_hard_negatives(rows, annotation, threshold=.4):
    """Greedy peak suppression within +/-6 original frames, on known background."""
    from ttnet.events.continuous import known
    scores = {r['frame_id']: r['probabilities'][0] for r in rows}
    eligible = []
    for r in rows:
        fid = r['frame_id']
        if r['probabilities'][0] < threshold or not known(annotation, fid):
            continue
        if any((scores[other], -other) > (r['probabilities'][0], -fid)
               for other in range(fid-6,fid+7) if other in scores):
            continue
        if any(e['kind'] == 'bounce' and abs(e['frame_id']-fid) <= 2 for e in annotation['events']):
            continue
        eligible.append(r)
    selected = []
    for row in sorted(eligible, key=lambda r: (-r['probabilities'][0], r['frame_id'])):
        if all(abs(row['frame_id']-other['frame_id']) > 6 for other in selected):
            selected.append(row)
    return sorted(selected, key=lambda r: r['frame_id'])


def smooth_window_tables(ids, frames, records, config):
    mode = config.get('table_smoothing', 'none')
    if mode == 'none':
        return None
    if mode != 'window_ema':
        raise ValueError(f'unsupported table smoothing: {mode}')
    tau = float(config.get('table_smoothing_tau_sec', .15))
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('table smoothing tau must be finite and positive')
    output = {}; previous_id = previous_time = previous_raw = smoothed = None
    for fid in ids:
        table = records[fid].get('table') if fid is not None else None
        raw = np.asarray(table['bbox_xyxy'], dtype=float) if table and table.get('valid') else None
        valid = (fid is not None and frames[fid].get('frame_valid', True) and raw is not None
                 and raw.shape == (4,) and np.isfinite(raw).all() and (raw[2:] > raw[:2]).all())
        if not valid:
            if fid is not None:
                output[fid] = None
            previous_id = previous_time = previous_raw = smoothed = None
            continue
        timestamp = frames[fid]['timestamp_sec']
        reset = previous_id is None or fid != previous_id+1 or timestamp <= previous_time
        if not reset:
            intersection = np.maximum(0, np.minimum(raw[2:], previous_raw[2:])-
                                      np.maximum(raw[:2], previous_raw[:2])).prod()
            union = (raw[2:]-raw[:2]).prod()+(previous_raw[2:]-previous_raw[:2]).prod()-intersection
            reset = intersection/max(union, 1e-12) < .1
        if reset:
            smoothed = raw.copy()
        else:
            alpha = -math.expm1(-(timestamp-previous_time)/tau)
            smoothed = smoothed + alpha*(raw-smoothed)
        output[fid] = dict(table, bbox_xyxy=smoothed.tolist())
        previous_id, previous_time, previous_raw = fid, timestamp, raw
    return output


def main():
    p = argparse.ArgumentParser(description='Mine reviewed PP training backgrounds from frozen features.')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from ttnet.events.data import EventCorpus, sample_identity
    from ttnet.events.continuous import attach_pp
    from ttnet.events.engine import EventSystem, load_cache, cache_identity
    from ttnet.events.inference import predict_windows
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = state['config']; corpus = EventCorpus(cfg['dataset']); attach_pp(corpus, cfg['pp_annotations'])
    system = EventSystem(cfg, torch.device('cuda'), initialize=False)
    system.load_state_dict(state['system']); system.configure('frozen', False)
    cache = load_cache(corpus, args.cache, parts=('train',), expected=cache_identity(system, corpus))
    samples = []
    for vid in corpus.split['train']:
        if vid not in corpus.pp_annotations:
            continue
        features, records = cache[vid]
        rows = predict_windows(system, corpus, vid, features, records)
        mined = mine_hard_negatives(rows, corpus.pp_annotations[vid])
        samples.extend(dict(video_id=vid, frame_id=r['frame_id'], score=r['probabilities'][0]) for r in mined)
    save(args.output, dict(schema=1, created=now(),
         identity=dict(dataset_id=corpus.identity, annotation_id=corpus.metadata['pp_annotation_id'],
                       train_samples_sha256=sample_identity(corpus.samples['train']),
                       train_split_sha256=sha(DATA/'splits/train.jsonl'),
                       checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                       cache_identity=json.loads((args.cache/'identity.json').read_text())),
         protocol=dict(split='train', threshold=.4, excluded_bounce_radius_frames=2,
                       suppression_radius_frames=6), samples=samples))
    print(json.dumps(dict(hard_negatives=len(samples), output=str(args.output))))


if __name__ == '__main__':
    main()
