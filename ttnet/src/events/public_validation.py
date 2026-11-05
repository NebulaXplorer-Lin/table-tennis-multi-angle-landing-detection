"""Explicit PP-all training / public landing validation, with no public loss labels.

The historical PP adapter remains immutable. This adapter only changes a new
in-memory corpus and binds the new partition to source and implementation hashes.
Public images are decoded directly to rgb24, as in predict_video; landing GT is
kept separately from the frame metadata and never becomes a training sample.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
import copy
import json
from pathlib import Path

import av
import numpy as np

from ttnet.common import identity, jsonl, sha
from ttnet.events.metrics import landing_match, peaks


class _PublicRGBReader:
    """A seekable original-PTS decoder with at most three retained RGB frames."""
    def __init__(self, source, timestamps):
        self.container = av.open(str(source))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = 'AUTO'
        self.decoder = iter(self.container.decode(self.stream))
        self.timestamps = timestamps
        self.last_sec = float('-inf')
        self.frames = OrderedDict()

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None

    def get(self, fid):
        if fid in self.frames:
            self.frames.move_to_end(fid)
            return self.frames[fid]
        wanted = self.timestamps[fid]
        if wanted <= self.last_sec or wanted - self.last_sec > 2.:
            tick = round(wanted / self.stream.time_base)
            self.container.seek(tick, stream=self.stream, backward=True, any_frame=False)
            self.decoder = iter(self.container.decode(self.stream))
        for frame in self.decoder:
            if frame.pts is None:
                raise ValueError('public source has missing PTS')
            actual = float(frame.pts * frame.time_base)
            self.last_sec = actual
            if actual < wanted - 1e-7:
                continue
            if abs(actual - wanted) > 1e-7:
                raise ValueError(f'public source PTS changed: expected {wanted}, found {actual}')
            image = frame.to_ndarray(format='rgb24')
            image.setflags(write=False)
            self.frames[fid] = image
            while len(self.frames) > 3:
                self.frames.popitem(last=False)
            return image
        raise ValueError(f'public source decode ended before frame {fid}')


class _PublicRGBStore:
    """Keep one public video decoder open; no persistent RGB cache or re-encoding."""
    def __init__(self, sources):
        self.sources = sources
        self.reader = None
        self.video_id = None

    def image(self, vid, fid):
        if vid != self.video_id:
            self.close()
            source, timestamps = self.sources[vid]
            self.reader = _PublicRGBReader(source, timestamps)
            self.video_id = vid
        return self.reader.get(fid)

    def close(self):
        if self.reader is not None:
            self.reader.close()
        self.reader = None
        self.video_id = None

    def __del__(self):
        self.close()


def _disjoint(split):
    ids = [vid for part in ('train', 'val', 'test') for vid in split[part]]
    if len(ids) != len(set(ids)):
        raise ValueError('video appears in more than one train/val/test partition')


def _public_metadata(root, video):
    source = (root / video['path']).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError('public video must be an existing file inside the public root')
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != (video['width'], video['height']):
            raise ValueError('public manifest resolution differs from source')
        if abs(float(stream.average_rate) - video['fps']) / video['fps'] > .001:
            raise ValueError('public manifest FPS differs from source')
        timestamps = []
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('public source has missing PTS')
            timestamps.append(float(frame.pts * frame.time_base))
    if len(timestamps) != video['n_frames'] or not timestamps or np.any(np.diff(timestamps) <= 0):
        raise ValueError('public manifest frame count or source timestamp order is invalid')
    first = timestamps[0]
    frames = {fid:dict(frame_id=fid, source_frame_id=fid, timestamp_sec=ts-first,
                       label_status='unknown', xy=None)
              for fid, ts in enumerate(timestamps)}
    return source, timestamps, frames


def attach_public_validation(corpus, root, config=None):
    """Move historical PP val to training and append evaluation-only public videos.

    ``attach_pp`` must already have verified the original PP annotations. OpenTT
    game_5 stays in val for auxiliary sparse diagnostics. ``public_split_audit``
    is JSON serializable and can be saved alongside a new experiment's config.
    ``public_id_mapping`` maps manifest IDs to the prefixed corpus IDs.
    """
    if not getattr(corpus, 'pp_annotations', None):
        raise ValueError('attach_pp must run before public validation is attached')
    if hasattr(corpus, 'public_split_audit'):
        raise ValueError('public validation is already attached')
    config = config or {}
    root = Path(root).resolve()
    original_split = copy.deepcopy(corpus.split)
    _disjoint(original_split)
    manifest_path, gt_path = root/'manifest.json', root/'gt_public.jsonl'
    manifest, ground_truth = json.loads(manifest_path.read_text()), jsonl(gt_path)
    entries = manifest['videos']
    if not entries or len({v['video_id'] for v in entries}) != len(entries):
        raise ValueError('public manifest needs distinct video IDs')
    pp_ids = set(corpus.pp_annotations)
    if not pp_ids <= set(original_split['train'] + original_split['val']):
        raise ValueError('reviewed PP video is outside the original train/val split')
    moved = [vid for vid in original_split['val'] if vid in pp_ids]
    mapping = {v['video_id']:'public_' + v['video_id'] for v in entries}
    if set(mapping.values()) & set(corpus.videos):
        raise ValueError('public video ID collides with an existing corpus video')
    if any(s['video_id'].startswith('public_') for part in corpus.samples.values() for s in part):
        raise ValueError('public samples may never enter a supervised corpus')
    known_hashes = {a['source_sha256']:vid for vid,a in corpus.pp_annotations.items()}
    for vid, video in corpus.videos.items():
        digest = video['meta'].get('source_identity', {}).get('video_sha256')
        if digest:
            known_hashes[digest] = vid
    videos, sources, source_audit = {}, {}, []
    public_truth = {vid:[] for vid in mapping.values()}
    manifest_by_id = {v['video_id']:v for v in entries}
    for row in ground_truth:
        if row['video_id'] not in mapping:
            raise ValueError('public GT references a video outside the manifest')
        video = manifest_by_id[row['video_id']]
        if (type(row['frame_id']) is not int or not 0 <= row['frame_id'] < video['n_frames']
                or type(row['valid']) is not bool or not np.isfinite([row['x'], row['y']]).all()
                or not 0 <= row['x'] < video['width'] or not 0 <= row['y'] < video['height']):
            raise ValueError('invalid public landing GT')
        public_truth[mapping[row['video_id']]].append(dict(row))
    hashes = set()
    for video in entries:
        original, vid = video['video_id'], mapping[video['video_id']]
        source, timestamps, frames = _public_metadata(root, video)
        digest = sha(source)
        if digest in known_hashes or digest in hashes:
            raise ValueError('public source duplicates a PP/OpenTT or another public video')
        hashes.add(digest)
        valid = sum(g['valid'] for g in public_truth[vid])
        if video.get('n_valid_landings', valid) != valid:
            raise ValueError('public manifest valid-event count differs from GT')
        meta = dict(n_frames=video['n_frames'], fps=video['fps'], width=video['width'],
                    height=video['height'], source=str(source), public_video_id=original,
                    angle=video['angle'], source_identity=dict(video_sha256=digest,
                    soft_radius_sec=config.get('soft_radius_sec', .04)))
        videos[vid] = dict(meta=meta, frames=frames, events=[])
        sources[vid] = (source, timestamps)
        source_audit.append(dict(video_id=vid, public_video_id=original, path=video['path'],
            source_sha256=digest, frames=len(frames), angle=video['angle'], valid_events=valid,
            ignored_events=len(public_truth[vid])-valid,
            timestamp_identity=identity(timestamps)))
    split = copy.deepcopy(original_split)
    split['train'].extend(moved)
    split['val'] = [vid for vid in split['val'] if vid not in pp_ids] + list(mapping.values())
    split['split_id'] = config.get('public_split_id', f'pp{len(pp_ids)}_public{len(entries)}_event_v1')
    split['policy'] = 'All reviewed PP train+old-val train; public landing validation only; OpenTT partition retained.'
    _disjoint(split)
    samples = {part:list(rows) for part,rows in corpus.samples.items()}
    samples['train'].extend(s for s in samples['val'] if s['video_id'] in pp_ids)
    samples['val'] = [s for s in samples['val'] if s['video_id'] not in pp_ids]
    if any(s['video_id'] not in split[part] for part,rows in samples.items() for s in rows):
        raise ValueError('supervised sample does not belong to its new video partition')
    adapter_hash = sha(Path(__file__))
    provenance = dict(schema='pp_all_public_landing_v1', original_dataset_id=corpus.identity,
        original_split=original_split, split=split, moved_pp_validation_videos=moved,
        manifest_sha256=sha(manifest_path), gt_sha256=sha(gt_path), public_sources=source_audit,
        implementation_sha256=adapter_hash, pyav_version=av.__version__,
        decode_policy='all original frames, actual PTS, direct rgb24, one decoder/three retained images',
        public_ground_truth_role='landing evaluation only; no public frame/sample enters a loss',
        exact_source_overlap=False,
        overlap_scope='exact source SHA256; same-source re-encoded clips are not established by this audit')
    dataset_id = identity(provenance)
    audit = dict(provenance, dataset_id=dataset_id, split_id=split['split_id'],
        video_counts={part:len(split[part]) for part in ('train','val','test')},
        sample_counts={part:len(samples[part]) for part in ('train','val','test')},
        public_training_samples=0, public_id_mapping=mapping,
        split_identities={part:identity(split[part]) for part in ('train','val','test')})
    store = _PublicRGBStore(sources)
    original_image = corpus.image
    def image(vid, fid):
        return store.image(vid, fid) if vid in videos else original_image(vid, fid)
    def clear():
        store.close()
        original_image.cache_clear()
    image.cache_clear = clear
    corpus.split, corpus.samples = split, samples
    corpus.pp_annotations = {vid:dict(a, split='train') for vid,a in corpus.pp_annotations.items()}
    corpus.videos.update(videos)
    corpus.image = image
    corpus.public_frame_store = store
    corpus.public_ground_truth = public_truth
    corpus.public_video_ids = {prefixed:original for original,prefixed in mapping.items()}
    corpus.public_id_mapping = mapping
    corpus.public_split_audit = audit
    corpus.identity = dataset_id
    corpus.metadata = dict(corpus.metadata, public_validation=audit)
    return audit


def _aggregate(items):
    totals = {key:sum(item[key] for item in items) for key in ('tp','fp','fn')}
    p = totals['tp'] / max(1, totals['tp'] + totals['fp'])
    r = totals['tp'] / max(1, totals['tp'] + totals['fn'])
    return dict(totals, precision=p, recall=r, f1=2*p*r/max(1e-12,p+r))


def score_public_rows(corpus, rows_by_video, threshold, nms_sec=.08):
    """Score only public landing GT; ignored events absorb unmatched nearby XY."""
    if set(rows_by_video) != set(corpus.public_video_ids):
        raise ValueError('public validation must cover every manifest video exactly once')
    per_video, angles = {}, defaultdict(list)
    for vid,rows in rows_by_video.items():
        expected = list(range(corpus.videos[vid]['meta']['n_frames']))
        if [row['frame_id'] for row in rows] != expected:
            raise ValueError('public validation must infer every original frame in order')
        detections = [row for row in peaks(rows, [threshold, 2.], nms_sec) if row['kind']=='bounce']
        predictions = [dict(frame_id=r['frame_id'], x=r['prediction']['x'], y=r['prediction']['y'],
                            confidence=r['confidence']) for r in detections
                       if r.get('prediction') and r['prediction']['valid']]
        truth = corpus.public_ground_truth[vid]
        landing = landing_match(predictions, truth)
        result = dict(landing=landing, predictions=predictions,
                      ground_truth=[g for g in truth if g['valid']],
                      ignored_ground_truth=[g for g in truth if not g['valid']],
                      events_without_coordinates=len(detections)-len(predictions),
                      public_video_id=corpus.public_video_ids[vid])
        per_video[vid] = result
        angles[corpus.videos[vid]['meta']['angle']].append(landing)
    by_angle = {angle:_aggregate(items) for angle,items in angles.items()}
    return dict(per_video=per_video, by_angle=by_angle, by_view=by_angle,
        landing=_aggregate([v['landing'] for v in per_video.values()]),
        macro_f1=float(np.mean([m['f1'] for m in by_angle.values()])), threshold=threshold,
        events_without_coordinates=sum(v['events_without_coordinates'] for v in per_video.values()),
        protocol=dict(frame_tolerance=1, original_pixel_tolerance=50,
            matching='maximum cardinality one-to-one; valid GT first; ignored GT absorbs unmatched predictions',
            selection='pp_public validation; no public loss labels', net='not scored'))


def validate_public_cached(system, corpus, cache, grid):
    """Return best/sweep/rows with the existing continuous-validation call shape."""
    from ttnet.events.inference import predict_windows
    rows = {}
    for vid in corpus.public_video_ids:
        if vid not in corpus.split['val'] or vid in corpus.split['train']:
            raise ValueError('public video is not exclusive to validation')
        if any(s['video_id']==vid for samples in corpus.samples.values() for s in samples):
            raise ValueError('public validation video has supervised samples')
        features, records = cache[vid]
        rows[vid] = predict_windows(system, corpus, vid, features, records)
    results = [score_public_rows(corpus, rows, float(t), system.config['nms_sec']) for t in grid]
    if not results:
        raise ValueError('public validation threshold grid is empty')
    best = max(results, key=lambda r:(r['macro_f1'], -r['landing']['fp'], r['threshold']))
    sweep = [dict(threshold=r['threshold'], macro_f1=r['macro_f1'], fp=r['landing']['fp']) for r in results]
    return best, sweep, rows
