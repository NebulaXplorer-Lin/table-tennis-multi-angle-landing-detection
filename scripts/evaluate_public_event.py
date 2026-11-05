"""Complete native-frame PP public validation of a frozen event checkpoint.

This is validation on pp_public, including when it selects a checkpoint or
hyperparameters. It is not an independent external-test claim. Inference sees
only source video pixels; labels are loaded separately for scoring/overlays.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import av
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttnet.common import PROJECT, SRC, identity, jsonl, save, save_jsonl, sha
from ttnet.events.data import EventCorpus
from ttnet.events.engine import EventSystem, cache_identity, load_cache
from ttnet.events.evaluate import paint
from ttnet.events.inference import predict_video, predict_windows
from ttnet.events.metrics import landing_match, peaks
from ttnet.scene.visual_cache import trajectory_records


PUBLIC_EXPECTED = dict(videos=15, frames=2109, valid_gt=92, ignored_gt=12)


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def measure(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r,
                f1=2 * p * r / max(1e-12, p + r))


def thresholds_from(state, override=None):
    values = override if override is not None else state.get('thresholds')
    if values is None or len(values) != 2 or any(not np.isfinite(t) or not 0 <= t <= 1 for t in values):
        raise ValueError('checkpoint must contain two finite thresholds in [0, 1], or supply --thresholds BOUNCE NET')
    return [float(v) for v in values]


def public_metadata(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    gt = jsonl(root / 'gt_public.jsonl')
    ids = [v['video_id'] for v in manifest['videos']]
    if len(set(ids)) != len(ids) or set(g['video_id'] for g in gt) - set(ids):
        raise ValueError('duplicate manifest ID or unknown GT video')
    counts = dict(videos=len(ids), frames=sum(v['n_frames'] for v in manifest['videos']),
                  valid_gt=sum(bool(g.get('valid', True)) for g in gt),
                  ignored_gt=sum(not g.get('valid', True) for g in gt))
    if counts != PUBLIC_EXPECTED:
        raise ValueError(f'pp_public identity has unexpected counts: {counts}; expected {PUBLIC_EXPECTED}')
    for g in gt:
        video = next(v for v in manifest['videos'] if v['video_id'] == g['video_id'])
        if not 0 <= g['frame_id'] < video['n_frames'] or not all(np.isfinite(g[k]) for k in ('x', 'y')):
            raise ValueError('invalid public GT coordinates/frame')
    return manifest, gt, counts


def scan_video(source, expected):
    """Actually decode every source frame; header frame counts alone do not pass."""
    times = []
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        dimensions = (stream.width, stream.height)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f'missing source PTS: {source}')
            times.append(float(frame.pts * frame.time_base))
            if (frame.width, frame.height) != dimensions:
                raise ValueError(f'source resolution changes: {source}')
    if len(times) != expected['n_frames'] or dimensions != (expected['width'], expected['height']):
        raise ValueError(f'incomplete/incorrect source decode: {source}: {len(times)} frames, {dimensions}')
    if not times or np.any(np.diff(times) <= 0):
        raise ValueError(f'non-monotonic source PTS: {source}')
    return dict(frames=len(times), width=dimensions[0], height=dimensions[1], fps=rate,
                relative_timestamps_sec=[t - times[0] for t in times], decode_complete=True)


def validate_rows(rows, records, video, source_times=None):
    expected = list(range(video['n_frames']))
    if [r['frame_id'] for r in rows] != expected or sorted(records) != expected:
        raise ValueError(f'incomplete native-frame coverage: {video["video_id"]}')
    for row in rows:
        fid = row['frame_id']
        probabilities = row['probabilities']
        if row.get('source_frame_id') != fid or len(probabilities) != 2:
            raise ValueError(f'source/frame mapping changed: {video["video_id"]}:{fid}')
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in probabilities):
            raise ValueError('invalid event probability')
        if not np.isfinite(row['timestamp_sec']):
            raise ValueError('non-finite prediction timestamp')
        if fid and row['timestamp_sec'] <= rows[fid - 1]['timestamp_sec']:
            raise ValueError('non-monotonic prediction timestamp')
        if source_times is not None and abs(row['timestamp_sec'] - source_times[fid]) > 1e-6:
            raise ValueError('online timestamp differs from decoded source')
        pred = row.get('prediction')
        if pred and pred.get('valid', False) and any(not np.isfinite(pred[k]) for k in ('x', 'y')):
            raise ValueError('non-finite predicted coordinate')


def score_video(rows, truth, thresholds, nms_sec=.08):
    spotted = peaks(rows, thresholds, nms_sec=nms_sec)
    bounces = [dict(video_id=r.get('video_id'), frame_id=r['frame_id'], timestamp_sec=r['timestamp_sec'],
                    x=r['prediction']['x'], y=r['prediction']['y'], confidence=r['confidence'])
               for r in spotted if r['kind'] == 'bounce' and r.get('prediction') and r['prediction'].get('valid', False)]
    missing = [r for r in spotted if r['kind'] == 'bounce' and not (r.get('prediction') and r['prediction'].get('valid', False))]
    result = landing_match(bounces, truth, time_tolerance=1, distance=50)
    valid = [g for g in truth if g.get('valid', True)]
    errors = []
    for i in result['false_positives']:
        errors.append(dict(kind='FP', frame_id=bounces[i]['frame_id'], prediction_index=i, prediction=bounces[i]))
    for i in result['false_negatives']:
        errors.append(dict(kind='FN', frame_id=valid[i]['frame_id'], ground_truth_index=i, ground_truth=valid[i]))
    for row in missing:
        errors.append(dict(kind='event_without_coordinate', frame_id=row['frame_id'], event=row))
    result.update(valid_gt=len(valid), ignored_gt=len(truth) - len(valid), predictions=len(bounces),
                  events_without_coordinates=len(missing), selected_bounce_peaks=len(bounces) + len(missing))
    return result, bounces, missing, spotted, errors


def summarize(per_video, manifest):
    by_angle = {}
    for angle in sorted({v['angle'] for v in manifest['videos']}):
        subset = [per_video[v['video_id']] for v in manifest['videos'] if v['angle'] == angle]
        by_angle[angle] = measure(*(sum(v[k] for v in subset) for k in ('tp', 'fp', 'fn')))
        by_angle[angle]['ignored'] = sum(v['ignored'] for v in subset)
    return dict(overall=measure(*(sum(v[k] for v in per_video.values()) for k in ('tp', 'fp', 'fn'))),
                by_angle=by_angle, macro_f1=sum(v['f1'] for v in by_angle.values()) / len(by_angle),
                events_without_coordinates=sum(v['events_without_coordinates'] for v in per_video.values()))


def landing_decisions(result):
    return {key:result[key] for key in ('tp', 'fp', 'fn', 'ignored', 'matches', 'false_positives', 'false_negatives', 'events_without_coordinates')}


def event_signature(rows, thresholds, nms_sec):
    return [(r['kind'], r['frame_id']) for r in peaks(rows, thresholds, nms_sec=nms_sec)]


def compare_frames(cached, online, thresholds, nms_sec=.08, probability_atol=2e-4, coordinate_atol=.1):
    """Check decisions as well as numeric tolerances; small flips must not pass."""
    if [r['frame_id'] for r in cached] != [r['frame_id'] for r in online]:
        raise ValueError('cache/online frame coverage differs')
    counts = Counter()
    differences = []
    probability_max = coordinate_max = timestamp_max = 0.
    for a, b in zip(cached, online):
        prob_error = float(np.max(np.abs(np.array(a['probabilities']) - np.array(b['probabilities']))))
        probability_max = max(probability_max, prob_error)
        timestamp_error = abs(a['timestamp_sec'] - b['timestamp_sec'])
        timestamp_max = max(timestamp_max, timestamp_error)
        pa, pb = a.get('prediction'), b.get('prediction')
        va, vb = bool(pa and pa.get('valid', False)), bool(pb and pb.get('valid', False))
        coord_error = max(abs(pa[k] - pb[k]) for k in ('x', 'y')) if va and vb else None
        if coord_error is not None:
            coordinate_max = max(coordinate_max, coord_error)
        kinds = []
        checks = dict(probabilities_unequal=a['probabilities'] != b['probabilities'],
                      probabilities_outside_atol=prob_error > probability_atol,
                      threshold_decision_differences=any((x >= t) != (y >= t) for x, y, t in zip(a['probabilities'], b['probabilities'], thresholds)),
                      coordinate_validity_differences=va != vb,
                      coordinates_unequal=coord_error is not None and coord_error != 0,
                      coordinates_outside_atol=coord_error is not None and coord_error > coordinate_atol,
                      timestamp_unequal=timestamp_error != 0,
                      timestamp_outside_atol=timestamp_error > 1e-6,
                      source_frame_differences=a.get('source_frame_id') != b.get('source_frame_id'))
        for key, value in checks.items():
            if value:
                counts[key] += 1
                kinds.append(key)
        if kinds:
            differences.append(dict(frame_id=a['frame_id'], differences=kinds, probability_max_abs=prob_error,
                                    coordinate_max_abs_pixels=coord_error, cached=a, online=b))
    cache_events = event_signature(cached, thresholds, nms_sec)
    online_events = event_signature(online, thresholds, nms_sec)
    decision_equal = not any(counts[k] for k in ('threshold_decision_differences', 'coordinate_validity_differences', 'source_frame_differences')) and cache_events == online_events
    numeric_equal = not any(counts[k] for k in ('probabilities_outside_atol', 'coordinates_outside_atol', 'timestamp_outside_atol'))
    exact = not differences and cache_events == online_events
    return dict(frames=len(online), exact=exact, within_tolerance=numeric_equal, decisions_equal=decision_equal,
                passed=numeric_equal and decision_equal, counts=dict(counts),
                probability_max_abs=probability_max, coordinate_max_abs_pixels=coordinate_max,
                timestamp_max_abs_sec=timestamp_max, cache_peak_decisions=cache_events, online_peak_decisions=online_events,
                peak_decisions_equal=cache_events == online_events,
                probability_atol=probability_atol, coordinate_atol_pixels=coordinate_atol), differences


def overlays(source, video, rows, records, truth, predictions, match, errors, dest, thresholds):
    """Every source frame, with all scored errors plus no-coordinate events covered."""
    vid = video['video_id']
    video_path = dest / 'overlays' / f'{vid}.mp4'
    video_path.parent.mkdir(parents=True, exist_ok=True)
    wanted = {e['frame_id'] for e in errors}
    valid = [g for g in truth if g.get('valid', True)]
    pred_status = {i: 'TP' for i in range(len(predictions))}
    for i in match['false_positives']:
        pred_status[i] = 'FP'
    matched = {r['prediction'] for r in match['matches']}
    for i in range(len(predictions)):
        if i not in matched and i not in match['false_positives']:
            pred_status[i] = 'ignored'
    requested = defaultdict(list)
    for e in errors:
        requested[e['frame_id']].append(e['kind'])
    frames = 0
    writer = None
    try:
        with av.open(str(source)) as container:
            for fid, frame in enumerate(container.decode(video=0)):
                if fid >= len(rows):
                    raise ValueError('overlay decode exceeds inference coverage')
                near_pred = [(i, p) for i, p in enumerate(predictions) if abs(p['frame_id'] - fid) <= 1]
                labels = [g for g in valid if abs(g['frame_id'] - fid) <= 1]
                kinds = ','.join(sorted(set(requested[fid]))) or 'full-video validation'
                current = near_pred[0][1] if near_pred else records[fid].get('prediction')
                im = paint(frame.to_ndarray(format='rgb24'), vid, fid, kinds, current, labels, rows[fid]['probabilities'],
                           label_note=f'event GT +/-1f; thresholds={thresholds}; ball labels unknown')
                scale = im.shape[1] / frame.width
                for g in truth:
                    if not g.get('valid', True) and abs(g['frame_id'] - fid) <= 1:
                        point = (round(g['x'] * scale), round(g['y'] * scale))
                        cv2.drawMarker(im, point, (0, 230, 230), cv2.MARKER_CROSS, 16, 2)
                        cv2.putText(im, f'ignored GT f={g["frame_id"]}', point, 0, .4, (0, 230, 230), 1)
                trajectory = records[fid].get('prediction')
                if trajectory and trajectory.get('valid', False):
                    cv2.circle(im, (round(trajectory['x'] * scale), round(trajectory['y'] * scale)), 4, (255, 255, 0), 1)
                for i, p in near_pred:
                    point = (round(p['x'] * scale), round(p['y'] * scale))
                    cv2.circle(im, point, 8, (60, 60, 255), 2)
                    cv2.putText(im, f'{pred_status[i]} f={p["frame_id"]} conf={p["confidence"]:.3f}',
                                (max(0, point[0] - 40), max(125, point[1] - 12)), 0, .4, (60, 60, 255), 1)
                for g in labels:
                    label = 'FN' if g in [valid[i] for i in match['false_negatives']] else 'matched GT'
                    point = (round(g['x'] * scale), round(g['y'] * scale))
                    cv2.putText(im, f'{label} f={g["frame_id"]}', (max(0, point[0] - 40), point[1] + 18), 0, .4, (60, 240, 60), 1)
                if writer is None:
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), float(video['fps']), (im.shape[1], im.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('cannot open full-video overlay writer')
                writer.write(im)
                frames += 1
                if fid in wanted:
                    path = dest / 'images' / f'{vid}_{fid:06d}.jpg'
                    path.parent.mkdir(exist_ok=True)
                    if not cv2.imwrite(str(path), im):
                        raise RuntimeError(f'cannot save {path}')
    finally:
        if writer is not None:
            writer.release()
    if frames != video['n_frames']:
        raise ValueError('overlay failed to cover the complete source')
    # Reopen and fully decode the encoded artifact, not merely its container header.
    with av.open(str(video_path)) as container:
        verified = sum(1 for _ in container.decode(video=0))
    if verified != frames:
        raise ValueError(f'incomplete encoded overlay: {verified}/{frames}')
    for error in errors:
        error.update(video_id=vid, visualization=f'images/{vid}_{error["frame_id"]:06d}.jpg',
                     overlay_video=f'overlays/{vid}.mp4', timestamp_sec=rows[error['frame_id']]['timestamp_sec'])
    return dict(video_id=vid, path=str(video_path.relative_to(dest)), frames=frames,
                decoded_verification_frames=verified, complete=True, sha256=sha(video_path))


def source_identity(root, manifest, dest):
    result = dict(manifest_sha256=sha(root / 'manifest.json'), gt_sha256=sha(root / 'gt_public.jsonl'), videos=[])
    for video in manifest['videos']:
        source = root / video['path']
        scan = scan_video(source, video)
        result['videos'].append(dict(video_id=video['video_id'], source=str(source.resolve()), sha256=sha(source), **scan))
    save(dest / 'source_identity.json', result)
    return {v['video_id']: v for v in result['videos']}


def code_identity():
    files = [Path(__file__).resolve(), *sorted((SRC / 'events').glob('*.py')), *sorted((SRC / 'models').glob('*.py')),
             *sorted((SRC / 'tracking').glob('*.py')), *sorted((SRC / 'scene').glob('*.py')),
             SRC / 'data_tools/geometry.py', SRC / 'data_tools/frame_store.py', SRC / 'training/dataset.py']
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, capture_output=True, text=True)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=PROJECT, capture_output=True, text=True)
    diff = subprocess.run(['git', 'diff', '--binary', 'HEAD'], cwd=PROJECT, capture_output=True)
    return dict(git_commit=revision.stdout.strip() if revision.returncode == 0 else None,
                git_status=status.stdout, tracked_diff_sha256=hashlib.sha256(diff.stdout).hexdigest(),
                source_sha256={str(p.relative_to(PROJECT)): sha(p) for p in files})


def cache_inputs(args, system):
    if args.cache is None:
        return None
    cfg = system.config
    root = args.corpus or Path(cfg['dataset'])
    corpus = EventCorpus(root)
    from ttnet.events.continuous import attach_pp
    from ttnet.events.public_validation import attach_public_validation
    attach_pp(corpus, args.pp_annotations or cfg['pp_annotations'])
    attach_public_validation(corpus, args.public, cfg)
    mapping = dict(corpus.public_id_mapping)
    if args.cache_mapping:
        mapping = json.loads(args.cache_mapping.read_text())
    selected = set(mapping.values())
    for part in ('train', 'val', 'test'):
        corpus.split[part] = [vid for vid in corpus.split[part] if vid in selected]
    cache_ident = json.loads((args.cache / 'identity.json').read_text())
    expected = cache_identity(system, corpus)
    if cache_ident != expected:
        changed = sorted(key for key in cache_ident.keys() | expected.keys()
                         if cache_ident.get(key) != expected.get(key))
        raise ValueError('public feature cache inference identity differs from the loaded checkpoint/corpus: ' + ', '.join(changed))
    cache = load_cache(corpus, args.cache, parts=('train', 'val', 'test'), expected=expected)
    if set(cache) != selected:
        raise ValueError('cache does not cover all mapped public videos')
    return corpus, cache, mapping, cache_ident


def error_page(dest, errors, videos):
    cards = ''.join(f'<figure><a href="{html.escape(e["visualization"])}"><img loading="lazy" width="480" src="{html.escape(e["visualization"])}"></a>'
                    f'<figcaption>{html.escape(e["video_id"])}:{e["frame_id"]} {html.escape(e["kind"])} '
                    f'<a href="{html.escape(e["overlay_video"])}#t={e["timestamp_sec"]:.3f}">video</a></figcaption></figure>' for e in errors)
    links = ''.join(f'<li><a href="{html.escape(v["path"])}">{html.escape(v["video_id"])}: {v["frames"]} frames</a></li>' for v in videos)
    (dest / 'errors.html').write_text('<!doctype html><meta charset="utf-8"><title>PP public validation</title>'
        '<style>body{background:#111;color:#eee;font:16px sans-serif}a{color:#8cf}main{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0}</style>'
        '<h1>PP public validation</h1><p>All scored FP/FN and bounce events without coordinates. GREEN: valid GT; YELLOW: ignored GT; RED: prediction; CYAN: current trajectory.</p>'
        '<h2>Complete overlay videos</h2><ul>' + links + '</ul><h2>Errors</h2><main>' + cards + '</main>')


def evaluate(args):
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    started = timestamp()
    state_info = dict(status='evaluating', started=started, checkpoint=str(args.checkpoint.resolve()),
                      validation_set='pp_public', validation_role='validation; not independent external test')
    save(dest / 'state.json', state_info)
    try:
        checkpoint_hash = sha(args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if sha(args.checkpoint) != checkpoint_hash:
            raise ValueError('checkpoint changed while being loaded')
        cfg = deepcopy(checkpoint['config'])
        thresholds = thresholds_from(checkpoint, args.thresholds)
        nms_sec = args.nms_sec if args.nms_sec is not None else float(cfg.get('nms_sec', .08))
        if not np.isfinite(nms_sec) or nms_sec < 0:
            raise ValueError('checkpoint NMS duration must be finite and nonnegative')
        public_hashes = {name:sha(args.public / name) for name in ('manifest.json', 'gt_public.jsonl')}
        manifest, gt, counts = public_metadata(args.public)
        if any(sha(args.public / name) != digest for name, digest in public_hashes.items()):
            raise ValueError('public manifest/GT changed during loading')
        sources = source_identity(args.public, manifest, dest)
        save(dest / 'config.yaml', cfg)
        ident = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                     checkpoint_epoch=checkpoint.get('epoch'), checkpoint_step=checkpoint.get('step'),
                     config_sha256=identity(checkpoint['config']), thresholds=thresholds,
                     threshold_source='explicit command line override' if args.thresholds is not None else 'checkpoint thresholds',
                     threshold_selection='No threshold search in this evaluation; pp_public is the final validation set.',
                     nms_sec=nms_sec, nms_source='command line override' if args.nms_sec is not None else 'checkpoint config (fallback .08s)',
                     public=str(args.public.resolve()), public_counts=counts,
                     command=[sys.executable, *sys.argv], started=started, code=code_identity())
        save(dest / 'identity.json', ident)
        torch.set_num_threads(cfg.get('torch_threads', 2))
        cv2.setNumThreads(1)
        init_started = time.perf_counter()
        system = EventSystem(cfg, torch.device(args.device), initialize=False)
        system.load_state_dict(checkpoint['system'])
        system.configure(cfg['mode'], False)
        if system.device.type == 'cuda':
            torch.cuda.synchronize()
        initialization_sec = time.perf_counter() - init_started
        caches = cache_inputs(args, system)
        if caches:
            save(dest / 'cache_identity.json', dict(path=str(args.cache.resolve()), identity=caches[3], mapping=caches[2],
                 identity_sha256=sha(args.cache / 'identity.json'), note='Cache identity and numerical/decision parity are reported separately.'))
        all_predictions, all_errors, all_events, per_video, performance, overlay_info = [], [], [], {}, [], []
        cache_comparison, cached_scores = {}, {}
        wall_started = time.perf_counter()
        for video in manifest['videos']:
            vid = video['video_id']
            print(f'PP public native-frame validation: {vid}', flush=True)
            # GT is not passed into predict_video, nor attached to its StreamingCorpus.
            rows, records, speed = predict_video(system, args.public / video['path'], vid, target_fps=None)
            validate_rows(rows, records, video, sources[vid]['relative_timestamps_sec'])
            for row in rows:
                row['video_id'] = vid
            speed['video_id'] = vid
            performance.append(speed)
            save_jsonl(dest / 'frames' / f'{vid}.jsonl', rows)
            save_jsonl(dest / 'trajectory' / f'{vid}.jsonl', trajectory_records(records[fid] for fid in sorted(records)))
            truth = [g for g in gt if g['video_id'] == vid]
            scored, predictions, missing, events, errors = score_video(rows, truth, thresholds, nms_sec)
            per_video[vid] = scored
            all_predictions.extend(predictions)
            all_events.extend(dict(r, video_id=vid) for r in events)
            save_jsonl(dest / 'events_without_coordinates' / f'{vid}.jsonl', missing)
            overlay_info.append(overlays(args.public / video['path'], video, rows, records, truth,
                                         predictions, scored, errors, dest, thresholds))
            all_errors.extend(errors)
            if caches:
                corpus, cache, mapping, _ = caches
                cache_vid = mapping[vid]
                features, cache_records = cache[cache_vid]
                if sorted(features) != list(range(video['n_frames'])):
                    raise ValueError('cache lacks complete public frame features')
                cache_rows = predict_windows(system, corpus, cache_vid, features, cache_records)
                validate_rows(cache_rows, cache_records, video, sources[vid]['relative_timestamps_sec'])
                for row in cache_rows:
                    row['video_id'] = vid
                save_jsonl(dest / 'cache_frames' / f'{vid}.jsonl', cache_rows)
                parity, differences = compare_frames(cache_rows, rows, thresholds, nms_sec,
                                                       args.probability_atol, args.coordinate_atol)
                save_jsonl(dest / 'cache_differences' / f'{vid}.jsonl', differences)
                cached_scores[vid] = score_video(cache_rows, truth, thresholds, nms_sec)[0]
                parity['landing_decisions_equal'] = landing_decisions(cached_scores[vid]) == landing_decisions(scored)
                parity['passed'] = parity['passed'] and parity['landing_decisions_equal']
                cache_comparison[vid] = parity
            save(dest / 'progress.json', dict(status='evaluating', finished_videos=list(per_video), updated=timestamp()))
        if sum(p['frames'] for p in performance) != counts['frames']:
            raise ValueError('full public inference frame total differs')
        if sha(args.checkpoint) != checkpoint_hash:
            raise ValueError('checkpoint changed during evaluation')
        if any(sha(args.public / v['path']) != sources[v['video_id']]['sha256'] for v in manifest['videos']):
            raise ValueError('source video changed during evaluation')
        if any(sha(args.public / name) != digest for name, digest in public_hashes.items()):
            raise ValueError('public manifest/GT changed during evaluation')
        metric = summarize(per_video, manifest)
        metric.update(status='evaluated', validation_set='pp_public',
                      validation_role='Final validation set; checkpoint/hyperparameter selection may use this set. Not independent external test.',
                      protocol='Native source frames, complete video inference; bounce peaks; +/-1 frame and <=50px; maximum-cardinality one-to-one valid-GT-first matching; unmatched predictions near ignored GT ignored.',
                      thresholds=thresholds, threshold_source=ident['threshold_source'], nms_sec=nms_sec,
                      checkpoint_sha256=checkpoint_hash, public_counts=counts,
                      ground_truth_input_to_inference=False,
                      e2e_fps=counts['frames'] / sum(p['elapsed_sec'] for p in performance),
                      evaluated_at=timestamp(), artifacts=dict(errors='errors.jsonl', error_gallery='errors.html', predictions='predictions.jsonl',
                      frames='frames/', trajectory='trajectory/', full_overlay_videos='overlays/', source_identity='source_identity.json'))
        if caches:
            metric['cache_online_parity'] = dict(passed=all(p['passed'] for p in cache_comparison.values()),
                 exact=all(p['exact'] for p in cache_comparison.values()),
                 decisions_equal=all(p['decisions_equal'] for p in cache_comparison.values()),
                 landing_decisions_equal=all(p['landing_decisions_equal'] for p in cache_comparison.values()),
                 scope='Every frame of all 15 videos; threshold decisions and post-NMS peak frames included; cache metrics use the same validation labels.',
                 online_metrics=summarize(per_video, manifest), cache_metrics=summarize(cached_scores, manifest))
            save(dest / 'cache_parity.json', dict(**metric['cache_online_parity'], per_video=cache_comparison))
            save(dest / 'cache_per_video.json', cached_scores)
        save(dest / 'metrics.json', metric)
        save(dest / 'per_video.json', per_video)
        save(dest / 'performance.json', dict(videos=performance, initialization_sec=initialization_sec,
             full_evaluation_wall_sec=time.perf_counter() - wall_started,
             inference_e2e_fps=metric['e2e_fps'], hardware=torch.cuda.get_device_name(system.device) if system.device.type == 'cuda' else platform.processor(),
             device=str(system.device), backend='PyTorch', torch_version=torch.__version__, cuda_version=torch.version.cuda,
             parameters=sum(p.numel() for p in system.parameters()) if hasattr(system, 'parameters') else None,
             macs=None, flops=None, compute_note='Compute profiling is left to the separate checkpoint benchmark.',
             python=sys.version, opencv_version=cv2.__version__, av_version=av.__version__,
             precision='AMP' if cfg.get('amp') and system.device.type == 'cuda' else 'FP32',
             speed_note='Observed complete native-video latency from predict_video, including CPU decode and JSON serialization; excludes model initialization, disk I/O, scoring and overlays. No exclusive-GPU or deployment-hardware claim.',
             forward_latency_ms=None, forward_latency_note='No separate component benchmark performed.'))
        save(dest / 'overlay_verification.json', overlay_info)
        save_jsonl(dest / 'predictions.jsonl', all_predictions)
        save_jsonl(dest / 'events.jsonl', all_events)
        save_jsonl(dest / 'errors.jsonl', all_errors)
        save(dest / 'error_index.json', dict(coverage='All scored FP/FN and all bounce peaks without coordinates; every source frame also in complete overlay videos.',
             counts=dict(Counter(e['kind'] for e in all_errors)), errors=all_errors))
        error_page(dest, all_errors, overlay_info)
        lines = ['# PP public 验证', '', '状态：`evaluated`。这是最终验证集成绩；可用于选择 checkpoint 与超参数，不作为独立外测结论。', '',
                 f'Checkpoint: `{args.checkpoint.resolve()}`；SHA256 `{checkpoint_hash}`；epoch {checkpoint.get("epoch")}。',
                 f'阈值：{thresholds}（{ident["threshold_source"]}）；NMS {nms_sec}s。',
                 f'完整原生帧推理：{counts["videos"]} 视频 / {counts["frames"]} 帧；{counts["valid_gt"]} valid GT / {counts["ignored_gt"]} ignored GT。推理不输入 GT。', '',
                 '| 视角 | P | R | F1 | TP | FP | FN |', '|---|---:|---:|---:|---:|---:|---:|']
        for angle, item in metric['by_angle'].items():
            lines.append(f'| {angle} | {item["precision"]:.4f} | {item["recall"]:.4f} | {item["f1"]:.4f} | {item["tp"]} | {item["fp"]} | {item["fn"]} |')
        total = metric['overall']
        lines += ['', f'宏 F1：{metric["macro_f1"]:.6f}；总 TP/FP/FN：{total["tp"]}/{total["fp"]}/{total["fn"]}。',
                  f'无坐标 bounce peaks：{metric["events_without_coordinates"]}。', '',
                  '[全部错误与完整叠加视频](errors.html) · [指标](metrics.json) · [逐视频](per_video.json) · [预测](predictions.jsonl) · [性能](performance.json) · [身份与代码哈希](identity.json)']
        if caches:
            lines += ['', f'全帧 cache/online parity: `{metric["cache_online_parity"]["passed"]}`；[详情](cache_parity.json)。']
        (dest / 'summary.md').write_text('\n'.join(lines) + '\n')
        save(dest / 'progress.json', dict(status='evaluated', finished_videos=list(per_video), updated=timestamp()))
        save(dest / 'state.json', dict(state_info, status='evaluated', finished=timestamp(), macro_f1=metric['macro_f1'], checkpoint_sha256=checkpoint_hash))
        print(json.dumps(dict(output=str(dest), macro_f1=metric['macro_f1'], overall=total, status='evaluated'), ensure_ascii=False), flush=True)
        return metric
    except BaseException as exc:
        save(dest / 'state.json', dict(state_info, status='failed', finished=timestamp(), error=repr(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing evaluations are never overwritten')
    parser.add_argument('--public', type=Path, default=PROJECT / 'pingpang/public_data')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--thresholds', nargs=2, type=float, metavar=('BOUNCE', 'NET'))
    parser.add_argument('--nms-sec', type=float, help='Defaults to checkpoint config.nms_sec, falling back to .08')
    parser.add_argument('--cache', type=Path, help='Optional public feature cache for every-frame cache/online parity')
    parser.add_argument('--corpus', type=Path, help='Base EventCorpus root, defaults to checkpoint config.dataset')
    parser.add_argument('--pp-annotations', type=Path)
    parser.add_argument('--cache-mapping', type=Path, help='Optional JSON {public_video_id: cache_video_id}')
    parser.add_argument('--probability-atol', type=float, default=2e-4)
    parser.add_argument('--coordinate-atol', type=float, default=.1)
    args = parser.parse_args()
    if (args.nms_sec is not None and args.nms_sec < 0) or args.probability_atol < 0 or args.coordinate_atol < 0:
        parser.error('NMS and numerical tolerances must be nonnegative')
    evaluate(args)


if __name__ == '__main__':
    main()
