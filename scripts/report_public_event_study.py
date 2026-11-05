"""CPU-only integrity audit and report for the nine PP154/public15 event runs.

No training, inference, threshold search, task-state mutation or model-run writes
are performed. All nine completed runs are required before any report is written.
Overlay hashes and complete-decoding receipts are checked; videos are not decoded
again. Generated reports and additive completion receipts belong to this script.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from ttnet.common import identity, jsonl, save, sha
from ttnet.configuration import load_config
from ttnet.events.continuous import attach_pp, mixed_samples
from ttnet.events.data import EventCorpus, sample_identity
from ttnet.events.engine import cache_identity
from ttnet.events.public_validation import attach_public_validation
from ttnet.models.build import build_event_head
from scripts.evaluate_public_event import (
    PUBLIC_EXPECTED, compare_frames, landing_decisions, score_video, summarize,
    validate_rows,
)
from scripts.report_b_event_study import PROTOCOL_KEYS, frozen_weights, normalize
from scripts.run_b_event_replication import validate_table_cache

ARMS = ('F0', 'concat', 'gated')
SEEDS = (42, 43, 44)
DEFAULT_TASK = PROJECT/'out/experiments/20261105-092319_public_event_study'


class ReportError(ValueError):
    pass


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT/path).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


class Audit:
    def __init__(self):
        self.checks, self.sources = [], {}

    def require(self, name, passed, **details):
        self.checks.append(dict(name=name, passed=bool(passed), **details))
        if not passed:
            raise ReportError(name + ': ' + json.dumps(details, ensure_ascii=False))

    def source(self, path):
        path = resolve(path)
        if not path.is_file():
            raise ReportError('missing artifact: '+str(path))
        key = str(path)
        if key not in self.sources:
            self.sources[key] = sha(path)
        return self.sources[key]

    digest = source  # The established table-cache validator uses this name.

    def json(self, path):
        self.source(path)
        return read(resolve(path))

    def unchanged(self):
        changed = [p for p, digest in self.sources.items() if sha(p) != digest]
        self.require('all_sources_unchanged_during_report', not changed, changed=changed)


def close_number(left, right):
    return (isinstance(left, (int, float)) and isinstance(right, (int, float))
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12))


def tensor_digest(state, prefix=''):
    digest = hashlib.sha256()
    tensors = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    for key, tensor in sorted(tensors.items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def official_cache_identity(config):
    corpus = SimpleNamespace(identity=config['dataset_id'])
    for name, backend in config.get('pixel_backends', {}).items():
        if name in ('frame_store', 'pp_frame_store'):
            setattr(corpus, name, SimpleNamespace(
                is_video=True, backend_identity=lambda value=backend:value))
    return cache_identity(SimpleNamespace(config=config), corpus)


def collect_epochs(run, corpus, config, audit):
    """Reconstruct exact scheduled sample identities, not just counts or seeds."""
    epochs, duplicate_receipts = {}, []
    for path in sorted(run.glob('attempts/*/epochs.jsonl')):
        audit.source(path)
        for row in jsonl(path):
            epoch = row['epoch']
            if epoch in epochs:
                keys = ('schedule_sha256', 'thresholds', 'continuous', 'sampling_counts')
                audit.require(run.name+':duplicate_epoch_'+str(epoch),
                              all(epochs[epoch].get(k) == row.get(k) for k in keys))
                duplicate_receipts.append(epoch)
            epochs[epoch] = row
    audit.require(run.name+':contiguous_epochs', bool(epochs) and
                  sorted(epochs) == list(range(1, max(epochs, default=0)+1)))
    audit.require(run.name+':no_unrecorded_hard_negative_sampling',
                  not config.get('pp_hard_negative_manifest'))
    schedules = []
    for epoch, row in sorted(epochs.items()):
        schedule = mixed_samples(corpus.samples['train'], config['seed'], epoch-1, config)
        pp = [s for s in schedule if s.get('source') == 'pp']
        expected = dict(total=len(schedule), pp=len(pp),
                        pp_positive=sum(s['target'][0] > 0 for s in pp), pp_hard_negative=0)
        digest = sample_identity(schedule)
        audit.require(run.name+':epoch_'+str(epoch)+'_sampling',
                      row.get('schedule_sha256') == digest and row.get('sampling_counts') == expected,
                      epoch=epoch, expected_sha256=digest, expected_counts=expected)
        audit.require(run.name+':epoch_'+str(epoch)+'_public_selection',
                      row.get('selection_metric') == 'pp_landing_macro_f1'
                      and row.get('continuous') is not None
                      and set(row['continuous']['per_video']) == set(corpus.public_video_ids))
        schedules.append(dict(epoch=epoch, sha256=digest, counts=expected))
    return epochs, dict(epochs=schedules, duplicate_receipts=duplicate_receipts,
                        method='Reconstructed mixed_samples on the exact current corpus; no optimizer work.')


def validate_cache(spec, corpus, audit):
    receipt_path = resolve(spec['receipt'])
    receipt = audit.json(receipt_path)
    dest, source = resolve(receipt['cache']), resolve(spec['source'])
    ident = audit.json(dest/'identity.json')
    derivation_path = resolve(receipt['derivation'])
    derivation = audit.json(derivation_path)
    source_ident = audit.json(source/'identity.json')
    audit.require('cache:'+str(dest)+':receipt', receipt.get('status') == 'completed'
                  and receipt['identity_sha256'] == audit.source(dest/'identity.json')
                  and receipt['derivation_sha256'] == audit.source(derivation_path)
                  and derivation_path == dest/'public_split_derivation.json')
    audit.require('cache:'+dest.name+':derivation',
                  derivation.get('status') == 'completed'
                  and resolve(derivation['source']) == source
                  and derivation['source_identity'] == source_ident
                  and derivation['source_identity_sha256'] == audit.source(source/'identity.json')
                  and derivation['target_identity'] == ident
                  and derivation['split_audit'] == corpus.public_split_audit
                  and derivation['implementation_sha256'] == audit.source(PROJECT/'scripts/run_public_event_study.py')
                  and dest.name == identity(ident)[:20])
    before = deepcopy(ident)
    before['dataset_id'] = corpus.public_split_audit['original_dataset_id']
    audit.require('cache:'+dest.name+':only_split_identity_changed', before == source_ident)
    expected = {vid for part in ('train', 'val') for vid in corpus.split[part]}
    public = set(corpus.public_video_ids)
    audit.require('cache:'+dest.name+':all_video_receipts',
                  set(derivation['reused']) == expected-public
                  and set(derivation['added']) == public
                  and len(expected-public) == 159 and len(public) == 15)
    for category in ('reused', 'added'):
        for vid, evidence in derivation[category].items():
            ids = sorted(corpus.videos[vid]['frames'])
            with np.load(dest/(vid+'.npz'), allow_pickle=False) as arrays:
                audit.require('cache:'+dest.name+':'+vid+':array_frame_ids',
                              arrays['frame_ids'].tolist() == ids)
                if ident.get('table_visual'):
                    audit.require('cache:'+dest.name+':'+vid+':table_arrays',
                                  {'table_context', 'table_roi'} <= set(arrays.files))
            records = jsonl(dest/(vid+'.jsonl'))
            audit.require('cache:'+dest.name+':'+vid+':record_frame_ids',
                          [r['frame_id'] for r in records] == ids and evidence['frames'] == len(ids))
            for suffix in ('.npz', '.jsonl'):
                digest = audit.source(dest/(vid+suffix))
                audit.require('cache:'+dest.name+':'+vid+suffix+':hash',
                              digest == evidence['sha256'][suffix])
                if category == 'reused':
                    audit.require('cache:'+dest.name+':'+vid+suffix+':source_copy',
                                  digest == audit.source(source/(vid+suffix)))
    table_proof = validate_table_cache(source, audit) if ident.get('table_visual') else None
    return dict(path=str(dest), identity=ident, receipt=str(receipt_path),
                identity_sha256=audit.source(dest/'identity.json'),
                derivation=str(derivation_path), derivation_sha256=audit.source(derivation_path),
                reused_videos=159, added_public_videos=15, source_table_validation=table_proof)


def completed_step(state, name, audit):
    step = state.get('steps', {}).get(name, {})
    audit.require('task_step:'+name, step.get('status') == 'completed'
                  and step.get('exit_code', 0) == 0)
    if step.get('log'):
        audit.source(step['log'])
        receipt = resolve(step['log']).with_suffix('.json')
        audit.require('task_step_receipt:'+name, audit.json(receipt) == step)
    return step


def score_signature(metric):
    """Cross-format landing summary: training uses landing, evaluator overall."""
    overall = metric.get('overall', metric.get('landing'))
    angles = metric.get('by_angle', metric.get('by_view'))
    return dict(overall={k:overall[k] for k in ('tp', 'fp', 'fn')},
                by_angle={angle:{k:s[k] for k in ('tp', 'fp', 'fn')}
                          for angle,s in angles.items()}, macro_f1=metric['macro_f1'])


def require_score_equal(audit, name, left, right):
    a, b = score_signature(left), score_signature(right)
    audit.require(name, a['overall'] == b['overall'] and a['by_angle'] == b['by_angle']
                  and close_number(a['macro_f1'], b['macro_f1']), left=a, right=b)


def validate_evaluation(evaluation, cfg, best, checkpoint_sha, selection, train_rows,
                        cache, corpus, manifest, ground_truth, audit, label):
    result = audit.json(evaluation/'metrics.json')
    ident = audit.json(evaluation/'identity.json')
    performance = audit.json(evaluation/'performance.json')
    source = audit.json(evaluation/'source_identity.json')
    overlays = audit.json(evaluation/'overlay_verification.json')
    parity_receipt = audit.json(evaluation/'cache_parity.json')
    per_video = audit.json(evaluation/'per_video.json')
    cached_video = audit.json(evaluation/'cache_per_video.json')
    state = audit.json(evaluation/'state.json')
    eval_config = audit.json(evaluation/'config.yaml')
    cache_proof = audit.json(evaluation/'cache_identity.json')
    thresholds = best['thresholds']
    audit.require(label+':evaluation_identity', result.get('status') == state.get('status') == 'evaluated'
                  and result.get('validation_set') == 'pp_public'
                  and result.get('public_counts') == ident.get('public_counts') == PUBLIC_EXPECTED
                  and result.get('ground_truth_input_to_inference') is False
                  and result.get('checkpoint_sha256') == ident.get('checkpoint_sha256') == checkpoint_sha
                  and ident.get('checkpoint_epoch') == best['epoch'] and ident.get('checkpoint_step') == 0
                  and ident.get('config_sha256') == identity(cfg) and eval_config == cfg
                  and result.get('thresholds') == ident.get('thresholds') == thresholds
                  and ident.get('threshold_source') == 'checkpoint thresholds'
                  and result.get('nms_sec') == cfg['nms_sec'] == ident.get('nms_sec'))
    audit.require(label+':evaluation_cache_identity', resolve(cache_proof['path']) == Path(cache['path'])
                  and cache_proof['identity'] == cache['identity']
                  and cache_proof['identity_sha256'] == cache['identity_sha256']
                  and cache_proof['mapping'] == corpus.public_id_mapping)
    # Fixed historical references have no public_validation_root in their old config.
    public_root = resolve(cfg.get('public_validation_root', ident['public']))
    audit.require(label+':evaluation_public_sources',
                  source.get('manifest_sha256') == audit.source(public_root/'manifest.json')
                  and source.get('gt_sha256') == audit.source(public_root/'gt_public.jsonl'))
    for path, digest in ident.get('code', {}).get('source_sha256', {}).items():
        audit.require(label+':evaluation_code:'+path, audit.source(path) == digest)
    audit.require(label+':evaluation_code_snapshot', bool(ident.get('code', {}).get('source_sha256')))
    source_by_id = {v['video_id']:v for v in source['videos']}
    overlay_by_id = {v['video_id']:v for v in overlays}
    expected_ids = {v['video_id'] for v in manifest['videos']}
    audit.require(label+':all_public_videos', set(per_video) == set(cached_video)
                  == set(source_by_id) == set(overlay_by_id) == expected_ids
                  and len(overlays) == len(source['videos']) == 15
                  and (train_rows is None or set(train_rows) == set(corpus.public_video_ids)))
    expected_errors, all_predictions = [], []
    recomputed_online, recomputed_cache, recomputed_training = {}, {}, {}
    max_probability, max_coordinate, decoded = 0., 0., 0
    matches, all_gt = set(), {}
    for video in manifest['videos']:
        vid = video['video_id']; prefixed = corpus.public_id_mapping[vid]
        src = source_by_id[vid]; overlay = overlay_by_id[vid]
        truth = [g for g in ground_truth if g['video_id'] == vid]
        valid_gt = [g for g in truth if g['valid']]
        audit.require(label+':source:'+vid, src.get('decode_complete') is True
                      and src['frames'] == video['n_frames']
                      and (src['width'], src['height']) == (video['width'], video['height'])
                      and src['sha256'] == audit.source(public_root/video['path'])
                      and resolve(src['source']) == (public_root/video['path']).resolve())
        audit.require(label+':overlay:'+vid, overlay.get('complete') is True
                      and overlay['frames'] == overlay['decoded_verification_frames'] == video['n_frames']
                      and audit.source(evaluation/overlay['path']) == overlay['sha256'])
        decoded += overlay['decoded_verification_frames']
        rows_path, cache_path = evaluation/'frames'/f'{vid}.jsonl', evaluation/'cache_frames'/f'{vid}.jsonl'
        audit.source(rows_path); audit.source(cache_path)
        rows, cached = jsonl(rows_path), jsonl(cache_path)
        empty_records = {fid:{} for fid in range(video['n_frames'])}
        validate_rows(rows, empty_records, video, src['relative_timestamps_sec'])
        validate_rows(cached, empty_records, video, src['relative_timestamps_sec'])
        if train_rows is not None:
            validate_rows(train_rows[prefixed], empty_records, video, src['relative_timestamps_sec'])
        audit.source(evaluation/'trajectory'/f'{vid}.jsonl')
        audit.source(evaluation/'events_without_coordinates'/f'{vid}.jsonl')
        audit.source(evaluation/'cache_differences'/f'{vid}.jsonl')
        parity, _ = compare_frames(cached, rows, thresholds, cfg['nms_sec'])
        audit.require(label+':recomputed_frame_parity:'+vid, parity['passed'])
        recorded = parity_receipt['per_video'][vid]
        audit.require(label+':recorded_frame_parity:'+vid,
                      recorded.get('passed') is True and recorded.get('landing_decisions_equal') is True
                      and all(recorded.get(k) == json.loads(json.dumps(v)) for k,v in parity.items()))
        max_probability = max(max_probability, parity['probability_max_abs'])
        max_coordinate = max(max_coordinate, parity['coordinate_max_abs_pixels'])
        scored, predictions, _, _, errors = score_video(rows, truth, thresholds, cfg['nms_sec'])
        cached_score = score_video(cached, truth, thresholds, cfg['nms_sec'])[0]
        audit.require(label+':recomputed_landing_decisions:'+vid,
                      landing_decisions(scored) == landing_decisions(per_video[vid])
                      == landing_decisions(cached_score) == landing_decisions(cached_video[vid]))
        if train_rows is not None:
            trained_score = score_video(train_rows[prefixed], truth, thresholds, cfg['nms_sec'])[0]
            train_score = selection['continuous']['per_video'][prefixed]['landing']
            audit.require(label+':selected_training_decisions:'+vid,
                          landing_decisions(scored) == landing_decisions(trained_score)
                          and all(train_score[k] == scored[k] for k in
                              ('tp', 'fp', 'fn', 'ignored', 'matches', 'false_positives', 'false_negatives')))
            recomputed_training[vid] = trained_score
        recomputed_online[vid], recomputed_cache[vid] = scored, cached_score
        all_predictions.extend(predictions)
        expected_errors.extend((vid, e['frame_id'], e['kind']) for e in errors)
        for i, gt in enumerate(valid_gt):
            key = (vid, gt['frame_id'])
            audit.require(label+':unique_gt:'+str(key), key not in all_gt)
            all_gt[key] = dict(video_id=vid, frame_id=gt['frame_id'], x=gt['x'], y=gt['y'], angle=video['angle'])
            if i not in scored['false_negatives']:
                matches.add(key)
    audit.source(evaluation/'predictions.jsonl')
    audit.require(label+':saved_predictions', jsonl(evaluation/'predictions.jsonl') == all_predictions)
    actual_errors = jsonl(evaluation/'errors.jsonl'); audit.source(evaluation/'errors.jsonl')
    index = audit.json(evaluation/'error_index.json')
    audit.require(label+':complete_error_coverage',
                  Counter(expected_errors) == Counter((e['video_id'], e['frame_id'], e['kind']) for e in actual_errors)
                  and index['errors'] == actual_errors
                  and index['counts'] == dict(Counter(e['kind'] for e in actual_errors)))
    for error in actual_errors:
        audit.source(evaluation/error['visualization'])
        audit.source(evaluation/error['overlay_video'])
    audit.source(evaluation/'errors.html')
    online, cached = (summarize(values, manifest) for values in (recomputed_online, recomputed_cache))
    comparisons = [('recorded_online', result), ('recorded_parity_online', parity_receipt['online_metrics']),
                         ('recorded_parity_cache', parity_receipt['cache_metrics']),
                         ('recomputed_cached', cached)]
    if train_rows is not None:
        comparisons += [('recomputed_training', summarize(recomputed_training, manifest)),
                        ('best_epoch_selection', selection['continuous'])]
    for name, metric in comparisons:
        require_score_equal(audit, label+':score_equal:'+name, online, metric)
    audit.require(label+':summary_parity', result['cache_online_parity']['passed'] is True
                  and parity_receipt['passed'] is True and parity_receipt['decisions_equal'] is True
                  and parity_receipt['landing_decisions_equal'] is True)
    frames = sum(p['frames'] for p in performance['videos'])
    elapsed = sum(p['elapsed_sec'] for p in performance['videos'])
    fps = frames/elapsed
    audit.require(label+':complete_speed', frames == decoded == PUBLIC_EXPECTED['frames']
                  and len(performance['videos']) == 15
                  and {p['video_id'] for p in performance['videos']} == expected_ids
                  and close_number(fps, result['e2e_fps'])
                  and close_number(fps, performance['inference_e2e_fps'])
                  and all(p.get('native_frame_ids') is True for p in performance['videos']))
    return dict(metrics=result, performance=performance, per_video=per_video,
                matched_events=matches, ground_truth=all_gt,
                overlays=dict(videos=15, decoded_frames=decoded,
                    verification='Hash-verified complete-frame decoder receipts; this CPU report does not re-decode videos.'),
                errors=dict(counts=index['counts'], indexed_errors=len(actual_errors), gallery=str(evaluation/'errors.html')),
                parity=dict(passed=True, exact=parity_receipt['exact'],
                    probability_max_abs=max_probability, coordinate_max_abs_pixels=max_coordinate))


def optional_references(task, caches, corpus, manifest, truth, audit):
    """Absent historical evaluations never block the nine-run study itself."""
    path = task/'historical_reference_plan.json'
    if not path.exists():
        return dict(status='not_planned', completed={}, pending=[])
    plan = audit.json(path); results, pending = {}, []
    prior_path = PROJECT/'experiments/records/20261105_B_event_closeout/evaluations/audit.json'
    prior = audit.json(prior_path) if prior_path.exists() else None
    for spec in plan['experiments']:
        output = resolve(spec['output']); label = spec['name']
        if not (output/'metrics.json').is_file() or not (output/'state.json').is_file() or read(output/'state.json').get('status') != 'evaluated':
            pending.append(dict(name=label, output=str(output)))
            continue
        checkpoint_path = resolve(spec['checkpoint']); digest = audit.source(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        selection = audit.json(checkpoint_path.parents[1]/'best_selection.json')
        audit.require(label+':old_internal_selection', checkpoint['epoch'] == selection['epoch']
                      and checkpoint['thresholds'] == selection['thresholds'])
        if prior is not None:
            audit.require(label+':previously_locked_checkpoint',
                          prior['validation']['sources_sha256'].get(str(checkpoint_path)) == digest)
        matches = [c for c in caches.values() if resolve(c['receipt']) == resolve(spec['cache_receipt'])]
        audit.require(label+':reference_cache_receipt', len(matches) == 1)
        evaluated = validate_evaluation(output, checkpoint['config'], checkpoint, digest, None,
            None, matches[0], corpus, manifest, truth, audit, label)
        metric = evaluated['metrics']
        results[label] = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=digest,
            epoch=checkpoint['epoch'], thresholds=checkpoint['thresholds'],
            macro_f1=metric['macro_f1'], overall=metric['overall'], by_angle=metric['by_angle'],
            e2e_fps=metric['e2e_fps'], parity=evaluated['parity'], overlays=evaluated['overlays'],
            errors=evaluated['errors'], evaluation=str(output))
    return dict(status='completed' if not pending else 'partial_or_pending', completed=results,
        pending=pending, plan=str(path), selection=plan.get('selection'),
        interpretation='Old seed42 checkpoints and thresholds were selected on the former internal validation. New runs select epochs and thresholds on pp_public. This is a practical fixed reference, not a pure training-data ablation.')


def locator_diagnostics(task, caches, root, truth, audit):
    path = task/'public_locator_diagnostics.json'
    if not path.is_file():
        return dict(status='unavailable')
    diagnostic = audit.json(path); audit.source(task/'public_locator_diagnostics.md')
    cache = caches['ball']
    audit.require('locator_diagnostics:identity', resolve(diagnostic['cache']) == Path(cache['path'])
                  and diagnostic['cache_identity_sha256'] == cache['identity_sha256']
                  and diagnostic['cache_identity'] == cache['identity']
                  and diagnostic['manifest_sha256'] == audit.source(root/'manifest.json')
                  and diagnostic['gt_sha256'] == audit.source(root/'gt_public.jsonl'))
    expected = {(g['video_id'],g['frame_id'],g['x'],g['y']) for g in truth if g['valid']}
    actual = {(e['ground_truth']['video_id'],e['ground_truth']['frame_id'],e['ground_truth']['x'],e['ground_truth']['y']) for e in diagnostic['events']}
    audit.require('locator_diagnostics:GT_population', actual == expected and len(diagnostic['events']) == len(expected) == 92)
    for vid, digest in diagnostic['record_sha256'].items():
        audit.require('locator_diagnostics:record:'+vid,
                      audit.source(Path(cache['path'])/f'public_{vid}.jsonl') == digest)
    stage = diagnostic['overall']['stages']['final_prediction']
    for mode in ('exact', 'within_one_frame'):
        covered = sum(e[mode]['final_prediction']['covered'] for e in diagnostic['events'])
        audit.require('locator_diagnostics:coverage:'+mode,
                      stage[mode]['covered'] == covered and stage[mode]['total'] == 92
                      and close_number(stage[mode]['coverage'], covered/92))
    return dict(status='verified', json=str(path), markdown=str(task/'public_locator_diagnostics.md'),
                final_prediction=stage, overall=diagnostic['overall'], by_angle=diagnostic['by_angle'],
                explanation='GT-assisted coordinate coverage at a fixed landing GT point in the same or +/-1 original frame. It excludes event decisions, FP and one-to-one matching and is neither event F1 nor a strict event-recall ceiling.')


def validate_benchmark(run, cfg, head_parameters, performance, audit):
    path = run/'evaluations/benchmark/benchmark.json'
    result = audit.json(path)
    samples = audit.json(path.with_name('benchmark_samples_ms.json'))
    audit.require(run.name+':benchmark_protocol', result.get('warmup') == 50
                  and result.get('measured') == len(samples) == 600 and result.get('batch') == 1
                  and all(math.isfinite(v) and v > 0 for v in samples)
                  and result['parameters']['head'] == head_parameters
                  and result['amp'] == cfg['amp'] and result['device'].split(':')[0] == 'cuda')
    for percentile, key in [(50, 'head_forward_p50_ms'), (95, 'head_forward_p95_ms')]:
        audit.require(run.name+':benchmark_'+key, close_number(result[key], float(np.percentile(samples, percentile))))
    signature = (result['hardware'], 'PyTorch', 'AMP' if result['amp'] else 'FP32')
    audit.require(run.name+':benchmark_evaluation_backend',
                  signature == (performance['hardware'], performance['backend'], performance['precision']))
    audit.require(run.name+':compute_accounting', set(result['macs']) == set(result['flops'])
                  and all(result['flops'][k] == 2*v for k,v in result['macs'].items()))
    return dict(path=str(path), sha256=audit.source(path), **result,
                backend='PyTorch', precision=signature[2],
                backend_evidence='PyTorch-only head_benchmark producer; original benchmark schema omits backend.')


def validate_run(spec, state, corpus, caches, manifest, truth, audit, originals):
    label, run = spec['name'], resolve(spec['run'])
    for suffix in ('smoke', 'train', 'evaluate', 'benchmark'):
        completed_step(state, label+'_'+suffix, audit)
    smoke_step = state['steps'][label+'_smoke']
    smoke_command = smoke_step.get('command', [])
    smoke_run = (resolve(smoke_step['run']) if smoke_step.get('run') else
                 resolve(smoke_command[smoke_command.index('--run')+1]))
    smoke = audit.json(smoke_run/'smoke.json')
    audit.require(label+':smoke_passed', smoke.get('passed') is True
                  and smoke.get('micro_batch_checked') == 8)
    cfg = audit.json(run/'config.yaml')
    runtime = audit.json(run/'run.json')
    selection = audit.json(run/'best_selection.json')
    wanted, provenance = load_config(resolve(spec['config']))
    for path in provenance['sources']:
        audit.source(path)
    mismatches = [k for k,v in wanted.items() if normalize(cfg.get(k)) != normalize(v)]
    expected_fusion = 'none' if spec['arm'] == 'F0' else spec['arm']
    audit.require(label+':actual_config', not mismatches, mismatched_keys=mismatches)
    audit.require(label+':fixed_recipe', cfg.get('seed') == spec['seed']
                  and cfg.get('mode') == 'frozen' and cfg.get('event_head_type') == 'tcn'
                  and cfg.get('context_head') is True
                  and cfg.get('table_fusion', {}).get('mode') == expected_fusion
                  and cfg.get('split_id') == state['protocol']['split_id']
                  and cfg.get('dataset_id') == corpus.identity
                  and cfg.get('train_samples_sha256') == sample_identity(corpus.samples['train'])
                  and cfg.get('val_samples_sha256') == sample_identity(corpus.samples['val'])
                  and cfg.get('public_validation_root') == str(resolve(wanted['public_validation_root']))
                  and cfg.get('smoke') is False
                  and all(cfg.get(k) == .7 for k in ('global_threshold', 'local_threshold', 'localization_threshold')))
    audit.require(label+':split_audit', audit.json(run/'public_split_audit.json') == corpus.public_split_audit)
    audit.require(label+':saved_record_config', audit.json(PROJECT/'experiments/records'/run.name/'config.yaml') == cfg)
    cache = caches[spec['cache']]
    audit.require(label+':official_training_cache_identity', official_cache_identity(cfg) == cache['identity'])
    receipts = list(run.glob('attempts/*/feature_cache.json'))
    audit.require(label+':training_cache_receipt_exists', bool(receipts))
    for receipt_path in receipts:
        receipt = audit.json(receipt_path)
        audit.require(label+':actual_training_cache:'+receipt_path.parent.name,
                      resolve(receipt['path']) == Path(cache['path'])
                      and receipt['identity'] == cache['identity']
                      and receipt['identity_sha256'] == cache['identity_sha256']
                      and receipt.get('public_split_derivation_sha256') == cache['derivation_sha256'])
        environment = audit.json(receipt_path.with_name('environment.json'))
        audit.require(label+':training_code_snapshot', bool(environment.get('code_hashes')))
        for source, digest in environment.get('code_hashes', {}).items():
            audit.require(label+':training_code:'+source, audit.source(source) == digest)
        audit.require(label+':head_training_code',
                      environment['head_code_sha256'] == audit.source(PROJECT/'ttnet/src/models/event.py'))
    epochs, sampling = collect_epochs(run, corpus, cfg, audit)
    states, checkpoints = {}, {}
    with torch.random.fork_rng(devices=[]):
        head = build_event_head(cfg)
    structure = head.state_dict()
    head_parameters = sum(p.numel() for p in head.parameters())
    for name in ('best', 'last'):
        path = run/'checkpoints'/f'{name}.pt'
        digest = audit.source(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        states[name] = checkpoint
        audit.require(label+':'+name+'_config', checkpoint['config'] == cfg)
        current = {k.removeprefix('head.'):v for k,v in checkpoint['system'].items() if k.startswith('head.')}
        audit.require(label+':'+name+'_head_structure', set(current) == set(structure)
                      and all(current[k].shape == structure[k].shape for k in structure))
        for stage in ('global', 'local', 'table'):
            source = (cfg.get('table_frontend') or {}).get('checkpoint') if stage == 'table' else cfg[stage+'_checkpoint']
            if not source:
                continue
            source = resolve(source)
            source_sha = audit.source(source)
            if str(source) not in originals:
                originals[str(source)] = torch.load(source, map_location='cpu', weights_only=False)
            prefix = 'table_encoder.' if stage == 'table' else stage+'_model.'
            audit.require(label+':'+name+'_'+stage+'_frozen_tensors', frozen_weights(checkpoint, originals[str(source)], prefix))
            if stage != 'table':
                audit.require(label+':'+stage+'_source_sha', cfg[stage+'_sha256'] == source_sha)
        checkpoints[name] = dict(path=str(path), sha256=digest, epoch=checkpoint['epoch'],
                                  head_tensor_sha256=tensor_digest(checkpoint['system'], 'head.'))
    best, last = states['best'], states['last']
    selected = min(epochs.values(), key=lambda row:(-row['continuous']['macro_f1'], row['continuous']['landing']['fp'], row['epoch']))
    audit.require(label+':best_selection_rule', selected['epoch'] == selection['epoch'] == best['epoch']
                  and selected['continuous'] == selection['continuous']
                  and selected['thresholds'] == selection['thresholds'] == best['thresholds'])
    audit.require(label+':finished_training', runtime.get('status') == 'trained_pending_eval'
                  and runtime['epochs_completed'] == last['epoch'] == max(epochs)
                  and last.get('step') == best.get('step') == 0)
    step = state['steps'][label+'_evaluate']; command = step.get('command', [])
    audit.require(label+':recorded_evaluation_output', '--output' in command)
    evaluation = resolve(command[command.index('--output')+1])
    audit.require(label+':evaluation_binding', evaluation.is_relative_to(run/'evaluations')
                  and (not spec.get('evaluation') or resolve(spec['evaluation']) == evaluation))
    audit.source(run/'predictions/best_pp_continuous.json')
    train_rows = read(run/'predictions/best_pp_continuous.json')
    evaluated = validate_evaluation(evaluation, cfg, best, checkpoints['best']['sha256'], selection,
                                    train_rows, cache, corpus, manifest, truth, audit, label)
    benchmark = validate_benchmark(run, cfg, head_parameters, evaluated['performance'], audit)
    metric = evaluated['metrics']
    row = dict(experiment_id=run.name, name=label, arm=spec['arm'], seed=spec['seed'],
               split_id=cfg['split_id'], dataset_id=cfg['dataset_id'],
               config_sha256=audit.source(run/'config.yaml'),
               best_epoch=best['epoch'], epochs_completed=max(epochs),
               best_sha256=checkpoints['best']['sha256'], head_parameters=head_parameters,
               macro_f1=metric['macro_f1'], **{k:metric['overall'][k] for k in ('tp', 'fp', 'fn')},
               e2e_fps=metric['e2e_fps'], threshold=best['thresholds'][0],
               hardware=evaluated['performance']['hardware'], backend=evaluated['performance']['backend'],
               precision=evaluated['performance']['precision'],
               head_forward_p50_ms=benchmark['head_forward_p50_ms'], head_forward_p95_ms=benchmark['head_forward_p95_ms'],
               peak_memory_mib=max(v['peak_cuda_bytes'] for v in evaluated['performance']['videos'])/1024**2,
               config=str(run/'config.yaml'), checkpoint=str(run/'checkpoints/best.pt'),
               metrics=str(evaluation/'metrics.json'), errors=str(evaluation/'errors.html'),
               benchmark=benchmark['path'], status='completed')
    for angle, score in metric['by_angle'].items():
        for key in ('f1', 'tp', 'fp', 'fn'):
            row[angle+'_'+key] = score[key]
    curve = [dict(epoch=e['epoch'], train=e['train'], val_event_loss=e['val_event_loss'],
                  continuous={k:e['continuous'][k] for k in ('macro_f1', 'landing')},
                  thresholds=e['thresholds'], schedule_sha256=e['schedule_sha256'],
                  sampling_counts=e['sampling_counts'], patience=e['patience'], lr=e['lr'],
                  elapsed_sec=e['elapsed_sec'], peak_cuda_bytes=e['peak_cuda_bytes'])
             for _,e in sorted(epochs.items())]
    return dict(row=row, config=cfg, checkpoints=checkpoints, sampling=sampling,
                epochs=curve, benchmark=benchmark,
                evaluation=str(evaluation), training_cache_receipts=[str(p) for p in receipts],
                **evaluated)


def event_analysis(runs):
    first = next(iter(runs.values()))['ground_truth']
    if not all(run['ground_truth'] == first for run in runs.values()):
        raise ReportError('cross-run public GT populations differ')
    common, paired = {}, {}
    for arm in ARMS:
        chosen = [run for run in runs.values() if run['row']['arm'] == arm]
        missed = set(first)-set.union(*(run['matched_events'] for run in chosen))
        common[arm] = [first[k] for k in sorted(missed)]
    for left, right in [('F0', 'concat'), ('concat', 'gated'), ('F0', 'gated')]:
        by_seed = {}
        for seed in SEEDS:
            a, b = runs[f'{left}_seed{seed}'], runs[f'{right}_seed{seed}']
            by_seed[str(seed)] = dict(macro_f1_delta=b['row']['macro_f1']-a['row']['macro_f1'],
                newly_matched=[first[k] for k in sorted(b['matched_events']-a['matched_events'])],
                lost_matches=[first[k] for k in sorted(a['matched_events']-b['matched_events'])],
                fp_delta=b['row']['fp']-a['row']['fp'])
        changes = [v['macro_f1_delta'] for v in by_seed.values()]
        paired[left+'_to_'+right] = dict(by_seed=by_seed, mean_delta=float(np.mean(changes)),
            std_population=float(np.std(changes)), positive_seeds=sum(v > 0 for v in changes))
    all_missed = set(first)-set.union(*(run['matched_events'] for run in runs.values()))
    return dict(common_false_negatives_by_arm=common,
                common_false_negatives_all_nine=[first[k] for k in sorted(all_missed)], paired=paired,
                limitation='GT event lists describe TP/FN only. FP differences are full-video totals; no FP is assigned to a nearby GT event.')


def aggregate(runs):
    result = {}
    for arm in ARMS:
        rows = [runs[f'{arm}_seed{seed}']['row'] for seed in SEEDS]
        angles = runs[f'{arm}_seed42']['metrics']['by_angle']
        result[arm] = dict(macro_f1_mean=float(np.mean([r['macro_f1'] for r in rows])),
            macro_f1_std_population=float(np.std([r['macro_f1'] for r in rows])),
            by_seed={str(r['seed']):r for r in rows},
            mean_counts={k:float(np.mean([r[k] for r in rows])) for k in ('tp', 'fp', 'fn')},
            e2e_fps_mean=float(np.mean([r['e2e_fps'] for r in rows])),
            head_parameters=rows[0]['head_parameters'],
            by_angle_mean={angle:{k:float(np.mean([runs[f'{arm}_seed{s}']['metrics']['by_angle'][angle][k] for s in SEEDS]))
                                  for k in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn')} for angle in angles})
    return result


def audit_study(task):
    task = resolve(task); audit = Audit(); state = audit.json(task/'task.json')
    audit.source(__file__); audit.source(task/'pretraining_plan.md')
    audit.require('nine_preregistered_runs', state.get('seeds') == list(SEEDS)
                  and state.get('arms') == list(ARMS) and len(state.get('experiments', [])) == 9
                  and {(s['seed'], s['arm']) for s in state['experiments']} == {(s,a) for s in SEEDS for a in ARMS}
                  and len({s['run'] for s in state['experiments']}) == 9
                  and {s['name'] for s in state['experiments']} == {f'{a}_seed{s}' for s in SEEDS for a in ARMS})
    for spec in state['experiments']:
        for suffix in ('smoke', 'train', 'evaluate', 'benchmark'):
            step = state.get('steps', {}).get(spec['name']+'_'+suffix, {})
            audit.require('ready:'+spec['name']+'_'+suffix, step.get('status') == 'completed')
    initial, _ = load_config(resolve(state['experiments'][0]['config']))
    torch.set_num_threads(initial.get('torch_threads', 4))
    corpus = EventCorpus(initial['dataset']); attach_pp(corpus, initial['pp_annotations'])
    attach_public_validation(corpus, initial['public_validation_root'], initial)
    split = corpus.public_split_audit
    audit.require('PP154_public15_split', split['split_id'] == state['protocol']['split_id']
                  and len(corpus.pp_annotations) == 154
                  and len(split['moved_pp_validation_videos']) == 40
                  and split['video_counts'] == {'train':158, 'val':16, 'test':7}
                  and split['public_training_samples'] == 0
                  and all(a['split'] == 'train' for a in corpus.pp_annotations.values()))
    root = resolve(initial['public_validation_root'])
    manifest = audit.json(root/'manifest.json'); truth = jsonl(root/'gt_public.jsonl'); audit.source(root/'gt_public.jsonl')
    audit.require('fixed_public_counts', dict(videos=len(manifest['videos']),
        frames=sum(v['n_frames'] for v in manifest['videos']), valid_gt=sum(g['valid'] for g in truth),
        ignored_gt=sum(not g['valid'] for g in truth)) == PUBLIC_EXPECTED)
    caches = {name:validate_cache(spec, corpus, audit) for name,spec in state['caches'].items()}
    originals, runs = {}, {}
    for spec in state['experiments']:
        print('Auditing '+spec['name'], flush=True)
        runs[spec['name']] = validate_run(spec, state, corpus, caches, manifest, truth, audit, originals)
    keys = [k for k in PROTOCOL_KEYS if k != 'seed'] + ['public_validation_root']
    configs = [r['config'] for r in runs.values()]
    audit.require('cross_run_training_controls', all({k:c.get(k) for k in keys} == {k:configs[0].get(k) for k in keys} for c in configs))
    audit.require('cross_run_backend', len({(r['row']['hardware'], r['row']['backend'], r['row']['precision']) for r in runs.values()}) == 1)
    events = event_analysis(runs); scores = aggregate(runs)
    references = optional_references(task, caches, corpus, manifest, truth, audit)
    diagnostic = locator_diagnostics(task, caches, root, truth, audit)
    candidate = max(ARMS, key=lambda a:(scores[a]['macro_f1_mean'], -scores[a]['mean_counts']['fp'], -ARMS.index(a)))
    audit.unchanged()
    serial_runs = {}
    for name, result in runs.items():
        serial_runs[name] = {k:v for k,v in result.items() if k not in ('matched_events', 'ground_truth')}
    analysis = dict(schema='public_event_study_report_v1', status='completed', created=timestamp(),
        task=str(task), validation_set='pp_public', validation_role='Final validation; used for checkpoint/hyperparameter selection, not independent test.',
        seeds=list(SEEDS), arms=list(ARMS), split_audit=split, cache_audit=caches,
        runs=serial_runs, arm_summary=scores, highest_mean_validation_candidate=candidate,
        event_analysis=events, historical_fixed_references=references,
        public_locator_diagnostics=diagnostic,
        total_full_overlay_videos=sum(r['overlays']['videos'] for r in runs.values()),
        previously_decoded_overlay_frames=sum(r['overlays']['decoded_frames'] for r in runs.values()),
        checks=audit.checks, sources_sha256=audit.sources, human_review_performed=False,
        mutable_source_policy='The task.json SHA is an orchestration snapshot. Later orchestration-status/report links may be appended there; immutable model runs, evaluations, configurations and all other source artifacts must retain their recorded hashes.',
        limitations=['PP public is validation and selects checkpoints/thresholds; no independent-test claim.',
            'PP154/public15 is a new split. Old PP114/40 scores are not a matched before/after comparison.',
            'Three seeds share one validation set; no statistical-significance claim.',
            'Overlay artifacts are hash-checked against actual complete-decoding receipts; this report does not re-decode them.',
            'FPS is local offline full-video timing, including decode/serialization and excluding initialization, scoring, overlays and writes.',
            'Benchmark p50/p95 is head-only. MAC/FLOP hooks cover only the stated operators; no complete-system forward latency claim.',
            'No human semantic review or sign-off is claimed.'])
    return analysis


def relative(path, base):
    return os.path.relpath(path, base).replace(os.sep, '/')


def write_reports(task, analysis):
    """All writes are owned report files or additive lightweight record receipts."""
    task = resolve(task)
    os.environ.setdefault('MPLCONFIGDIR', str(task/'report_matplotlib_config'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r['row'] for r in analysis['runs'].values()]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for i, seed in enumerate(SEEDS):
        for arm in ARMS:
            run = analysis['runs'][f'{arm}_seed{seed}']; epochs = run['epochs']
            x = [r['epoch'] for r in epochs]
            axes[i,0].plot(x, [r['train']['event'] for r in epochs], label=arm)
            axes[i,1].plot(x, [r['continuous']['macro_f1'] for r in epochs], label=arm)
            selected = next(r for r in epochs if r['epoch'] == run['row']['best_epoch'])
            axes[i,1].scatter(selected['epoch'], selected['continuous']['macro_f1'], marker='*', s=130)
        axes[i,0].set_title(f'seed{seed}: training event BCE'); axes[i,1].set_title(f'seed{seed}: public angle macro F1')
        axes[i,1].set_ylim(0,1)
        for ax in axes[i]:
            ax.set_xlabel('Epoch'); ax.grid(alpha=.25); ax.legend()
    fig.suptitle('PP154 train / public15 validation — frozen B locators + 13-frame TCN')
    fig.savefig(task/'training_curves.png', dpi=160); plt.close(fig)
    fields = list(rows[0])
    with (task/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ['# PP154训练 / pp_public验证：三种子事件实验', '',
        '状态：completed（机器产物审计）。PP public是最终验证集，已用于checkpoint、阈值与方案选择；不称为独立测试。旧40视频已经并入训练。', '',
        '| 方案 | seed42 | seed43 | seed44 | 均值 ± 总体标准差 | 平均TP/FP/FN | 平均FPS | Head参数 |',
        '|---|---:|---:|---:|---:|---|---:|---:|']
    for arm in ARMS:
        s = analysis['arm_summary'][arm]; seed_values = ' | '.join(f"{s['by_seed'][str(seed)]['macro_f1']:.2%}" for seed in SEEDS)
        counts = '/'.join(f"{s['mean_counts'][k]:.2f}" for k in ('tp','fp','fn'))
        lines.append(f"| {arm} | {seed_values} | {s['macro_f1_mean']:.2%} ± {s['macro_f1_std_population']*100:.2f}pp | {counts} | {s['e2e_fps_mean']:.2f} | {s['head_parameters']:,} |")
    lines += ['', '标准差使用ddof=0；先按每个run三视角宏F1评分，再对三个种子取均值。TP/FP/FN为各run计数的均值，不混成单次评估。', '',
        '| 方案 | 视角 | 平均P | 平均R | 平均F1 | 平均TP/FP/FN |', '|---|---|---:|---:|---:|---|']
    for arm in ARMS:
        for angle, s in analysis['arm_summary'][arm]['by_angle_mean'].items():
            counts = '/'.join(f"{s[k]:.2f}" for k in ('tp','fp','fn'))
            lines.append(f"| {arm} | {angle} | {s['precision']:.2%} | {s['recall']:.2%} | {s['f1']:.2%} | {counts} |")
    candidate = analysis['highest_mean_validation_candidate']
    cfg = next(iter(analysis['runs'].values()))['config']
    lines += ['', f"当前验证均值最高：{candidate}。冻结B SE Global与Residual FPN Local，阈值{cfg['global_threshold']}/{cfg['local_threshold']}；13原帧TCN，{cfg['optimizer']} lr{cfg['lr']:g}/wd{cfg['weight_decay']:g}、batch{cfg['effective_batch']}、AMP、最多{cfg['epochs']}轮、patience{cfg['patience']}。训练PP154 + OpenTT games1–4，两来源50/50；OpenTT game5仅保留辅助稀疏诊断。", '',
        f"九组共{analysis['total_full_overlay_videos']}条完整叠加视频，解码收据共{analysis['previously_decoded_overlay_frames']:,}帧；本报告复核文件SHA及实际解码收据，没有重复解码或宣称人类审核。", '',
        '三种子共同FN（同一GT在该方案三个种子均漏检）：'+ '；'.join(f"{a} {len(analysis['event_analysis']['common_false_negatives_by_arm'][a])}个" for a in ARMS)+'。',
        f"全部九组共同FN：{len(analysis['event_analysis']['common_false_negatives_all_nine'])}个。具体事件与跨方案追回/回退见analysis.json。", '']
    if candidate == 'gated':
        paired = analysis['event_analysis']['paired']['concat_to_gated']
        lines += [f"gated均值比concat高{paired['mean_delta']*100:.2f}pp，但仅{paired['positive_seeds']}/3种子改善，且总体标准差更大。因此它是按既定规则选出的public验证候选，尚不能声称稳定优于concat；单次最佳成绩与三种子均值分别报告。", '']
    for pair, data in analysis['event_analysis']['paired'].items():
        details = '，'.join(f"seed{s} {v['macro_f1_delta']*100:+.2f}pp" for s,v in data['by_seed'].items())
        lines.append(f"- {pair}：{details}；均值{data['mean_delta']*100:+.2f}pp，{data['positive_seeds']}/3种子提升。")
    diagnostic = analysis['public_locator_diagnostics']
    if diagnostic.get('status') == 'verified':
        exact = diagnostic['final_prediction']['exact']; neighbor = diagnostic['final_prediction']['within_one_frame']
        lines += ['', f"GT辅助定位诊断：最终坐标精确帧覆盖{exact['covered']}/{exact['total']}（{exact['coverage']:.2%}），±1原帧覆盖{neighbor['covered']}/{neighbor['total']}（{neighbor['coverage']:.2%}）。该邻域覆盖仅把固定落点GT与邻帧球坐标比较；未包含事件判别、FP及一对一匹配，不能当作事件F1或严格事件召回上限。", '',
            f"[定位诊断解释]({relative(diagnostic['markdown'],task)}) · [逐事件定位诊断]({relative(diagnostic['json'],task)})。"]
    references = analysis['historical_fixed_references']
    if references.get('completed'):
        lines += ['', '旧内部选定checkpoint的固定公共参考（原checkpoint与原阈值，无新阈值搜索）：', '',
            '| 固定参考 | 旧best epoch | 原Bounce阈值 | 公共宏F1 | TP/FP/FN | FPS | 结果 |',
            '|---|---:|---:|---:|---|---:|---|']
        for name, ref in references['completed'].items():
            c = ref['overall']
            lines.append(f"| {name} | {ref['epoch']} | {ref['thresholds'][0]} | {ref['macro_f1']:.2%} | {c['tp']}/{c['fp']}/{c['fn']} | {ref['e2e_fps']:.2f} | [错误与视频]({relative(Path(ref['evaluation'])/'errors.html',task)}) |")
        lines += ['', '旧参考的epoch与阈值由旧内部验证选定；新九组的epoch与阈值由pp_public选定。公共评估数据相同，但选型协议也变化，因此只作实际参考，不作为纯训练数据消融。']
    if references.get('pending'):
        lines += ['', '固定历史参考尚未完成：'+ '、'.join(r['name'] for r in references['pending'])+'。不影响九组本体的机器验收；后续参考评估完成后可重新生成报告。']
    lines += ['', '逐运行结果与全部错误入口：', '', '| 运行 | 宏F1 | TP/FP/FN | best/完成轮数 | 阈值 | FPS | 产物 |', '|---|---:|---|---|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['name']} | {row['macro_f1']:.2%} | {row['tp']}/{row['fp']}/{row['fn']} | {row['best_epoch']}/{row['epochs_completed']} | {row['threshold']} | {row['e2e_fps']:.2f} | [错误与视频]({relative(row['errors'],task)}) · [指标]({relative(row['metrics'],task)}) · [权重]({relative(row['checkpoint'],task)}) |")
    lines += ['', '[逐轮曲线](training_curves.png) · [机器审计](analysis.json) · [逐运行CSV](comparison.csv) · [独立完成收据](completion.json)。', '',
        '速度为同一台本机PyTorch AMP离线完整视频观测值，包含解码至JSON序列化，排除初始化、评分、叠加与写盘；head-only p50/p95及部分算子MAC/FLOP另列在各benchmark。', '',
        'PP154/public15是新数据协议，不能与旧PP114/40内部成绩作直接收益比较。三种子不构成显著性检验；未使用OpenTT test选型。']
    (task/'summary.md').write_text('\n'.join(lines)+'\n')
    save(task/'analysis.json', analysis)
    report_hashes = {name:sha(task/name) for name in ('analysis.json','comparison.csv','summary.md','training_curves.png')}
    completion = dict(schema='public_event_completion_v1', status='completed', completed=timestamp(),
        task=str(task), validation_set='pp_public', runs=9, seeds=list(SEEDS), arms=list(ARMS),
        highest_mean_validation_candidate=candidate, human_review_performed=False,
        machine_audit_checks=len(analysis['checks']), all_checks_passed=all(c['passed'] for c in analysis['checks']),
        sources_sha256=analysis['sources_sha256'], reports_sha256=report_hashes,
        model_run_files_modified=False, completion_scope='Nine-run training/evaluation/parity/error-coverage/overlay-decode-receipt/benchmark machine audit.')
    save(task/'completion.json', completion)
    for run in analysis['runs'].values():
        row = run['row']; record = PROJECT/'experiments/records'/row['experiment_id']
        summary = ['# PP public事件训练结果', '', '状态：completed（独立机器完成收据；模型run.json保留原始训练状态）。', '',
            f"方案{row['arm']}，seed{row['seed']}；best epoch {row['best_epoch']} / 完成{row['epochs_completed']}轮；阈值{row['threshold']}。", '',
            f"PP public验证宏F1 {row['macro_f1']:.4%}；TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}；完整视频{row['e2e_fps']:.2f}FPS。", '',
            f"[完整三种子比较]({relative(task/'summary.md',record)}) · [全部错误/完整叠加视频]({relative(row['errors'],record)}) · [指标]({relative(row['metrics'],record)}) · [最佳权重]({relative(row['checkpoint'],record)})。", '',
            '训练使用全部154个PP视频；pp_public用于验证和选型，不作为独立外测。全部15视频完整帧cache/online及落点决策验收通过；所有FP/FN/无坐标事件均有图和完整视频。未宣称人类审核。']
        (record/'summary.md').write_text('\n'.join(summary)+'\n')
        reference = dict(status='completed', validation_set='pp_public', eval_id='public',
            metrics=row['metrics'], metrics_sha256=sha(row['metrics']), checkpoint=row['checkpoint'],
            checkpoint_sha256=row['best_sha256'], best_epoch=row['best_epoch'], metrics_summary=dict(macro_f1=row['macro_f1'],tp=row['tp'],fp=row['fp'],fn=row['fn']),
            errors=row['errors'], benchmark=row['benchmark'], study_analysis=str(task/'analysis.json'),
            study_analysis_sha256=report_hashes['analysis.json'], completion_receipt=str(task/'completion.json'))
        save(record/'evaluations/public.json', reference)
        save(record/'public_completion.json', dict(schema='public_event_run_completion_v1', status='completed',
            completed=completion['completed'], experiment_id=row['experiment_id'], human_review_performed=False,
            model_run_files_modified=False, study_completion=str(task/'completion.json'),
            study_completion_sha256=sha(task/'completion.json'), evaluation=reference))
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, default=DEFAULT_TASK)
    parser.add_argument('--check-only', action='store_true', help='Require/audit all nine runs without writing any report')
    args = parser.parse_args()
    analysis = audit_study(args.task)
    if args.check_only:
        print(json.dumps(dict(status='passed', runs=len(analysis['runs']), checks=len(analysis['checks'])), ensure_ascii=False))
    else:
        result = write_reports(args.task, analysis)
        print(json.dumps({k:result[k] for k in ('status','runs','highest_mean_validation_candidate')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
