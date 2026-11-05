"""Audit and report the three-arm B-locator study without changing its state.

Reads existing CPU artifacts only. No inference, training, threshold search,
public/test evaluation, or task.json mutation is performed. Missing inputs fail
before reports are written; failed integrity checks are retained in the report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from ttnet.common import identity
from ttnet.configuration import load_config
from ttnet.events.continuous import score_continuous
from ttnet.events.metrics import peaks
from ttnet.models.build import build_event_head

ARMS = ('F0', 'geometry', 'concat')
VIEWS = ('side', 'diagonal', 'front')
PROTOCOL_KEYS = (
    'seed', 'dataset_id', 'split_id', 'train_samples_sha256', 'val_samples_sha256',
    'mode', 'temporal_mode', 'tracker_type', 'association', 'global_sha256',
    'local_sha256', 'global_threshold', 'local_threshold', 'localization_threshold',
    'global_localizer_type', 'local_localizer_type', 'global_input', 'local_input',
    'local_crop_side_1080', 'window', 'event_window_mode', 'target_fps',
    'time_encoding', 'context_head', 'table_geometry_version', 'motion_max_step_fraction',
    'optimizer', 'lr', 'weight_decay', 'effective_batch', 'micro_batch', 'epochs',
    'patience', 'lr_schedule', 'warmup_epochs', 'sampling_protocol', 'augmentation',
    'loss_weights', 'threshold_grid', 'nms_sec', 'head_dropout', 'amp',
    'pp_positive_fraction', 'pp_hard_negative_fraction', 'pp_hard_negative_manifest_sha256',
)


class ReportError(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT/path).resolve()


def normalize(value):
    if isinstance(value, dict):
        return {key:normalize(item) for key,item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, str) and (value.startswith(('/', 'data/', 'out/', 'ttnet/'))):
        return str(resolve(value))
    return value


class Audit:
    def __init__(self):
        self.checks = []
        self.sources = {}

    def check(self, name, passed, **details):
        self.checks.append(dict(name=name, passed=bool(passed), **details))

    def source(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise ReportError(f'missing required artifact: {path}')
        if str(path) not in self.sources:
            self.sources[str(path)] = sha(path)
        return self.sources[str(path)]


def evaluation_path(state, arm):
    """Honor the persisted recovery output rather than inventing a new attempt."""
    step = state.get('steps', {}).get(arm+'_evaluate', {})
    command = step.get('command', [])
    recorded = state.get('evaluations', {}).get(arm)
    output = None
    if step.get('status') == 'completed' and '--output' in command:
        output = resolve(command[command.index('--output')+1])/'metrics.json'
    if recorded:
        recorded = resolve(recorded)
        if output is not None and output != recorded:
            raise ReportError(f'{arm}: task evaluation and completed step disagree')
        output = recorded
    output = output or resolve(state['runs'][arm])/'evaluations/continuous/metrics.json'
    if not output.is_file():
        raise ReportError(f'{arm}: completed evaluation metrics unavailable: {output}')
    return output


def collect_epochs(run, planned, audit):
    """Recover duplicate epoch receipts across attempts without hiding conflicts."""
    by_epoch = {}; duplicates = []
    expected = {int(row['epoch']):row for row in planned['epochs']}
    for path in sorted(Path(run).glob('attempts/*/epochs.jsonl')):
        audit.source(path)
        for row in jsonl(path):
            epoch = int(row['epoch'])
            if epoch in by_epoch:
                old = by_epoch[epoch]
                # Elapsed time/peak memory may differ after a recovery, outcomes may not.
                keys = ('schedule_sha256', 'thresholds', 'continuous', 'sampling_counts')
                audit.check(f'{run.name}:duplicate_epoch_{epoch}',
                            all(old.get(k) == row.get(k) for k in keys))
                duplicates.append(epoch)
            by_epoch[epoch] = row
    if not by_epoch:
        raise ReportError(f'no completed epoch records: {run}')
    audit.check(f'{run.name}:contiguous_epochs', sorted(by_epoch) == list(range(1,max(by_epoch)+1)))
    evidence = []
    for epoch,row in sorted(by_epoch.items()):
        wanted = expected.get(epoch, {})
        count = row.get('sampling_counts', {})
        matches = bool(wanted) and row.get('schedule_sha256') == wanted.get('sha256')
        matches = matches and count.get('total') == wanted.get('samples')
        matches = matches and count.get('pp') == wanted.get('sources', {}).get('pp')
        matches = matches and count.get('pp_positive') == wanted.get('pp_positive')
        audit.check(f'{run.name}:epoch_{epoch}_planned_sampling', matches,
                    actual_sha256=row.get('schedule_sha256'), planned_sha256=wanted.get('sha256'))
        evidence.append(dict(epoch=epoch, matched=matches, sha256=row.get('schedule_sha256'),
                             sampling_counts=count))
    return by_epoch, dict(epochs=evidence, duplicate_receipts=duplicates)


def ground_truth_manifest(metrics):
    manifest = {}; matched = set()
    for vid,score in metrics['per_video'].items():
        gt = score['ground_truth']; landing = score['landing']
        owners = {int(pair['ground_truth']) for pair in landing['matches']}
        if len(owners) != landing['tp'] or owners & set(landing['false_negatives']):
            raise ReportError(f'{vid}: inconsistent GT matching indices')
        if owners | set(landing['false_negatives']) != set(range(len(gt))):
            raise ReportError(f'{vid}: matching is not a partition of GT')
        for i,event in enumerate(gt):
            key = (vid, int(event['frame_id']))
            if key in manifest:
                raise ReportError(f'duplicate GT event: {key}')
            manifest[key] = (float(event['x']), float(event['y']))
            if i in owners:
                matched.add(key)
    return manifest, matched


def aggregate(scores):
    counts = {key:sum(s[key] for s in scores) for key in ('tp','fp','fn')}
    tp,fp,fn = (counts[k] for k in ('tp','fp','fn'))
    return dict(counts, precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn))


def paired_events(historical, results, diagnostics):
    manifest, old_matches = ground_truth_manifest(historical)
    diag = {(row['video_id'],row['frame_id']):row for row in diagnostics['events']}
    if len(manifest) != 189 or set(manifest) != set(diag) or len(old_matches) != 144:
        raise ReportError('expected the fixed189 GT, old144 TP /45 FN diagnostic population')
    for key,xy in manifest.items():
        if tuple(diag[key]['ground_truth_xy']) != xy:
            raise ReportError(f'diagnostic GT coordinates changed: {key}')
        if bool(diag[key]['old_landing_fn']) != (key not in old_matches):
            raise ReportError(f'diagnostic historical matching changed: {key}')
    matches = {}
    for arm,result in results.items():
        current, matched = ground_truth_manifest(result)
        if current != manifest:
            raise ReportError(f'{arm}: GT coordinates/population differ from historical reference')
        matches[arm] = matched
    rows=[]
    for key in sorted(manifest):
        d = diag[key]
        rows.append(dict(video_id=key[0],frame_id=key[1],view=d['view'],group=d['group'],
                         old_matched=key in old_matches,
                         exact_frame_repair=bool(d['old_landing_fn'] and d['exact_coverage_transition']=='fixed'),
                         no_roi_13window=d['yolo_13window_valid_frames']==0,
                         matched={arm:key in matched for arm,matched in matches.items()}))
    slices = {
        'old45FN':lambda r:not r['old_matched'], 'old144TP':lambda r:r['old_matched'],
        'six_exact_frame_repairs':lambda r:r['exact_frame_repair'],
        'group22':lambda r:r['group']=='22', 'other_groups':lambda r:r['group']!='22',
        'no_roi_entire_13window':lambda r:r['no_roi_13window'],
        'roi_present_somewhere_13window':lambda r:not r['no_roi_13window'],
    }
    slices.update({'view:'+v:(lambda r,v=v:r['view']==v) for v in VIEWS})
    buckets = {}
    for name,predicate in slices.items():
        chosen = [r for r in rows if predicate(r)]
        buckets[name] = dict(events=len(chosen), old_matched=sum(r['old_matched'] for r in chosen),
                             arms={arm:dict(tp=sum(r['matched'][arm] for r in chosen),
                                            fn=sum(not r['matched'][arm] for r in chosen),
                                            recovered_old_fn=sum(not r['old_matched'] and r['matched'][arm] for r in chosen),
                                            lost_old_tp=sum(r['old_matched'] and not r['matched'][arm] for r in chosen))
                                   for arm in ARMS})
    pairwise={}
    for left,right in (('F0','geometry'),('F0','concat'),('geometry','concat')):
        pairwise[left+'_to_'+right] = dict(
            newly_matched=[list(k) for k in sorted(matches[right]-matches[left])],
            lost_matches=[list(k) for k in sorted(matches[left]-matches[right])])
    by_group={}
    for arm,metric in results.items():
        by_group[arm]={group:aggregate([score['landing'] for vid,score in metric['per_video'].items()
                                      if (vid[:2]==group if group!='other_groups' else vid[:2]!='22')])
                       for group in sorted({vid[:2] for vid in metric['per_video']})+['other_groups']}
    return dict(events=rows,buckets=buckets,pairwise=pairwise,by_group=by_group,
                limitation='GT-only diagnosis. Event buckets report TP/FN; FP is attributed only to full video groups, not to a nearby GT event.')


def nested_differences(left, right, path='', output=None):
    output = [] if output is None else output
    if isinstance(left,dict) and isinstance(right,dict):
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                output.append((path+'.'+key,None))
            else:
                nested_differences(left[key],right[key],path+'.'+key,output)
    elif isinstance(left,list) and isinstance(right,list):
        if len(left)!=len(right):
            output.append((path+'.length',None))
        for i,(a,b) in enumerate(zip(left,right)):
            nested_differences(a,b,f'{path}[{i}]',output)
    elif isinstance(left,(int,float)) and not isinstance(left,bool) and isinstance(right,(int,float)) and not isinstance(right,bool):
        if left!=right:
            output.append((path,abs(float(left)-float(right))))
    elif left!=right:
        output.append((path,None))
    return output


def frame_rows(rows, label):
    ids = [int(row['frame_id']) for row in rows]
    if ids != list(range(len(ids))):
        raise ReportError(f'{label}: expected every original frame once in chronological order')
    for row in rows:
        p = row.get('probabilities', [])
        if len(p)!=2 or not all(math.isfinite(v) and 0<=v<=1 for v in p):
            raise ReportError(f'{label}: invalid event probabilities')
    return {int(row['frame_id']):row for row in rows}


def metric_signature(metric):
    decisions={}
    for original,score in metric['per_video'].items():
        vid=original.removeprefix('pp_')
        if vid in decisions:raise ReportError('duplicate normalized video identity: '+vid)
        def point(item):return [item['frame_id'],item.get('x'),item.get('y')]
        gt=score['ground_truth'];pred=score['predictions'];landing=score['landing']
        temporal_gt=score.get('temporal_ground_truth',gt);detected=score['detected']
        temporal=score['continuous_bounce']
        decisions[vid]=dict(
            ground_truth=[point(item) for item in gt],
            matches=[[point(gt[pair['ground_truth']]),point(pred[pair['prediction']])] for pair in landing['matches']],
            false_negatives=[point(gt[i]) for i in landing['false_negatives']],
            false_positives=[point(pred[i]) for i in landing['false_positives']],
            temporal_ground_truth=[item['frame_id'] for item in temporal_gt],
            temporal_matches=[[temporal_gt[pair['ground_truth']]['frame_id'],detected[pair['prediction']]['frame_id']] for pair in temporal['matches']],
            temporal_false_negatives=[temporal_gt[i]['frame_id'] for i in temporal['false_negatives']],
            temporal_false_positives=[detected[i]['frame_id'] for i in temporal['false_positives']],
            detected_frames=[item['frame_id'] for item in detected],
        )
    return dict(landing={k:metric['landing'][k] for k in ('tp','fp','fn')},
                continuous_bounce={k:metric['continuous_bounce'][k] for k in ('tp','fp','fn')},
                macro_f1=metric['macro_f1'],
                by_view={view:{k:s[k] for k in ('tp','fp','fn')} for view,s in metric['by_view'].items()},
                per_video_decisions=decisions)


def compare_cached_online(cached, online, annotations, threshold, nms_sec, atol=2e-4):
    if set(cached)!=set(online) or set(online)!=set(annotations):
        raise ReportError('cache, online predictions and annotations must have identical40 video identities')
    probability_errors=[]; by_video={}; examples=[]
    total=Counter(); ball_fields=Counter(); max_ball=0.; max_xy=0.; max_conf=0.
    for vid in sorted(online):
        a=frame_rows(cached[vid], 'cache:'+vid);b=frame_rows(online[vid], 'online:'+vid)
        if a.keys()!=b.keys():
            raise ReportError(f'{vid}: cache/online frame coverage differs')
        counts=Counter(); local_max=0.; ball_max=0.
        for fid in a:
            x,y=a[fid],b[fid]; delta=np.abs(np.asarray(x['probabilities'])-y['probabilities'])
            probability_errors.extend(delta.tolist());local_max=max(local_max,float(delta.max()))
            counts['frames']+=1;counts['probability_unequal_frames']+=bool(delta.max()>0)
            counts['probability_outside_atol_frames']+=bool(delta.max()>atol)
            threshold_diff=(x['probabilities'][0]>=threshold)!=(y['probabilities'][0]>=threshold)
            counts['bounce_threshold_decision_differences']+=threshold_diff
            counts['source_frame_id_differences']+=x.get('source_frame_id')!=y.get('source_frame_id')
            counts['timestamp_unequal_frames']+=x['timestamp_sec']!=y['timestamp_sec']
            counts['timestamp_outside_atol_frames']+=abs(x['timestamp_sec']-y['timestamp_sec'])>1e-7
            diff=nested_differences(x.get('prediction'),y.get('prediction'))
            counts['ball_record_unequal_frames']+=bool(diff)
            for field,error in diff:
                ball_fields[field]+=1
                if error is None:counts['ball_structural_or_categorical_field_differences']+=1
                else:
                    ball_max=max(ball_max,error);max_ball=max(max_ball,error)
                    if field in ('.x','.y'):max_xy=max(max_xy,error)
                    if field=='.confidence':max_conf=max(max_conf,error)
            if len(examples)<20 and (delta.max()>atol or diff or threshold_diff):
                examples.append(dict(video_id=vid,frame_id=fid,probability_max_abs=float(delta.max()),
                                     ball_fields=[p for p,_ in diff][:10]))
        total.update(counts);by_video[vid]=dict(counts,probability_max_abs=local_max,ball_numeric_max_abs=ball_max)
    cache_score=score_continuous(cached,annotations,threshold,nms_sec)
    online_score=score_continuous(online,annotations,threshold,nms_sec)
    event_diffs=[]; match_diffs=[]; landing_diffs=[]
    for vid in sorted(online):
        a,b=cache_score['per_video'][vid],online_score['per_video'][vid]
        ca={r['frame_id'] for r in peaks(cached[vid],[threshold,2.],nms_sec) if r['kind']=='bounce'}
        cb={r['frame_id'] for r in peaks(online[vid],[threshold,2.],nms_sec) if r['kind']=='bounce'}
        if ca!=cb:event_diffs.append(dict(video_id=vid,cache_only=sorted(ca-cb),online_only=sorted(cb-ca)))
        ga={p['ground_truth'] for p in a['landing']['matches']};gb={p['ground_truth'] for p in b['landing']['matches']}
        if ga!=gb:match_diffs.append(dict(video_id=vid,cache_only_gt_indices=sorted(ga-gb),online_only_gt_indices=sorted(gb-ga)))
        la={r['frame_id'] for r in a['predictions']};lb={r['frame_id'] for r in b['predictions']}
        if la!=lb or any(a['landing'][k]!=b['landing'][k] for k in ('tp','fp','fn')):
            landing_diffs.append(dict(video_id=vid,cache_only_landing_frames=sorted(la-lb),online_only_landing_frames=sorted(lb-la),
                                      cache_counts={k:a['landing'][k] for k in ('tp','fp','fn')},
                                      online_counts={k:b['landing'][k] for k in ('tp','fp','fn')}))
        by_video[vid]['cache_events']=len(ca);by_video[vid]['online_events']=len(cb)
        by_video[vid]['cache_reviewed_events']=len(a['detected']);by_video[vid]['online_reviewed_events']=len(b['detected'])
    decision_same=not event_diffs and not match_diffs and not landing_diffs and not total['bounce_threshold_decision_differences']
    exact=(decision_same and not total['probability_unequal_frames'] and not total['ball_record_unequal_frames']
           and not total['source_frame_id_differences'] and not total['timestamp_unequal_frames'])
    status='exact' if exact else 'decision_difference' if not decision_same else 'numerical_difference_same_decisions'
    return dict(status=status,fully_consistent=exact,decisions_equal=decision_same,
                probability_atol=atol,probability_max_abs=max(probability_errors,default=0.),
                probability_mean_abs=float(np.mean(probability_errors)) if probability_errors else None,
                probability_p95_abs=float(np.percentile(probability_errors,95)) if probability_errors else None,
                ball_numeric_max_abs=max_ball,ball_xy_max_abs_px=max_xy,ball_confidence_max_abs=max_conf,
                ball_field_difference_counts=dict(ball_fields),totals=dict(total),per_video=by_video,
                event_decision_differences=event_diffs,matched_GT_differences=match_diffs,landing_decision_differences=landing_diffs,
                cache_metrics=metric_signature(cache_score),online_metrics=metric_signature(online_score),
                examples=examples,
                scope='All fields of the prediction object saved in best_pp_continuous.json, both event probabilities, source frame IDs, timestamps and fixed-threshold/NMS outcomes. This file does not contain all raw trajectory candidates or encoder descriptors.'), online_score


def audit_errors(directory, metrics, rows, audit, label):
    directory=Path(directory);index=directory/'error_index.jsonl';audit.source(index);audit.source(directory/'index.html')
    entries=jsonl(index);actual=set();invalid=[]
    frame_sets={vid:{r['frame_id'] for r in values} for vid,values in rows.items()}
    for entry in entries:
        vid=entry['video_id'];fid=entry['frame_id'];path=directory/entry['visualization']
        if vid not in rows or fid not in frame_sets.get(vid,set()) or not path.is_file() or path.stat().st_size==0:
            invalid.append(dict(video_id=vid,frame_id=fid,path=str(path)))
        for kind in entry['kinds']:actual.add((vid,fid,kind))
    expected=set();missing_overlays=[]
    for vid,score in metrics['per_video'].items():
        overlay=directory/f'{vid}_overlay.mp4'
        if not overlay.is_file() or overlay.stat().st_size==0:missing_overlays.append(str(overlay))
        for group,prefix,preds,gt in (
            ('landing','landing',score['predictions'],score['ground_truth']),
            ('continuous_bounce','continuous',score['detected'],score.get('temporal_ground_truth',score['ground_truth'])),
        ):
            for i in score[group]['false_positives']:expected.add((vid,preds[i]['frame_id'],prefix+'_false_positive'))
            for i in score[group]['false_negatives']:expected.add((vid,gt[i]['frame_id'],prefix+'_false_negative'))
    missing=sorted(expected-actual)
    audit.check(label+':error_coverage',not missing and not invalid and not missing_overlays,
                missing_event_errors=[list(x) for x in missing],invalid_index_entries=invalid,missing_overlays=missing_overlays)
    return dict(videos=len(metrics['per_video']),indexed_frames=len(entries),required_event_errors=len(expected),
                missing_event_errors=[list(x) for x in missing],invalid_index_entries=invalid,
                missing_overlays=missing_overlays,index=str(directory/'index.html'),
                overlays={vid:str(directory/f'{vid}_overlay.mp4') for vid in metrics['per_video']},
                verification='Nonempty files and complete scored event-error index coverage; video codec/frame rendering is not re-decoded by this CPU report.')


def frozen_weights(state, original, prefix):
    current={key[len(prefix):]:value for key,value in state['system'].items() if key.startswith(prefix)}
    expected=original['model']
    return (current.keys()==expected.keys() and
            all(torch.equal(current[key].cpu(),expected[key].cpu()) for key in expected))


def load_arm(arm, run, eval_path, state, planned, locked, audit, model_cache):
    cfg=read(run/'config.yaml');runtime=read(run/'run.json');selection=read(run/'best_selection.json')
    result=read(eval_path)
    for path in (run/'config.yaml',run/'run.json',run/'best_selection.json',eval_path):audit.source(path)
    wanted,_=load_config(resolve(state['configs'][arm]))
    audit.source(resolve(state['configs'][arm]))
    mismatches=[key for key,value in wanted.items() if normalize(cfg.get(key))!=normalize(value)]
    audit.check(arm+':preregistered_config',not mismatches,mismatched_keys=mismatches)
    audit.check(arm+':evaluation_config',all(normalize(result['config'].get(k))==normalize(v) for k,v in cfg.items()))
    for key in ('seed','dataset_id','train_samples_sha256','val_samples_sha256'):
        audit.check(arm+':planned_'+key,cfg.get(key)==planned.get(key))
    audit.check(arm+':locked_thresholds',all(cfg.get(k)==.7 for k in ('global_threshold','local_threshold','localization_threshold')))
    audit.check(arm+':fixed_mode',cfg.get('mode')=='frozen' and cfg.get('tracker_type')=='association' and cfg.get('table_fusion',{}).get('mode')==('none' if arm=='F0' else arm))
    epochs,sampling=collect_epochs(run,planned,audit)
    checkpoints={};states={}
    for name in ('best','last'):
        path=run/f'checkpoints/{name}.pt';digest=audit.source(path)
        current=torch.load(path,map_location='cpu',weights_only=False);states[name]=current
        checkpoints[name]=dict(path=str(path),sha256=digest,epoch=current['epoch'])
        audit.check(arm+':'+name+'_config',normalize(current['config'])==normalize(cfg))
        for stage in ('global','local'):
            source=resolve(cfg[stage+'_checkpoint']);source_sha=audit.source(source)
            audit.check(arm+':'+name+'_'+stage+'_locked_sha',source_sha==locked[stage]==cfg.get(stage+'_sha256'))
            if str(source) not in model_cache:model_cache[str(source)]=torch.load(source,map_location='cpu',weights_only=False)
            audit.check(arm+':'+name+'_'+stage+'_frozen_tensors',frozen_weights(current,model_cache[str(source)],stage+'_model.'))
        if cfg.get('table_frontend'):
            source=resolve(cfg['table_frontend']['checkpoint']);source_sha=audit.source(source)
            if str(source) not in model_cache:model_cache[str(source)]=torch.load(source,map_location='cpu',weights_only=False)
            audit.check(arm+':'+name+'_table_frozen_tensors',frozen_weights(current,model_cache[str(source)],'table_encoder.'))
            native_meta=source.parent/'metrics.json';audit.source(native_meta)
            audit.check(arm+':'+name+'_table_locked_sha',read(native_meta).get('checkpoint_sha256')==source_sha)
    best=states['best'];last=states['last'];last_epoch=max(epochs)
    audit.check(arm+':finished_training',runtime.get('status')=='trained_pending_eval' and runtime.get('epochs_completed')==last_epoch==last['epoch'])
    audit.check(arm+':best_epoch',best['epoch']==selection['epoch']==result.get('epoch') and best['epoch'] in epochs)
    audit.check(arm+':best_sha_matches_evaluation',checkpoints['best']['sha256']==result.get('checkpoint_sha256'))
    thresholds=best.get('thresholds')
    audit.check(arm+':locked_event_threshold',thresholds==selection.get('thresholds') and thresholds[0]==result['metrics']['threshold'] and result.get('threshold_sweep') is None)
    # Select by the existing epoch scores only; do not rerun a threshold search.
    selected=min(epochs.values(),key=lambda row:(-row['continuous']['macro_f1'],row['continuous']['landing']['fp'],row['epoch']))
    audit.check(arm+':best_selection_rule',selected['epoch']==best['epoch'])
    with torch.random.fork_rng(devices=[]):head=build_event_head(cfg)
    parameter_count=sum(p.numel() for p in head.parameters())
    table_sha=audit.source(resolve(cfg['table_frontend']['checkpoint'])) if cfg.get('table_frontend') else None
    return dict(config=cfg,result=result,selection=selection,sampling=sampling,checkpoints=checkpoints,
                training=dict(epochs_completed=last_epoch,best_epoch=best['epoch'],stop_reason=runtime.get('stop_reason'),
                              head_parameters=parameter_count,thresholds=thresholds,table_sha256=table_sha))


def link(path, output):
    return os.path.relpath(Path(path),Path(output)).replace(os.sep,'/')


def write_reports(output, analysis):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=analysis['comparison'];fields=list(rows[0])
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    table=['| Arm | 宏F1 | TP/FP/FN | 侧/斜/正F1 | best/轮数 | 阈值 | FPS | 追回旧FN/丢失旧TP | 缓存/在线 |','|---|---:|---|---|---|---:|---:|---|---|']
    for row in rows:
        table.append(f"| {row['arm']} | {row['macro_f1']:.4%} | {row['tp']}/{row['fp']}/{row['fn']} | {row['side_f1']:.2%}/{row['diagonal_f1']:.2%}/{row['front_f1']:.2%} | {row['best_epoch']}/{row['epochs_completed']} | {row['bounce_threshold']} | {row['e2e_fps']:.2f} | {row['recovered_old_fn']}/{row['lost_old_tp']} | {row['cache_online_status']} |")
    lines=['# 新B定位器三臂内部验证对比','',f"状态：{analysis['status']}；在线验证候选：{analysis['internal_online_candidate'] or '完整性检查未通过，暂不选型'}。",'',*table,'','三视角各占宏F1的1/3；历史主基线为79.56%，使用旧定位器和旧桌框，不能据差值归因于定位器单一改动。','']
    lines += ['## OpenTT game5稀疏分类验证', '', '这些指标来自所选epoch的稀疏锚点；不是完整视频落点F1。Bounce使用PP选出的阈值，Net固定0.5，未使用test调参。', '', '| Arm | Bounce F1 | Net F1 | 分类宏F1 |', '|---|---:|---:|---:|']
    for arm, data in analysis['arms'].items():
        sparse = data.get('sparse_validation') or {}
        def percentage(value):
            return '—' if value is None else f'{value:.2%}'
        lines.append(f"| {arm} | {percentage(sparse.get('bounce',{}).get('f1'))} | {percentage(sparse.get('net',{}).get('f1'))} | {percentage(sparse.get('macro_f1'))} |")
    lines.append('')
    for arm,data in analysis['arms'].items():
        lines.extend([f"## {arm}",'',f"[完整错误入口]({link(data['errors']['index'],output)}) · [在线指标]({link(data['evaluation'],output)}) · [best]({link(data['checkpoints']['best']['path'],output)}) · [last]({link(data['checkpoints']['last']['path'],output)})",'',f"best SHA256：`{data['checkpoints']['best']['sha256']}`；last SHA256：`{data['checkpoints']['last']['sha256']}`。",'',f"缓存/在线：{data['parity']['status']}，概率最大绝对误差{data['parity']['probability_max_abs']:.6g}，球xy最大差{data['parity']['ball_xy_max_abs_px']:.6g}px，原始阈值决策差{data['parity']['totals'].get('bounce_threshold_decision_differences',0)}帧，事件峰决策差{len(data['parity']['event_decision_differences'])}视频，匹配GT差{len(data['parity']['matched_GT_differences'])}视频。",'', '全部40条叠图：'+', '.join(f"[{vid}]({link(path,output)})" for vid,path in sorted(data['errors']['overlays'].items())), ''])
    lines+=['## GT配对反思','','| 固定诊断切片 | 事件数 | 历史TP | F0 TP/FN | geometry TP/FN | concat TP/FN |','|---|---:|---:|---|---|---|']
    for name,bucket in analysis['paired']['buckets'].items():
        cells=[f"{bucket['arms'][arm]['tp']}/{bucket['arms'][arm]['fn']}" for arm in ARMS]
        lines.append(f"| {name} | {bucket['events']} | {bucket['old_matched']} | "+' | '.join(cells)+' |')
    lines+=['',*analysis['reflections'],'','## 检查与限制','']
    failed=[item for item in analysis['checks'] if not item['passed']]
    lines += [f"- 未通过：{item['name']}。详见analysis.json。" for item in failed] or ['- 协议、采样、checkpoint身份及错误覆盖检查通过。']
    lines += ['- '+text for text in analysis['limitations']]
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    concise=['# 本轮训练结果','',f"状态：{analysis['status']}。候选：{analysis['internal_online_candidate'] or '待完整性复核'}。",'',*table,'',*analysis['reflections'],'','完整比较、全部错误视频入口与权重见[comparison.md](comparison.md)；逐事件配对、每帧差异汇总及来源哈希见[analysis.json](analysis.json)。','', f"GT仅用于诊断；本报告只有seed{analysis['seed']}。concat改变视觉分支及模型容量，历史基线不是定位器单变量对照。缓存数值误差小也不代表阈值/NMS决策一致；有差异的结果保留并单列。未使用public/test选型。"]
    (output/'summary.md').write_text('\n'.join(concise)+'\n')
    temporary=output/'analysis.json.tmp';temporary.write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(output/'analysis.json')


def report(task, output=None, historical=None, probability_atol=2e-4):
    task=resolve(task);output=resolve(output) if output else task
    audit=Audit();task_path=task/'task.json';state=read(task_path);audit.source(task_path);audit.source(Path(__file__))
    paths={arm:evaluation_path(state,arm) for arm in ARMS}  # Fail early until all arms finish.
    planned_path=task/'planned_schedule.json';planned=read(planned_path);audit.source(planned_path)
    diag_path=task/'pretraining_diagnostics.json';diagnostics=read(diag_path);audit.source(diag_path)
    historical=resolve(historical or 'result/tcn_frames_lr0.0003_d0.1/metrics.json')
    historical_result=read(historical);audit.source(historical)
    snapshots=diagnostics['sources_sha256']
    if snapshots.get(str(historical)) is None:
        relative=os.path.relpath(historical,PROJECT)
        expected_historical=snapshots.get(relative)
    else:expected_historical=snapshots[str(historical)]
    audit.check('historical_snapshot_sha',expected_historical==audit.source(historical))
    locks=[resolve(path) for path in snapshots if path.endswith('/selection_lock.json')]
    if len(locks)!=1:raise ReportError('diagnostic snapshot must identify exactly one locator selection lock')
    locked=read(locks[0])['weights']['B'];audit.source(locks[0])
    expected_lock=snapshots.get(str(locks[0]),snapshots.get(os.path.relpath(locks[0],PROJECT)))
    audit.check('locator_lock_snapshot_sha',expected_lock==audit.source(locks[0]))
    arms={};models={};configs=[];results={}
    for arm in ARMS:
        run=resolve(state['runs'][arm]);loaded=load_arm(arm,run,paths[arm],state,planned,locked,audit,models)
        cfg=loaded['config'];result=loaded['result'];configs.append(cfg);results[arm]=result['metrics']
        ann_path=resolve(cfg['pp_annotations']);audit.source(ann_path)
        expected_ann=snapshots.get(str(ann_path),snapshots.get(os.path.relpath(ann_path,PROJECT)))
        audit.check(arm+':annotation_snapshot_sha',expected_ann==audit.source(ann_path))
        document=read(ann_path);annotations={v['video_id']:v for v in document['videos'] if v['split']=='val'}
        audit.check(arm+':annotation_identity',result['annotation_id']==identity(document))
        audit.check(arm+':40_validation_videos',len(annotations)==40 and set(result['metrics']['per_video'])==set(annotations))
        protocol=identity(dict(annotation=result['annotation_id'],nms_sec=cfg['nms_sec'],frame_tolerance=1,pixel_tolerance=50))
        audit.check(arm+':protocol_id',result['protocol_id']==protocol and result['metrics']['protocol']==historical_result['metrics']['protocol'])
        cached_path=run/'predictions/best_pp_continuous.json';audit.source(cached_path)
        cached={vid.removeprefix('pp_'):rows for vid,rows in read(cached_path).items()}
        if len(cached)!=40:raise ReportError(f'{arm}: training best predictions must contain40 PP videos')
        online={}
        for vid in annotations:
            path=paths[arm].parent/f'{vid}_predictions.jsonl';audit.source(path);online[vid]=jsonl(path)
        parity,rescored=compare_cached_online(cached,online,annotations,result['metrics']['threshold'],cfg['nms_sec'],probability_atol)
        audit.check(arm+':online_metrics_recomputed',metric_signature(rescored)==metric_signature(result['metrics']))
        audit.check(arm+':cached_selection_recomputed',metric_signature(loaded['selection']['continuous'])==parity['cache_metrics'])
        error_report=audit_errors(paths[arm].parent,result['metrics'],online,audit,arm)
        arms[arm]=dict(evaluation=str(paths[arm]),training=loaded['training'],checkpoints=loaded['checkpoints'],
                       sampling=loaded['sampling'],parity=parity,errors=error_report,
                       scores=metric_signature(result['metrics']),by_view=result['metrics']['by_view'],
                       sparse_validation=loaded['selection'].get('val'),
                       performance=dict(e2e_fps=result['e2e_fps'],hardware=result.get('benchmark_hardware'),
                                        backend=result.get('backend'),precision=result.get('precision'),
                                        scope='Complete source decode, ball/table encoders, event head and JSON serialization; excludes initialization/disk writes. Offline inference includes6+1 future frames.'))
    audit.check('three_arm_fixed_protocol',all({key:configs[0].get(key) for key in PROTOCOL_KEYS}=={key:cfg.get(key) for key in PROTOCOL_KEYS} for cfg in configs[1:]))
    audit.check('table_frontend_fixed_geometry_concat',configs[1].get('table_frontend')==configs[2].get('table_frontend'))
    paired=paired_events(historical_result['metrics'],results,diagnostics)
    audit.check('preregistered_diagnostic_slice_sizes',paired['buckets']['six_exact_frame_repairs']['events']==6 and paired['buckets']['no_roi_entire_13window']['events']==40)
    valid=all(item['passed'] for item in audit.checks)
    parity_discrepancy=any(not data['parity']['decisions_equal'] for data in arms.values())
    status='audit_failed' if not valid else 'evaluated_pending_parity_review' if parity_discrepancy else 'analyzed_internal_single_seed'
    candidate=max(ARMS,key=lambda arm:(results[arm]['macro_f1'],-results[arm]['landing']['fp'],-ARMS.index(arm))) if valid else None
    rows=[]
    for arm in ARMS:
        metric=results[arm];data=arms[arm];training=data['training'];cfg=configs[ARMS.index(arm)]
        row=dict(arm=arm,seed=cfg['seed'],model='13-frame frozen TCN',dataset_id=cfg['dataset_id'],split_id=cfg['split_id'],
                 global_sha256=cfg['global_sha256'],local_sha256=cfg['local_sha256'],table_sha256=training['table_sha256'],
                 global_threshold=cfg['global_threshold'],local_threshold=cfg['local_threshold'],
                 optimizer=cfg['optimizer'],lr=cfg['lr'],weight_decay=cfg['weight_decay'],batch=cfg['effective_batch'],
                 epochs_completed=training['epochs_completed'],best_epoch=training['best_epoch'],head_parameters=training['head_parameters'],
                 best_sha256=data['checkpoints']['best']['sha256'],last_sha256=data['checkpoints']['last']['sha256'],
                 bounce_threshold=metric['threshold'],macro_f1=metric['macro_f1'],**{key:metric['landing'][key] for key in ('tp','fp','fn')})
        for view in VIEWS:
            row.update({view+'_'+key:metric['by_view'][view][key] for key in ('precision','recall','f1','tp','fp','fn')})
        row.update(e2e_fps=data['performance']['e2e_fps'],hardware=data['performance']['hardware'],backend=data['performance']['backend'],precision=data['performance']['precision'],
                   recovered_old_fn=paired['buckets']['old45FN']['arms'][arm]['tp'],lost_old_tp=paired['buckets']['old144TP']['arms'][arm]['fn'],
                   cache_online_status=data['parity']['status'],status=status,metrics=data['evaluation'],errors=data['errors']['index'])
        sparse=data.get('sparse_validation') or {}
        row.update(opentt_val_sparse_macro_f1=sparse.get('macro_f1'),
                   opentt_val_sparse_bounce_f1=sparse.get('bounce',{}).get('f1'),
                   opentt_val_sparse_net_f1=sparse.get('net',{}).get('f1'))
        rows.append(row)
    reflections=[]
    for arm in ARMS:
        recovered=paired['buckets']['old45FN']['arms'][arm]['tp'];lost=paired['buckets']['old144TP']['arms'][arm]['fn']
        no_roi=paired['buckets']['no_roi_entire_13window']['arms'][arm]
        reflections.append(f"- {arm}：相对历史旧45FN追回{recovered}个，旧144TP丢失{lost}个；40个完整窗无ROI事件中TP/FN={no_roi['tp']}/{no_roi['fn']}。")
    if candidate:
        delta=results[candidate]['macro_f1']-historical_result['metrics']['macro_f1']
        reflections.append(f"- 本轮在线验证候选{candidate}，相对历史宏F1变化{delta*100:+.2f}个百分点；此差值同时包含前端、桌框和事件配方变化。")
    if parity_discrepancy:reflections.append('- 缓存与在线存在阈值、NMS事件峰或匹配GT决策差异，不能声称完全一致；选型仍需复核训练验证与在线实现。')
    limitations=[
        'GT仅用于既有预测的评分与固定错误切片，不是训练/推理输入；±1可用球不等于可部署召回上限。',
        f"本报告仅seed{configs[0]['seed']}、内部验证选型；跨种子稳定性须另行汇总，没有基于public/test调参或新增外测。",
        'concat同时加入视觉特征及额外投影容量；geometry/concat差异不能单独归因于全图context。',
        '历史79.56%使用旧球定位器、旧桌框及历史事件设置，不是定位器单变量消融。',
        '提前停止使各臂实际轮数可能不同；已核对每个完成epoch与预定采样hash，不能据此宣称总曝光量相等。',
        '数值容差只用于描述概率差异；即使误差在容差内，只要阈值/NMS/匹配决策改变就单列待复核。',
        'FPS是记录的本机完整视频离线速度；包含解码至序列化、排除初始化/写盘，未来6+1帧等待另计，不代表赛事硬件速度。',
    ]
    analysis=dict(schema='B_event_study_report_v1',seed=configs[0]['seed'],created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  task=str(task),status=status,internal_online_candidate=candidate,adoption='not_automatically_adopted',
                  comparison=rows,arms=arms,paired=paired,checks=audit.checks,reflections=reflections,
                  limitations=limitations,sources_sha256=audit.sources)
    write_reports(output,analysis)
    return analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,help='Defaults to task directory; writes only the four report files.')
    parser.add_argument('--historical',type=Path,default=PROJECT/'result/tcn_frames_lr0.0003_d0.1/metrics.json')
    args=parser.parse_args()
    try:
        result=report(args.task,args.output,args.historical)
    except (ReportError,FileNotFoundError,KeyError) as error:
        parser.exit(2,f'Report inputs incomplete or invalid: {error}\n')
    print(json.dumps(dict(status=result['status'],candidate=result['internal_online_candidate'],output=str(args.output or args.task))))
    if result['status']=='audit_failed':raise SystemExit(2)


if __name__=='__main__':main()
