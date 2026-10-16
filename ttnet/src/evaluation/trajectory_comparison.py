"""Paired full-video A/B locator diagnostic, before training either event head."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ttnet.common import DATA, PROJECT, identity, jsonl, save, save_jsonl, sha
from ttnet.configuration import load_config
from ttnet.evaluation.ball_corpus import BallCorpus
from ttnet.evaluation.candidate_audit import audit, in_roi
from ttnet.evaluation.metrics import aggregate


def accepted(record):
    prediction = record.get('prediction')
    return prediction if prediction and prediction['valid'] else None


def hit(row, record, tolerance):
    prediction = accepted(record)
    return bool(row['label_status'] == 'positive' and prediction and
                np.linalg.norm(np.asarray(row['ball_centers'][0]) - [prediction['x'], prediction['y']]) <= tolerance)


def quantiles(values):
    return dict(count=len(values), median=float(np.median(values)) if values else None,
                p90=float(np.percentile(values, 90)) if values else None, max=max(values, default=0))


def measure(rows, records, bounce_ids):
    """GT is consulted only here, after chronological, GT-free inference."""
    if len(rows) != len(records):raise ValueError('row/prediction count mismatch')
    predictions = [dict(r['prediction'], crop_transform=r.get('crop_transform')) if r.get('prediction') else None for r in records]
    result = dict(localization={str(t): aggregate(rows, predictions, t) for t in (10, 50)})
    positives = roi_any = roi_selected = 0
    runs = {10: [], 50: []}; current = {10: 0, 50: 0}; raw_runs = []; raw_run = 0
    previous = None; jump_pairs = []; jump_errors = []; near_rows = []; near_predictions = []
    for row, record, prediction in zip(rows, records, predictions):
        key = (row['video_id'], row['frame_id'])
        if key != (record['video_id'], record['frame_id']):raise ValueError('prediction alignment mismatch')
        contiguous = previous is not None and row['video_id'] == previous[0]['video_id'] and row['frame_id'] == previous[0]['frame_id'] + 1
        positive = row['label_status'] == 'positive'
        for tolerance in runs:
            missed = positive and not hit(row, record, tolerance)
            if current[tolerance] and (not contiguous or not missed):
                runs[tolerance].append(current[tolerance]); current[tolerance] = 0
            if missed:current[tolerance] += 1
        raw_missed = positive and accepted(record) is None
        if raw_run and (not contiguous or not raw_missed):raw_runs.append(raw_run); raw_run = 0
        if raw_missed:raw_run += 1
        if positive:
            positives += 1; xy = row['ball_centers'][0]
            roi_any += any(in_roi(crop, xy) for crop in record.get('roi_transforms', []))
            roi_selected += in_roi(record.get('crop_transform'), xy)
        if contiguous and positive and previous[0]['label_status'] == 'positive' and accepted(record) and accepted(previous[1]):
            prior_row, prior_record = previous; dt = row['timestamp_sec'] - prior_row['timestamp_sec']
            if 0 < dt <= 1.5 / row['fps']:
                p, q = accepted(record), accepted(prior_record)
                residual = (np.array([p['x']-q['x'], p['y']-q['y']]) -
                            (np.asarray(row['ball_centers'][0])-prior_row['ball_centers'][0]))
                distance = float(np.linalg.norm(residual)); jump_pairs.append(distance)
                if distance > 50:jump_errors.append(dict(video_id=row['video_id'], frame_id=row['frame_id'], residual_px=distance))
        if any(abs(row['frame_id']-fid) <= 1 for fid in bounce_ids.get(row['video_id'], [])):
            near_rows.append(row); near_predictions.append(prediction)
        previous = (row, record)
    for tolerance in runs:
        if current[tolerance]:runs[tolerance].append(current[tolerance])
    if raw_run:raw_runs.append(raw_run)
    result.update(positive_frames=positives, roi_any_coverage=roi_any/max(1, positives),
        roi_selected_coverage=roi_selected/max(1, positives),
        consecutive_missed={str(k): quantiles(v) for k, v in runs.items()},
        consecutive_no_observation=quantiles(raw_runs),
        erroneous_jumps=dict(count=len(jump_errors), accepted_adjacent_pairs=len(jump_pairs),
            rate=len(jump_errors)/max(1, len(jump_pairs)), residual_px=quantiles(jump_pairs),
            definition='GT-relative displacement error >50 original pixels between consecutive accepted positive frames; real positive dt <=1.5/frame_rate'),
        bounce_neighborhood=dict(frames=len(near_rows), radius_frames=1,
            localization={str(t):aggregate(near_rows, near_predictions, t) for t in (10, 50)}))
    return result, jump_errors


def compare_outputs(output, dataset, annotations):
    dest = Path(output); corpus = BallCorpus(dataset)
    annotation = json.loads(Path(annotations).read_text())
    if annotation['ball_dataset_id'] != corpus.identity or annotation['ball_split_hashes'] != corpus.metadata['split_hashes']:
        raise ValueError('annotation/localization split identity mismatch')
    metadata = {v['video_id']: v for v in annotation['videos']}
    bounce_ids = {vid: [e['frame_id'] for e in v['events'] if e['kind'] == 'bounce'] for vid, v in metadata.items()}
    rows = []; observations = {k: [] for k in 'AB'}; paired = []; groups = defaultdict(list)
    summaries = {k:json.loads((dest/k/'summary.json').read_text()) for k in 'AB'}
    for key in ('dataset_id', 'global_sha256', 'local_sha256'):
        if summaries['A'][key] != summaries['B'][key]:raise ValueError('unpaired '+key)
    configs = {k:dict(v['config']) for k, v in summaries.items()}
    for cfg in configs.values():
        for key in ('tracker_type', 'association', 'scheme'):cfg.pop(key, None)
    if configs['A'] != configs['B']:raise ValueError('A/B differ beyond tracker configuration')
    for vid in corpus.split['val']:
        video_rows = corpus.rows[vid]; start = len(rows); rows.extend(video_rows)
        video_records = {k:jsonl(dest/k/f'{vid}_predictions.jsonl') for k in 'AB'}
        for scheme in 'AB':observations[scheme].extend(video_records[scheme])
        resolution = video_rows[0]['resolution']; ids = list(range(start, len(rows)))
        groups['view:'+metadata[vid]['view']].extend(ids)
        groups[f"resolution:{resolution['width']}x{resolution['height']}"].extend(ids)
        groups['video:'+vid].extend(ids)
        for row, a, b in zip(video_rows, video_records['A'], video_records['B']):
            if row['label_status'] not in ('positive', 'negative'):continue
            for tolerance in (10, 50):
                correct_a = hit(row, a, tolerance) if row['label_status'] == 'positive' else accepted(a) is None
                correct_b = hit(row, b, tolerance) if row['label_status'] == 'positive' else accepted(b) is None
                if correct_a != correct_b:
                    paired.append(dict(video_id=vid, frame_id=row['frame_id'], tolerance_px=tolerance,
                        change='recovered' if correct_b else 'regressed',
                        A_overlay=f'A/{vid}_overlay.mp4', B_overlay=f'B/{vid}_overlay.mp4'))
    results = {}; stratified = {k:{} for k in 'AB'}
    for scheme in 'AB':
        results[scheme], jumps = measure(rows, observations[scheme], bounce_ids)
        save_jsonl(dest/f'{scheme}_erroneous_jumps.jsonl', jumps)
        for group, ids in groups.items():
            stratified[scheme][group] = measure([rows[i] for i in ids], [observations[scheme][i] for i in ids], bounce_ids)[0]
    changes = {str(t):dict(Counter(r['change'] for r in paired if r['tolerance_px'] == t)) for t in (10,50)}
    f1_a = results['A']['localization']['50']['f1']; f1_b = results['B']['localization']['50']['f1']
    result = dict(status='completed_locator_comparison_only', videos=len(corpus.split['val']), frames=len(rows),
        dataset_id=corpus.identity, split_hashes=corpus.metadata['split_hashes'],
        annotation_sha256=sha(annotations), implementation_sha256=sha(__file__),
        schemes=results, paired_changes=changes, f1_50px_delta=f1_b-f1_a,
        interpretation='Frozen locator diagnostic; no event-head accuracy or final event model selection. Bounce neighborhoods use original ball GT; annotations never enter inference. Camera views are coarse visual groups.',
        candidate_ceiling={k:summaries[k]['candidate'] for k in 'AB'})
    save(dest/'metrics.json', result); save(dest/'stratified.json', stratified); save_jsonl(dest/'paired_changes.jsonl', paired)
    lines = ['# 冻结定位器 A/B 轨迹关联对照', '',
        '同一固定 PP 验证集、定位权重、原图像素口径；只改变轨迹组件。GT 仅评分。', '',
        '| 指标 | A 独立解码 | B 多候选关联 |', '|---|---:|---:|']
    def add(label, a, b, percentage=False):
        fmt = (lambda x:f'{x:.2%}') if percentage else (lambda x:f'{x:.3f}' if isinstance(x, float) else str(x))
        lines.append(f'| {label} | {fmt(a)} | {fmt(b)} |')
    for tolerance in ('10', '50'):
        add(f'球心 F1 ({tolerance}px)', results['A']['localization'][tolerance]['f1'], results['B']['localization'][tolerance]['f1'], True)
    add('任一 Local ROI 覆盖率', results['A']['roi_any_coverage'], results['B']['roi_any_coverage'], True)
    add('选中 Local ROI 覆盖率', results['A']['roi_selected_coverage'], results['B']['roi_selected_coverage'], True)
    add('错误跳转次数（GT 相对位移偏差 >50px）', results['A']['erroneous_jumps']['count'], results['B']['erroneous_jumps']['count'])
    add('最长连续定位漏检（50px，帧）', results['A']['consecutive_missed']['50']['max'], results['B']['consecutive_missed']['50']['max'])
    add('最长连续无有效观测（帧）', results['A']['consecutive_no_observation']['max'], results['B']['consecutive_no_observation']['max'])
    add('触台 ±1 帧球心 F1 (50px)', results['A']['bounce_neighborhood']['localization']['50']['f1'], results['B']['bounce_neighborhood']['localization']['50']['f1'], True)
    lines += ['', '恢复与退化逐帧清单：`paired_changes.jsonl`。视角、分辨率和逐视频统计：`stratified.json`。',
        'A/B 的 `index.html` 提供覆盖全部评分错误的完整叠加视频。完整视频 FPS 单列于 `runtime.json`，不采用含叠加编码的诊断耗时。',
        '', '此结果仅判断定位轨迹组件的收益；事件收益仍需同协议训练 A/B 后评估。']
    (dest/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result


def runtime_check(dest, configs, dataset, device):
    from ttnet.events.engine import EventSystem
    from ttnet.events.inference import predict_video
    sources={r['video_id']:r for r in jsonl(Path(dataset)/'splits/val.jsonl')}
    results={}
    for scheme in 'AB':
        cfg=configs[scheme]; torch.manual_seed(cfg['seed'])
        system=EventSystem(cfg, torch.device(device)); system.configure('frozen', False)
        measurements=[]
        for vid, row in sources.items():
            _, _, speed=predict_video(system, PROJECT/row['video'], vid, target_fps=None, progress=False)
            measurements.append(dict(speed, video_id=vid))
            print(f'runtime {scheme} {vid}: {speed["e2e_fps"]:.1f} FPS', flush=True)
        results[scheme]=dict(videos=measurements, frames=sum(v['frames'] for v in measurements),
            e2e_fps=sum(v['frames'] for v in measurements)/sum(v['elapsed_sec'] for v in measurements))
        save(Path(dest)/'runtime_progress.json', results)
        del system
        if device.startswith('cuda'):torch.cuda.empty_cache()
    save(Path(dest)/'runtime.json', dict(status='completed_runtime_only', hardware=torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU',
        schemes=results, note='Same device/backend, trained frozen locators, same randomly initialized TCN; accuracy is not measured from these event heads. Full original-video decode to JSON, excludes disk writes/init.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--dataset', type=Path, default=DATA)
    p.add_argument('--annotations', type=Path, required=True); p.add_argument('--device', default='cuda')
    p.add_argument('--config-a', type=Path, required=True); p.add_argument('--config-b', type=Path, required=True)
    args=p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    configs={k:load_config(path)[0] for k, path in [('A',args.config_a),('B',args.config_b)]}
    save(args.output/'config.yaml', dict(configs=configs, dataset=str(args.dataset), annotations=str(args.annotations), device=args.device))
    try:
        for scheme in 'AB':
            save(args.output/'status.json', dict(status='running', step=scheme+'_audit'))
            audit(configs[scheme], args.dataset, args.output/scheme, args.device)
        result=compare_outputs(args.output, args.dataset, args.annotations)
        print(json.dumps(dict(f1_50px_delta=result['f1_50px_delta'], paired_changes=result['paired_changes'])), flush=True)
        save(args.output/'status.json', dict(status='running', step='full_video_runtime'))
        runtime_check(args.output, configs, args.dataset, args.device)
        save(args.output/'status.json', dict(status='completed', metric_sha256=sha(args.output/'metrics.json')))
    except BaseException as error:
        save(args.output/'status.json', dict(status='failed', error=repr(error))); raise


if __name__ == '__main__':main()
