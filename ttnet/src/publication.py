"""Publish self-contained final artifacts separately from execution records."""
from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

from ttnet.common import ROOT, identity, now, save, sha, output_directory

ARTIFACTS = {'checkpoints': 'weights', 'evaluations': 'eval',
             'visualizations': 'error_samples', 'predictions': 'predictions',
             'exports': 'exports', 'feature_cache': 'feature_cache'}
TERMINAL = {'completed', 'trained_pending_eval', 'evaluated_pending_export',
            'failed', 'interrupted', 'paused_by_user'}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def artifact_path(value):
    for old, new in ARTIFACTS.items():
        value = value.replace(old + '/', new + '/')
    return value


def source_checkpoint(config, root):
    value = config.get('global_checkpoint')
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    # Historical configs were launched from the project root.
    return next((p for p in [root.parent / path, root / path] if p.is_file()), root.parent / path)


def publish_run(run, root=ROOT):
    """Snapshot terminal non-smoke runs; mark missing or mismatched evidence partial."""
    run, root = Path(run).resolve(), Path(root).resolve()
    state, config = read(run/'run.json'), read(run/'config.yaml')
    if state.get('status') not in TERMINAL or config.get('smoke') or config.get('stage') not in {'global', 'local'}:
        return None
    output = output_directory(root)
    dest = output/run.name
    files = [p for name in ARTIFACTS for p in (run/name).rglob('*') if p.is_file() and '.tmp' not in p.name]
    files += [p for p in [run/'config.yaml', run/'run.json', run/'environment.json'] if p.is_file()]
    files += sorted((run/'logs').glob('attempt_*/epochs.jsonl'))
    dependency = source_checkpoint(config, root) if config['stage']=='local' else None
    if dependency and dependency.is_file():
        files.append(dependency)
    fingerprint = identity(dict(publisher_sha256=sha(Path(__file__)),
                                files=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(files)]))
    old = read(dest/'manifest.json')
    if old.get('source_fingerprint') == fingerprint:
        return dest
    output.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{run.name}-', dir=output))
    for source, target in ARTIFACTS.items():
        if (run/source).is_dir():
            shutil.copytree(run/source, temp/target, ignore=shutil.ignore_patterns('*.tmp', '*.tmp.*'))
    for name in ['config.yaml', 'run.json', 'environment.json']:
        if (run/name).is_file():
            shutil.copy2(run/name, temp/name)
    if dependency and dependency.is_file():
        (temp/'weights').mkdir(exist_ok=True)
        shutil.copy2(dependency, temp/'weights/global_best.pt')
    # All viewer references and error-index paths stay inside the published bundle.
    for directory in ['eval', 'error_samples']:
        for path in (temp/directory).rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.html', '.md'}:
                path.write_text(artifact_path(path.read_text()))
    records = []
    for path in sorted((run/'logs').glob('attempt_*/epochs.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    epochs = {r['epoch']: r for r in records}
    (temp/'stats').mkdir()
    save(temp/'stats/training.json', [epochs[e] for e in sorted(epochs)])
    metric = read(temp/'eval/best_val/metrics.json')
    benchmark = read(temp/'eval/benchmark/metrics.json')
    exports = read(temp/'exports/manifest.json')
    feature_identity = read(temp/'feature_cache/best_val/identity.json')
    missing = []
    for path in ['weights/best.pt', 'weights/last.pt', 'eval/best_val/metrics.json',
                 'eval/best_val/per_video.json', 'eval/best_val/errors.json',
                 'eval/best_val/index.html', 'predictions/best_val.jsonl',
                 'eval/benchmark/metrics.json', 'exports/manifest.json', 'feature_cache/best_val/identity.json']:
        if not (temp/path).is_file():
            missing.append(path)
    if not metric.get('center'):
        missing.append('评估未包含球心指标')
    if not benchmark.get('e2e_fps'):
        missing.append('测速未包含 E2E FPS')
    expected_key = 'local_checkpoint_sha256' if config['stage']=='local' else 'global_checkpoint_sha256'
    if metric and (not (temp/'weights/best.pt').is_file() or sha(temp/'weights/best.pt') != metric.get(expected_key)):
        missing.append('评估 checkpoint 与 best.pt 不一致')
    if config['stage']=='local' and (not (temp/'weights/global_best.pt').is_file() or sha(temp/'weights/global_best.pt') != metric.get('global_checkpoint_sha256')):
        missing.append('组合定位的 Global 权重缺失或不一致')
    if benchmark and metric:
        if any(benchmark.get(k) != metric.get(m) for k,m in [('global_sha256','global_checkpoint_sha256'), ('local_sha256','local_checkpoint_sha256')]):
            missing.append('测速 checkpoint 与评估不一致')
    if feature_identity and metric:
        if any(feature_identity.get(k) != metric.get(m) for k,m in [('global_sha256','global_checkpoint_sha256'), ('local_sha256','local_checkpoint_sha256')]):
            missing.append('特征缓存 checkpoint 与评估不一致')
    if exports:
        required = {'global', 'local'} if config['stage']=='local' else {'global'}
        if not required.issubset({model.get('stage') for model in exports.get('models', [])}):
            missing.append('ONNX 缺少阶段模型')
        for model in exports.get('models', []):
            onnx = temp/'exports'/f'{model["stage"]}.onnx'
            key = model['stage']+'_checkpoint_sha256'
            if not model.get('parity', {}).get('passed') or model.get('checkpoint_sha256') != metric.get(key) or not onnx.is_file() or sha(onnx) != model.get('onnx_sha256'):
                missing.append(f'{model["stage"]} ONNX 文件、权重身份或对齐验证缺失')
        if not exports.get('models'):
            missing.append('ONNX 模型清单为空')
    errors = json.loads((temp/'eval/best_val/errors.json').read_text()) if (temp/'eval/best_val/errors.json').is_file() else []
    for error in errors:
        if not (temp/error['visualization']).is_file():
            missing.append('错误图片缺失: '+error['visualization'])
    if not (temp/'eval/best_val/errors.html').exists() and (temp/'eval/best_val/errors.json').exists():
        import html
        tiles = ''.join(f'<figure><a href="../../{html.escape(e["visualization"])}"><img loading="lazy" src="../../{html.escape(e["visualization"])}"></a><figcaption>{html.escape(e["video_id"])} / {e["frame_id"]}: {html.escape(e["kind"])}</figcaption></figure>' for e in errors)
        (temp/'eval/best_val/errors.html').write_text('<meta charset="utf-8"><style>body{font:16px sans-serif}figure{display:inline-block;width:400px}img{width:100%}</style><h1>全部评分错误：绿色 GT，红色 Pred，蓝色 ROI</h1>'+tiles)
    status = 'completed' if state.get('status') in {'completed', 'evaluated_pending_export'} and not missing else 'partial'
    statistics = dict(stage=config['stage'], status=status, source_status=state.get('status'),
                      dataset_id=config.get('dataset_id'), split_id=config.get('split_id'),
                      best_epoch=state.get('best_epoch'), epochs_completed=state.get('epochs_completed'),
                      center=metric.get('center'), global_center=metric.get('global_center'),
                      roi_coverage=metric.get('roi_coverage'), error_count=len(errors) if metric else None,
                      errors_by_type=dict(Counter(e['kind'] for e in errors)),
                      e2e_fps=benchmark.get('e2e_fps'), global_forward=benchmark.get('global_forward'),
                      local_forward=benchmark.get('local_forward'), peak_cuda_bytes=benchmark.get('peak_cuda_bytes'),
                      global_compute=benchmark.get('global_compute'), local_compute=benchmark.get('local_compute'),
                      hardware=benchmark.get('gpu'), precision=benchmark.get('precision'),
                      missing=missing, ap=None, ap_note='球心热图不适用 bbox AP')
    save(temp/'stats/summary.json', statistics)
    weights = {str(p.relative_to(temp)): sha(p) for p in (temp/'weights').glob('*.pt')}
    lines = [f'# {run.name}', '', f'结果状态：**{status}**；运行状态：{state.get("status")}。', '',
             f'阶段：{config["stage"]}；最佳 epoch：{state.get("best_epoch")}；已完成 epoch：{state.get("epochs_completed")}。', '',
             f'数据身份：`{config.get("dataset_id")}`；划分：`{config.get("split_id")}`；train / val 视频：{config.get("train_videos")} / {config.get("val_videos")}。', '',
             f'输入 Global / Local：{config.get("global_input")} / {config.get("local_input")}；时序：{config.get("temporal_mode", "single")}；batch：{config.get("effective_batch")}；lr：{config.get("lr")}。', '',
             '| 容差 | Precision | Recall | F1 | TP | FP | FN | 误差中位数 / P90 px |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for tolerance, values in (metric.get('center') or {}).items():
        lines.append(f'| {tolerance} px | {values["precision"]:.4f} | {values["recall"]:.4f} | {values["f1"]:.4f} | {values["tp"]} | {values["fp"]} | {values["fn"]} | {values.get("error_median_px")} / {values.get("error_p90_px")} |')
    lines += ['', f'完整视频 E2E FPS：{benchmark.get("e2e_fps", "—")}；GPU：{benchmark.get("gpu", "—")}；精度：{benchmark.get("precision", "—")}。', '',
              '本机 PyTorch 速度包含解码、预处理、定位与序列化，不含磁盘写入；离线窗口推理。验证集同时用于模型选择，成绩不代表独立测试集。未知标签不计入评分；热图模型 AP 留空。', '',
              f'错误数：{statistics["error_count"]}；类型统计：{statistics["errors_by_type"]}。', '',
              '## 最终文件', '']
    for path, label in [('weights/best.pt','最佳权重'), ('weights/last.pt','最后权重（续训状态）'),
                        ('weights/global_best.pt','组合定位配套 Global 权重'), ('eval/best_val/metrics.json','完整评估'),
                        ('eval/best_val/per_video.json','逐视频统计'), ('eval/best_val/index.html','Pred / Label 预览'),
                        ('eval/best_val/errors.html','全部错误图册和已有慢放片段'), ('eval/best_val/errors.json','错误索引'),
                        ('eval/best_val/ERROR_REVIEW.md','错误分析'), ('eval/training_curve.svg','训练曲线'),
                        ('eval/benchmark/metrics.json','测速原始数据、延迟和计算量'), ('exports/manifest.json','ONNX 与对齐记录'),
                        ('stats/summary.json','总统计'), ('stats/training.json','每轮 loss、验证、时间与显存'),
                        ('config.yaml','实际配置'), ('manifest.json','权重哈希与产物清单')]:
        if (temp/path).is_file() or path=='manifest.json':
            lines.append(f'- [{label}]({path})')
    if missing:
        lines += ['', '缺项 / 身份检查：', ''] + ['- '+v for v in missing]
    (temp/'summary.md').write_text('\n'.join(lines)+'\n')
    inventory = {str(p.relative_to(temp)): p.stat().st_size for p in temp.rglob('*') if p.is_file()}
    save(temp/'manifest.json', dict(schema=1, result_id=run.name, published=now(), stage=config['stage'],
                                   status=status, source_status=state.get('status'), source_run=str(run),
                                   source_fingerprint=fingerprint, weights_sha256=weights,
                                   missing=missing, artifacts_bytes=inventory))
    if dest.exists():
        if not old:
            raise RuntimeError(f'拒绝覆盖未由结果发布器创建的目录: {dest}')
        history = output/'.history'
        history.mkdir(exist_ok=True)
        dest.rename(history/f'{run.name}-{old["source_fingerprint"][:12]}-{os.getpid()}')
    temp.rename(dest)
    return dest


def write_index(root=ROOT):
    root = Path(root).resolve()
    output = output_directory(root)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(output.glob('*/manifest.json')):
        manifest, stats = read(path), read(path.parent/'stats/summary.json')
        row = dict(result_id=path.parent.name, stage=manifest['stage'], status=manifest['status'],
                   source_status=manifest['source_status'], best_epoch=stats.get('best_epoch'),
                   e2e_fps=stats.get('e2e_fps'), dataset_id=stats.get('dataset_id'), split_id=stats.get('split_id'),
                   summary=path.parent.name+'/summary.md')
        for tolerance in ['10', '50']:
            for name in ['precision', 'recall', 'f1']:
                row[f'{name}_{tolerance}px'] = ((stats.get('center') or {}).get(tolerance) or {}).get(name)
        rows.append(row)
    save(output/'index.json', rows)
    save(output/'latest.json', {stage: next((r['result_id'] for r in reversed(rows) if r['stage']==stage and r['status']=='completed'), None) for stage in ['global','local']})
    with (output/'comparison.csv').open('w', newline='') as file:
        fields = list(rows[0]) if rows else ['result_id', 'stage', 'status']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    def percent(value):
        return f'{value:.2%}' if value is not None else '—'
    lines = ['# TTNet 最终结果', '', '每个目录汇总一次训练的最终产物。completed 表示已核对权重、评估、错误图、导出和测速；partial 表示缺项或尚未完成。', '',
             '| 任务 | 阶段 | 状态 | Recall 10px | F1 10px | F1 50px | E2E FPS |', '|---|---|---|---:|---:|---:|---:|']
    for row in reversed(rows):
        fps = f'{row["e2e_fps"]:.2f}' if row['e2e_fps'] is not None else '—'
        lines.append(f'| [{row["result_id"]}]({row["summary"]}) | {row["stage"]} | {row["status"]} | {percent(row["recall_10px"])} | {percent(row["f1_10px"])} | {percent(row["f1_50px"])} | {fps} |')
    lines += ['', '[机器可读统计](index.json) · [对比 CSV](comparison.csv) · [最新完整结果](latest.json)', '',
              'weights/ 保存 best、last 和 Local 所依赖的 Global；eval/ 保存评估和测速；error_samples/ 保存叠加图与片段；stats/ 保存汇总及训练统计。所有结果页面的产物链接均在 output 内。', '',
              '同一任务续训后更新快照时，旧版本保存在 `.history/`；runs 保留原始执行记录。', '']
    save_text(output/'README.md', '\n'.join(lines))
    return output/'README.md'


def save_text(path, value):
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(value)
    temp.replace(path)
