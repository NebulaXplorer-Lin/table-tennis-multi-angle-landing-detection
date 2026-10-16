"""Organize promising 20261016_table_ar results into out/results/<dir> by scheme.

Each scheme folder gets: README.md, best.pt (weights), config, training records
(attempts + curve + selection), eval metrics (continuous + external + latency),
and error-overlay videos (continuous index.html + mp4). Multi-seed variants are
stored under seeds/. Evidence-preserving copy; originals are untouched.
"""
import json
import shutil
from pathlib import Path

SRC = Path('out/experiments/20261016_table_ar')
DST = Path('out/results/20261016_table_ar')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def load_results():
    rows = read_json(SRC / 'results.json', [])
    by_name = {r['name']: r for r in rows}
    return by_name


def eval_metrics(run):
    m = read_json(SRC / run / 'evaluations/continuous/metrics.json')
    return m


def external_summary(run):
    m = read_json(SRC / run / 'evaluations/external/metrics.json')
    if not m:
        return None
    mm = m.get('metrics', {})
    out = {}
    if 'ttgames_test' in mm:
        t = mm['ttgames_test']
        out['ttgames_macro_f1'] = t.get('macro_f1')
        out['ttgames_bounce_f1'] = t.get('bounce', {}).get('f1')
        out['ttgames_net_f1'] = t.get('net', {}).get('f1')
    if 'pp_public' in mm:
        out['pp_public_macro_f1'] = mm['pp_public'].get('macro_f1')
    hb = mm.get('head_benchmark', {})
    if hb:
        out['head_params'] = hb.get('parameters', {}).get('head')
        out['head_p50_ms'] = hb.get('head_forward_p50_ms')
    return out


def copy_run_artifacts(run, dst, with_overlays=True):
    s = SRC / run
    d = dst
    d.mkdir(parents=True, exist_ok=True)
    # weights + config
    for f in ['checkpoints/best.pt']:
        if (s / f).exists():
            shutil.copy2(s / f, d / 'best.pt')
    for f in ['config.yaml', 'best_selection.json', 'run.json', 'training_curve.svg']:
        if (s / f).exists():
            shutil.copy2(s / f, d / f)
    # training records
    if (s / 'attempts').exists():
        shutil.copytree(s / 'attempts', d / 'attempts', dirs_exist_ok=True)
    # eval metrics
    ev = s / 'evaluations'
    if (ev / 'continuous/metrics.json').exists():
        shutil.copy2(ev / 'continuous/metrics.json', d / 'metrics.json')
    if (ev / 'latency.json').exists():
        shutil.copy2(ev / 'latency.json', d / 'latency.json')
    if (ev / 'external').exists():
        ext = d / 'external'
        ext.mkdir(exist_ok=True)
        for f in ['metrics.json', 'ttgames_test_result.json', 'pp_public_result.json', 'benchmark.json']:
            if (ev / 'external' / f).exists():
                shutil.copy2(ev / 'external' / f, ext / f)
    # error overlays
    if with_overlays and (ev / 'continuous').exists():
        ov = d / 'overlays'
        ov.mkdir(exist_ok=True)
        for f in (ev / 'continuous').iterdir():
            if f.name == 'index.html' or f.name.endswith('_overlay.mp4'):
                shutil.copy2(f, ov / f.name)


def fmt(v, nd=4):
    return '%.4f' % v if isinstance(v, (int, float)) else '—'


def scheme_readme(scheme, results, seed_rows):
    folder = scheme['folder']
    main = scheme['main_run']
    r = results.get(main, {})
    m = eval_metrics(main)
    ext = external_summary(main)
    lm = m.get('metrics', {}) if m else {}
    landing = lm.get('landing', {})
    by_view = lm.get('by_view', {})

    lines = [f"# {scheme['title']}（{folder}）", '']
    lines.append(scheme.get('description', ''))
    lines.append('')
    lines.append('## 结果（内部 PP 验证集，完整视频，落点宏 F1）')
    lines.append('')
    lines.append('| 指标 | 值 |')
    lines.append('|---|---:|')
    lines.append(f"| 落点宏 F1（seed42） | {fmt(r.get('f1'))} |")
    if landing:
        lines.append(f"| tp / fp / fn | {landing.get('tp')} / {landing.get('fp')} / {landing.get('fn')} |")
    lines.append(f"| e2e FPS | {fmt(r.get('fps'), 2)} |")
    if by_view:
        lines.append('')
        lines.append('| 视角 | F1 |')
        lines.append('|---|---:|')
        for v, x in by_view.items():
            lines.append(f"| {v} | {fmt(x.get('f1'))} |")
    if seed_rows:
        lines.append('')
        lines.append('## 多种子（内部验证宏 F1）')
        lines.append('')
        lines.append('| seed | 宏 F1 |')
        lines.append('|---:|---:|')
        for sr in seed_rows:
            lines.append(f"| {sr[0]} | {fmt(sr[1])} |")
    if ext:
        lines.append('')
        lines.append('## 外部评估（锁定阈值，未调参）')
        lines.append('')
        lines.append('| 数据集 | 宏 F1 |')
        lines.append('|---|---:|')
        if ext.get('ttgames_macro_f1') is not None:
            lines.append(f"| TTGames test | {fmt(ext['ttgames_macro_f1'])}（落点 {fmt(ext.get('ttgames_bounce_f1'))} / 擦网 {fmt(ext.get('ttgames_net_f1'))}） |")
        if ext.get('pp_public_macro_f1') is not None:
            lines.append(f"| PP public | {fmt(ext['pp_public_macro_f1'])} |")
        if ext.get('head_params') is not None:
            lines.append('')
            lines.append(f"事件头参数量：{ext['head_params']:,}；头部前向 p50：{fmt(ext.get('head_p50_ms'), 3)} ms")
    lines.append('')
    lines.append('## 文件说明')
    lines.append('')
    lines.append('- `best.pt`：事件头权重（seed42）。')
    lines.append('- `attempts/`：逐 epoch 训练日志；`training_curve.svg`：验证曲线。')
    lines.append('- `metrics.json`：连续验证集完整评估；`external/`：TTGames test + PP public；`latency.json`：逐帧延迟。')
    lines.append('- `overlays/`：错误叠图（`index.html` 浏览器查看，mp4 里红/绿框标出预测与真值，FP/FN 一目了然）。')
    if scheme.get('variant_runs'):
        lines.append('- `seeds/`：多种子权重与训练记录。')
    return '\n'.join(lines) + '\n'


def main():
    results = load_results()
    DST.mkdir(parents=True, exist_ok=True)

    schemes = [
        dict(
            folder='tcn_frames_lr0.0003_d0.1',
            title='TCN finalist（推荐）',
            main_run='tcn_frames_lr0.0003_d0.1',
            variant_runs=['tcn_frames_lr0.0003_d0.1_seed43', 'tcn_frames_lr0.0003_d0.1_seed44'],
            description='受控对比的推荐方案：TCN 时序头 + 桌框几何(box) + 时间编码 + frames 窗口 + lr3e-4 + dropout0.1。三种子平均 78.77%，是本次实验的最优结果。',
        ),
        dict(
            folder='scene_box',
            title='桌框场景融合（box）',
            main_run='scene_box',
            variant_runs=[],
            description='在规则关联 + TCN 基础上加入桌框位置/尺度几何分支（scene_mode=box），相对无场景基线 +0.76pp，是场景融合的正收益代表。',
        ),
        dict(
            folder='transformer_frames_lr0.0001_d0.1',
            title='Transformer finalist（对比）',
            main_run='transformer_frames_lr0.0001_d0.1',
            variant_runs=['transformer_frames_lr0.0001_d0.1_seed43', 'transformer_frames_lr0.0001_d0.1_seed44'],
            description='与 TCN 同前端、同时间编码的 Transformer 时序头（2 层、4 头），受控对比中落后 TCN 约 4.5pp，保留用于对照。',
        ),
        dict(
            folder='baseline_B',
            title='原始基线 B（参考）',
            main_run='baseline_B_seed42',
            variant_runs=['baseline_B_seed43', 'baseline_B_seed44'],
            description='原事件头（旧 EventHead）+ 规则关联，无场景、无时间编码，作为本次所有改进的参考基线。',
        ),
    ]

    top_lines = ['# 20261016 桌框 / 自回归 / 事件头实验结果', '']
    top_lines.append('实验根目录：`out/experiments/20261016_table_ar`；本次归档整理其中的有潜力结果。')
    top_lines.append('')
    top_lines.append('## 结论速览')
    top_lines.append('')
    top_lines.append('| 方案 | 内部验证宏 F1（seed42） | 三种子均值±std | 说明 |')
    top_lines.append('|---|---:|---:|---|')
    top_lines.append('| tcn_frames_lr0.0003_d0.1 | 79.56% | 78.77% ± 2.30pp | **推荐**（TCN+桌框+时间编码） |')
    top_lines.append('| scene_box | 78.73% | — | 桌框几何分支正收益 |')
    top_lines.append('| transformer_frames_lr0.0001_d0.1 | 75.03% | 75.72% ± 1.80pp | 对比（落后 TCN 4.5pp） |')
    top_lines.append('| baseline_B | 77.96% | 76.41% ± 1.11pp | 原始基线 |')
    top_lines.append('')
    top_lines.append('## 核心结论')
    top_lines.append('')
    top_lines.append('1. **桌框位置/尺度（box）有正收益（+0.76pp）**，加球台相对关系反而略降；桌框预测质量（IoU 38.75%）不是瓶颈。')
    top_lines.append('2. **自回归替换规则打分失败**（beam1/4/8 均 ~66%，落后 12pp），根因是 null 分支欠训练（训练集 null 目标仅 0.7%）。')
    top_lines.append('3. **TCN 显著优于 Transformer**（统一前端后 +4.53pp），且跨调参网格、跨种子、跨外部集一致。')
    top_lines.append('4. 最终推荐 `tcn_frames_lr0.0003_d0.1`，三种子平均 78.77%、全部 ≥50 FPS。')
    top_lines.append('')
    top_lines.append('## 目录')
    top_lines.append('')
    for s in schemes:
        top_lines.append(f"- [{s['folder']}](./{s['folder']}/README.md) — {s['title']}")
    top_lines.append('')
    (DST / 'README.md').write_text('\n'.join(top_lines) + '\n')

    for scheme in schemes:
        folder = DST / scheme['folder']
        main = scheme['main_run']
        copy_run_artifacts(main, folder, with_overlays=True)
        # variants
        if scheme['variant_runs']:
            seeds_dir = folder / 'seeds'
            seeds_dir.mkdir(exist_ok=True)
            for vr in scheme['variant_runs']:
                seed = read_json(SRC / vr / 'config.yaml', {}).get('seed', vr)
                sub = seeds_dir / f'seed{seed}'
                copy_run_artifacts(vr, sub, with_overlays=False)
        # seed rows for README
        seed_rows = []
        for name in [main] + scheme['variant_runs']:
            cfg = read_json(SRC / name / 'config.yaml', {})
            seed = cfg.get('seed', 42)
            f1 = results.get(name, {}).get('f1')
            seed_rows.append((seed, f1))
        (folder / 'README.md').write_text(scheme_readme(scheme, results, seed_rows))
        print('done', folder)

    print('organized into', DST)


if __name__ == '__main__':
    main()
