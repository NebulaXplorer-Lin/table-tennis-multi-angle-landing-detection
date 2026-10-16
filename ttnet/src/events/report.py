"""Publish event results separately from localization-only comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

from ttnet.common import OUTPUT, PROJECT, jsonl, now, save, sha


def report(run):
    run=Path(run);cfg=json.loads((run/'config.yaml').read_text());state=json.loads((run/'run.json').read_text())
    history=[r for path in sorted((run/'attempts').glob('*/epochs.jsonl')) for r in jsonl(path)]
    metrics_path=run/'evaluations/best/metrics.json'
    metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    best=json.loads((run/'best_selection.json').read_text()) if (run/'best_selection.json').exists() else None
    lines=[f'# EventHead {cfg["mode"]}', '',f'状态：{state["status"]}；更新：{now()}。', '',
           '数据：OpenTTGames game_1–4 训练，game_5 验证，test_1–7 独立测试；PP public 仅迁移评估。',
           '输入：25 FPS 最近原帧采样；13帧上下文；真实事件时间生成半宽40 ms三角软标签。',
           '模型：256维视觉特征 + 9维轨迹输入，128通道TCN，Bounce / Net 两类。',
           f'训练：AdamW；Head lr={cfg["lr"]}；定位 lr={cfg["locator_lr"] if cfg["mode"]=="joint" else "冻结"}；有效batch={cfg["effective_batch"]}；seed={cfg["seed"]}；最多{cfg["epochs"]} epoch。',
           'best按验证masked BCE最低选择；验证阈值固定后才评估测试与PP public。', '']
    if best:
        lines.extend([f'最佳epoch：{best["epoch"]}；验证BCE：{best["val_event_loss"]:.6f}；阈值：{best["thresholds"]}。', ''])
    if metrics:
        lines.extend(['| 数据 / 指标 | Bounce F1 | Net F1 | 宏平均 F1 |','|---|---:|---:|---:|'])
        for part in ('val','test'):
            m=metrics[part];lines.append(f'| {part} 标注样本 | {m["bounce"]["f1"]:.4f} | {m["net"]["f1"]:.4f} | {m["macro_f1"]:.4f} |')
        lines.extend(['',f'PP public 端到端宏平均F1：**{metrics["public"]["macro_f1"]:.4f}**；完整管线FPS：{metrics["public"]["e2e_fps"]:.2f}。',
                      f'PP事件成立但缺坐标：{metrics["public"]["events_without_coordinates"]}。', '',
                      'OpenTTGames成绩仅针对稀疏标注样本；完整留出视频已推理，未审阅时刻不计为背景。不能将样本分类F1称为连续视频事件F1。', '',
                      '[完整指标](evaluations/best/metrics.json) · [验证错误图册](evaluations/best/val/errors.html) · '
                      '[测试错误图册](evaluations/best/test/errors.html) · [PP错误图册](evaluations/best/public/errors.html)', '',
                      '## 诊断与改进方向', ''])
        m=metrics['test'];loc=m['localization']['50'];timing=m['local_timing']['bounce']
        lines.append(f'独立测试球定位F1@50px={loc["f1"]:.4f}；已知Bounce附近±1帧命中率={timing["hit_within_one_frame"]:.4f}。')
        if loc['recall']<.9:
            lines.append('定位召回不足：先检查OpenTTGames域差异、0.7定位阈值、ROI覆盖与缺失区间；后续可在独立验证方案中比较阈值校准和定位适配。')
        if m['macro_f1']<.8:
            lines.append('事件样本效果偏低：结合错误图册区分视觉描述符不足、缺失轨迹及类别混淆；后续优先考虑保留ROI空间特征或增加明确困难负样本。')
        if timing['hit_within_one_frame']<.8:
            lines.append('时间定位不足：检查相邻帧概率曲线及原始事件时间；后续可比较时间偏移回归与多采样相位，不以测试集调参数。')
        if metrics['public']['macro_f1']<.6:
            lines.append('PP迁移效果偏低：OpenTTGames单域稀疏标注不足以证明多视角落点能力；建议补PP train完整片段事件标签和连续负样本，并单独评估真实接触坐标回归。')
        diagnostic=metrics['public'].get('error_diagnostics')
        if diagnostic:
            lines.append(f'固定阈值错误拆分：事件时刻附近位置超差 {diagnostic.get("position_error_at_event_time",0)}；'
                         f'位置接近但时间偏2–3帧 {diagnostic.get("nearby_time_error_2_to_3_frames",0)}；'
                         f'其余未匹配事件 {diagnostic.get("other_unmatched_event",0)}；边界FP {diagnostic.get("boundary_false_positives",0)}（与前述类型有重叠）。'
                         '其余未匹配事件的接触类型需看动作片段复核，不直接断定为击球或落地。')
        lines.extend(['当前落点直接使用事件帧的预测球坐标；25 FPS采样可能错过接触瞬间，因此时间正确仍可能出现空间误差。',
                      '上述为基于现有结果的改进假设，未自动启动额外实验。', ''])
        if (run/'analysis.md').exists():lines.extend([(run/'analysis.md').read_text(),''])
    else:
        lines.extend(['尚未完成最终评估；未测指标留空。', ''])
    lines.extend(['[实际配置](config.yaml) · [运行状态](run.json) · [最佳验证记录](best_selection.json)', '',
                  f'已记录epoch：{len(history)}。'])
    (run/'summary.md').write_text('\n'.join(lines)+'\n')
    save(run/'training.json',history)
    if history:
        os.environ.setdefault('MPLCONFIGDIR','/tmp/pingpong-event-matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(10,3.5))
        ax[0].plot([r['epoch'] for r in history],[r['train']['event'] for r in history],label='train event BCE')
        ax[0].plot([r['epoch'] for r in history],[r['val_event_loss'] for r in history],label='val event BCE')
        ax[0].legend();ax[0].set_xlabel('epoch')
        ax[1].plot([r['epoch'] for r in history],[r['val']['macro_f1'] for r in history],label='val anchor macro F1')
        ax[1].legend();ax[1].set_ylim(0,1);ax[1].set_xlabel('epoch');fig.tight_layout()
        fig.savefig(run/'training_curve.png',dpi=130);plt.close(fig)


def publish(run):
    run=Path(run);report(run)
    cfg=json.loads((run/'config.yaml').read_text());state=json.loads((run/'run.json').read_text())
    if cfg['smoke']:return
    output=OUTPUT/'events';output.mkdir(parents=True,exist_ok=True)
    dest=output/run.name
    if not dest.exists():
        shutil.copytree(run,dest)
    else:
        # Existing output is a snapshot; evaluation runs get new identities.
        if state['status']=='completed':
            stamp=now().replace(':','').replace('.','_')
            snapshot=output/'.history'/f'{run.name}_{stamp}'
            shutil.copytree(dest,snapshot)
            shutil.copytree(run,dest,dirs_exist_ok=True)
    save(dest/'manifest.json',dict(stage='event',mode=cfg['mode'],status=state['status'],source=str(run),published=now(),
                                  files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    refresh(output)


def refresh(output=None):
    output=Path(output or OUTPUT/'events');output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for dest in sorted(output.iterdir()):
        if not dest.is_dir() or not (dest/'config.yaml').exists():continue
        cfg=json.loads((dest/'config.yaml').read_text());state=json.loads((dest/'run.json').read_text())
        path=dest/'evaluations/best/metrics.json';m=json.loads(path.read_text()) if path.exists() else None
        rows.append(dict(experiment_id=dest.name,mode=cfg['mode'],status=state['status'],dataset_id=cfg['dataset_id'],
                         val_anchor_macro_f1=m['val']['macro_f1'] if m else None,test_anchor_macro_f1=m['test']['macro_f1'] if m else None,
                         public_macro_f1=m['public']['macro_f1'] if m else None,e2e_fps=m['public']['e2e_fps'] if m else None,
                         summary=f'{dest.name}/summary.md'))
    fields=list(rows[0]) if rows else ['experiment_id','mode','status']
    with (output/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def number(v):return '—' if v is None else f'{v:.4f}'
    lines=['# EventHead 两版训练结果', '', '| 实验 | 模式 | 状态 | 验证样本宏F1 | 测试样本宏F1 | PP端到端宏F1 | FPS |',
           '|---|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| [{r["experiment_id"]}]({r["summary"]}) | {r["mode"]} | {r["status"]} | '+
                               ' | '.join(number(r[k]) for k in ('val_anchor_macro_f1','test_anchor_macro_f1','public_macro_f1','e2e_fps'))+' |')
    completed=[r for r in rows if r['status']=='completed']
    if len(completed)>=2:
        latest={r['mode']:r for r in completed}
        if {'frozen','joint'}<=latest.keys():
            a,b=latest['frozen'],latest['joint']
            if a['dataset_id']!=b['dataset_id']:raise ValueError('cannot compare different datasets')
            lines.extend(['',f'联合版相对冻结版：测试标注样本宏F1差值 {b["test_anchor_macro_f1"]-a["test_anchor_macro_f1"]:+.4f}；'
                         f'PP端到端宏F1差值 {b["public_macro_f1"]-a["public_macro_f1"]:+.4f}。',
                         '选优仅使用game_5。PP public为迁移诊断，独立测试结果不用于追加调参。'])
    (output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    publish(parser.parse_args().run)
