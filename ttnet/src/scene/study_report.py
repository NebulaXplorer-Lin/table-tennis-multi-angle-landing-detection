"""Publish measured outcomes, independent seeds and immutable artifact identities."""
import argparse,csv,json,html
from pathlib import Path
import numpy as np
import torch
from ttnet.common import PROJECT,save,sha,now,jsonl
from ttnet.events.engine import EventSystem


def curve(history,path):
    if not history:return
    series=[(h['epoch'],h.get('continuous',{}).get('macro_f1',0) if h.get('continuous') else h.get('val_mean_iou',h.get('val',{}).get('candidate_accuracy',0))) for h in history]
    maxx=max(x for x,y in series);points=' '.join(f'{40+680*x/maxx:.2f},{260-220*y:.2f}' for x,y in series)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="300"><rect width="760" height="300" fill="white"/><text x="20" y="20">Validation metric (0–1), epoch</text><path d="M40 40 V260 H730" fill="none" stroke="black"/><polyline points="{points}" fill="none" stroke="#2476ac" stroke-width="2"/></svg>')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();results=json.loads((root/'results.json').read_text());rows=[]
    for r in results:
        run=Path(r['run']);checkpoint=run/'checkpoints/best.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        system=EventSystem(state['config'],torch.device('cpu'),initialize=False)
        row=dict(name=r['name'],seed=r['config']['seed'],event_head=r['config']['event_head_type'],tracker=r['config'].get('tracker_type'),scene=r['config'].get('scene_mode','none'),cached_f1=r['cached_f1'],full_f1=r.get('f1'),e2e_fps=r.get('fps'),epoch=r['epoch'],head_parameters=sum(p.numel() for p in system.head.parameters()),checkpoint_sha256=sha(checkpoint),run=str(run))
        history=[]
        for f in sorted((run/'attempts').glob('*/epochs.jsonl')):history.extend(jsonl(f))
        curve(history,run/'training_curve.svg');rows.append(row)
        record=PROJECT/'experiments/records'/('20261016_table_ar_'+r['name']);save(record/'result.json',row);save(record/'config.yaml',state['config'])
        save(record/'run.json',dict(status='evaluated' if r.get('f1') is not None else 'trained_cached_validation',source=str(run),updated=now()))
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table=json.loads((root/'table/metrics.json').read_text());lines=['# 桌框、自回归与事件头实验结果','',f'桌框验证平均IoU：{table["mean_iou"]:.2%}；IoU≥0.5比例：{table["iou50_recall"]:.2%}。','', '| 实验 | seed | 缓存验证宏F1 | 完整视频宏F1 | FPS |','|---|---:|---:|---:|---:|']
    for r in rows:
        score=f'{r["full_f1"]:.2%}' if r['full_f1'] is not None else '未测';fps=f'{r["e2e_fps"]:.2f}' if r['e2e_fps'] else '未测'
        lines.append(f'| {r["name"]} | {r["seed"]} | {r["cached_f1"]:.2%} | {score} | {fps} |')
    locked=json.loads((root/'locked_selection.json').read_text());groups={}
    for final in locked['finalists']:
        matching=[r for r in rows if r['name']==final['name'] or r['name'].startswith(final['name']+'_seed')]
        values=[r['full_f1'] for r in matching if r['full_f1'] is not None];groups[final['name']]=dict(seeds=[r['seed'] for r in matching],mean=float(np.mean(values)),std=float(np.std(values)))
    base=[r['full_f1'] for r in rows if r['name'].startswith('baseline_B_seed')];groups['B']=dict(mean=float(np.mean(base)),std=float(np.std(base)),seeds=[42,43,44])
    feasible={}
    for name,stats in groups.items():
        members=[r for r in rows if (r['name'].startswith('baseline_B_seed') if name=='B' else r['name']==name or r['name'].startswith(name+'_seed'))]
        if len(members)==3 and all(r['e2e_fps'] is not None and r['e2e_fps']>=50 for r in members):feasible[name]=stats
    chosen=max(feasible,key=lambda name:feasible[name]['mean']) if feasible else None
    save(root/'recommendation.json',dict(recommended=chosen,criterion='highest three-seed mean among candidates >=50 FPS for all seeds',stability=groups))
    lines+=['',f'满足三次均≥50 FPS时，按三种子平均宏F1推荐：{chosen or "无候选达到速度门槛"}。']
    for name in ('table','ar_table','ar_no_table'):
        if (root/name/'history.json').exists():curve(json.loads((root/name/'history.json').read_text()),root/name/'training_curve.svg')
    save(root/'stability.json',groups);lines+=['','## 多种子',*['- '+k+f'：{v["mean"]:.2%} ± {v["std"]:.2%}（种子标准差）' for k,v in groups.items()],'','人工桌框仅为oracle诊断；实际方案使用预测桌框。缓存与完整视频结果分列。PP public属于历史已查看的迁移集。','所有配置和阈值在外部评估前锁定；错误叠加在各实验 evaluations/continuous/index.html，桌框叠加在 table/overlays/index.html。']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    save(PROJECT/'experiments/records/20261016_table_ar/result.json',dict(status='completed',root=str(root),table=table,stability=groups,completed=now()))

if __name__=='__main__':main()
