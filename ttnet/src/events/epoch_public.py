"""Capture an exact completed-epoch checkpoint, then evaluate PP public separately."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import cv2
import torch

from ttnet.common import OUTPUT, PROJECT, jsonl, now, save, sha
from ttnet.events.engine import EventSystem
from ttnet.events.evaluate import public_evaluation


def completed_epoch(state,epoch):
    return state.get('epoch')==epoch and state.get('step',0)==0 and 'thresholds' in state


def capture(run,dest,epoch,timeout):
    checkpoint=dest/'checkpoint.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if not completed_epoch(state,epoch):raise ValueError('snapshot does not match requested epoch')
        return state
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        seen=[]
        for filename in ('last.pt','best.pt'):
            source=run/'checkpoints'/filename
            if not source.exists():continue
            # Reading an opened inode is safe while training atomically replaces last.pt.
            data=source.read_bytes();state=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
            seen.append((state.get('epoch'),state.get('step',0)))
            if completed_epoch(state,epoch):
                temp=checkpoint.with_suffix('.tmp');temp.write_bytes(data);temp.replace(checkpoint)
                save(dest/'identity.json',dict(epoch=epoch,source=str(source),checkpoint_sha256=hashlib.sha256(data).hexdigest(),
                     thresholds=state['thresholds'],dataset_id=state['config']['dataset_id'],captured=now(),
                     protocol_sha256=sha(__file__),selection='exact epoch; thresholds selected on game_5 only'))
                print(f'Captured epoch {epoch}: {checkpoint}',flush=True)
                return state
        if seen and all(e is not None and (e>epoch or (e==epoch and step>0)) for e,step in seen):
            raise RuntimeError('requested epoch has already been overwritten and is not the best checkpoint')
        save(dest/'state.json',dict(status='waiting_for_epoch',epoch=epoch,observed=seen,updated=now()))
        time.sleep(2)
    raise TimeoutError(f'epoch {epoch} not ready within {timeout}s')


def evaluate_epoch(args):
    dest=args.run/'evaluations'/f'epoch{args.epoch:03d}_public'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'state.json').exists() and json.loads((dest/'state.json').read_text()).get('status')=='completed':return
    state=capture(args.run,dest,args.epoch,args.timeout)
    cfg=state['config'];torch.set_num_threads(cfg['torch_threads']);cv2.setNumThreads(1)
    save(dest/'state.json',dict(status='evaluating_public',epoch=args.epoch,updated=now()))
    public=dest/'public'
    if (public/'metrics.json').exists():metrics=json.loads((public/'metrics.json').read_text())
    else:
        if public.exists():public.rename(dest/f'public_partial_{len(list(dest.glob("public_partial_*")))+1:03d}')
        system=EventSystem(cfg,torch.device(args.device),initialize=False)
        system.load_state_dict(state['system']);system.configure(cfg['mode'],False)
        metrics=public_evaluation(system,args.public,public,state['thresholds'])
    metrics['speed_note']='GPU shared with continuing training; FPS is not a standalone speed comparison'
    save(public/'metrics.json',metrics)
    totals={k:sum(v[k] for v in metrics['by_angle'].values()) for k in ('tp','fp','fn')}
    lines=[f'# Joint epoch {args.epoch}: PP public', '',
           f'固定第{args.epoch}轮完整权重；阈值来自该轮game_5验证：{state["thresholds"]}。未在public调参。',
           f'权重SHA256：`{sha(dest/"checkpoint.pt")}`。', '',
           '| 视角 | P | R | F1 | TP | FP | FN |','|---|---:|---:|---:|---:|---:|---:|']
    for angle,m in metrics['by_angle'].items():
        lines.append(f'| {angle} | {m["precision"]:.4f} | {m["recall"]:.4f} | {m["f1"]:.4f} | {m["tp"]} | {m["fp"]} | {m["fn"]} |')
    lines.extend(['',f'宏平均F1：**{metrics["macro_f1"]:.4f}**；TP/FP/FN：{totals["tp"]}/{totals["fp"]}/{totals["fn"]}。',
                  '按±1帧、50px、一对一匹配评估全部15条public视频。', '',
                  '[完整指标](public/metrics.json) · [逐视频](public/per_video.json) · [错误图册](public/errors.html) · [预测](public/predictions.jsonl)', '',
                  '这是中途快照评估，联合训练继续运行。GPU与训练共享，本次FPS不用于独立测速对比。'])
    frozen=PROJECT/'ttnet/output/events/20261014-181231_event25_frozen/evaluations/best/public/metrics.json'
    if frozen.exists():
        reference=json.loads(frozen.read_text())['macro_f1']
        lines.extend(['',f'冻结版最终best的public宏F1为{reference:.4f}；本次第{args.epoch}轮差值为{metrics["macro_f1"]-reference:+.4f}，不代表联合版最终结果。'])
    if metrics['macro_f1']<.6:
        lines.extend(['', '当前迁移效果仍偏低。固定阈值错误拆分：`'+json.dumps(metrics.get('error_diagnostics',{}),ensure_ascii=False)+'`。',
                      '优先结合错误图册复核连续误报与场景迁移；此评估不触发额外训练或public阈值搜索。'])
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    save(dest/'state.json',dict(status='completed',epoch=args.epoch,macro_f1=metrics['macro_f1'],
                              checkpoint_sha256=sha(dest/'checkpoint.pt'),finished=now()))
    output=OUTPUT/'events'/f'{args.run.name}_epoch{args.epoch:03d}_public'
    if not output.exists():shutil.copytree(dest,output)
    print(f'Public epoch {args.epoch} macro F1={metrics["macro_f1"]:.6f}; {dest/"summary.md"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--public',type=Path,default=PROJECT/'data/pp_public_data')
    parser.add_argument('--device',default='cuda');parser.add_argument('--timeout',type=int,default=3600)
    args=parser.parse_args()
    try:evaluate_epoch(args)
    except Exception as exc:
        dest=args.run/'evaluations'/f'epoch{args.epoch:03d}_public';dest.mkdir(parents=True,exist_ok=True)
        save(dest/'state.json',dict(status='failed',error=repr(exc),epoch=args.epoch,finished=now()))
        raise


if __name__=='__main__':main()
