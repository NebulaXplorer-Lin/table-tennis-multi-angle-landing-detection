"""Refresh final-output snapshots and the training progress entry point."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path

from ttnet.common import ROOT, now, save
from ttnet.progress import epoch_records, training_curve
from ttnet.publication import TERMINAL, publish_run, read, save_text, write_index


def _refresh_results(root=ROOT):
    root = Path(root).resolve()
    runs = sorted(p for p in (root/'runs').glob('*') if p.is_dir())
    errors = []
    for run in runs:
        config = read(run/'config.yaml')
        records = epoch_records(run)
        if records and config.get('stage') in {'global', 'local'} and not config.get('smoke'):
            training_curve(run, records)
        try:
            publish_run(run, root)
        except Exception as error:
            errors.append(dict(run=run.name, error=repr(error)))
            print(f'Result publication failed for {run.name}: {error}', flush=True)
    index = write_index(root)
    for run in runs:
        state = read(run/'run.json')
        if state.get('status')!='completed' or not state.get('global_run') or not state.get('local_run'):
            continue
        stages = {name: Path(state[name+'_run']).name for name in ['global', 'local']}
        if not all((root/'output'/value/'manifest.json').is_file() for value in stages.values()):
            continue
        dest = root/'output'/run.name
        dest.mkdir(exist_ok=True)
        completed = all(read(root/'output'/value/'manifest.json').get('status')=='completed' for value in stages.values())
        status = 'completed' if completed else 'partial'
        save(dest/'task.json', dict(status=status, source_run=str(run), stages=stages))
        save_text(dest/'summary.md', '# 训练任务总结果\n\n状态：'+status+'\n\n'+
                  '\n\n'.join(f'[{name} 最终结果](../{value}/summary.md)' for name,value in stages.items())+'\n')
    tasks = sorted((root/'output').glob('*/task.json'))
    if tasks:
        text = index.read_text()+'\n## 完整训练任务\n\n'
        text += '\n'.join(f'- [{p.parent.name}]({p.parent.name}/summary.md)' for p in reversed(tasks))+'\n'
        save_text(index, text)
    lines = ['# TTNet 结果入口', '', f'更新于 {now()}。', '',
             '[查看全部最终结果、权重、Recall / F1、测速和错误样本](output/README.md)', '',
             '最终文件集中在 `output/<任务编号>/`；每个 summary.md 提供直接链接。`runs/` 保存启动、续训、日志和中间产物。', '',
             '| 当前实验 | 运行状态 | 完成 epoch | 最终结果 |', '|---|---|---:|---|']
    latest = {}
    for run in runs:
        config = read(run/'config.yaml')
        if config.get('stage') in {'global', 'local'} and not config.get('smoke'):
            latest[config['stage']] = run
    for stage, run in latest.items():
        state = read(run/'run.json')
        records = epoch_records(run)
        result = root/'output'/run.name/'summary.md'
        target = f'[汇总](output/{run.name}/summary.md)' if result.is_file() else '尚未生成'
        lines.append(f'| {run.name} | {state.get("status", "unknown")} | {records[-1]["epoch"] if records else 0} | {target} |')
    if errors:
        lines += ['', '结果发布错误（未影响训练；修复后重新运行 `python -m ttnet.results`）：', '']
        lines += [f'- {error["run"]}: {error["error"]}' for error in errors]
    save(root/'output/publication_errors.json', errors)
    save_text(root/'RESULTS.md', '\n'.join(lines)+'\n')
    return root/'RESULTS.md'


def _write_results(root=ROOT, run=None):
    output = Path(root)/'output'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if run is not None:
            publish_run(run, root)
        return _refresh_results(root)


def write_results(root=ROOT):
    try:
        return _write_results(root)
    except Exception as error:
        print(f'Result overview could not refresh: {error}', flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description='Publish final snapshots without training or inference')
    parser.add_argument('--run', type=Path, help='publish one terminal run, then refresh all indexes')
    args = parser.parse_args()
    if args.run:
        state = read(args.run/'run.json')
        config = read(args.run/'config.yaml')
        if state.get('status') not in TERMINAL or config.get('smoke') or config.get('stage') not in {'global', 'local'}:
            parser.error('run 尚未结束、是 smoke，或缺少阶段配置')
    result = _write_results(run=args.run)
    if read(ROOT/'output/publication_errors.json'):
        raise SystemExit('存在结果发布失败，见 output/publication_errors.json')
    print(result)


if __name__=='__main__':
    main()
