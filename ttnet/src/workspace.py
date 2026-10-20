"""Workspace inventory, non-destructive archival, and a task-specific result index."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

from ttnet.common import PROJECT,ROOT,RUNS,OUTPUT,identity,now,save,sha


def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def scheme(config):
    if config.get('scheme'):return config['scheme']
    if config.get('stage')=='event':return f"event_{config.get('event_head_type','tcn')}_{config.get('mode','unknown')}"
    return f"ball_{config.get('temporal_mode','single')}"


def relative(path):
    return os.path.relpath(Path(path),PROJECT)


def refresh_index():
    """Read archived evidence only; never re-publish or modify historical runs."""
    entries=[];seen=set()
    for base in (ROOT/'runs',RUNS):
        for run in sorted(base.glob('*')):
            if not run.is_dir() or run.resolve() in seen:continue
            seen.add(run.resolve())
            config=read(run/'config.yaml');state=read(run/'run.json')
            if not config:continue
            stage=config.get('stage','pipeline')
            group=scheme(config)
            published=next((p for p in (OUTPUT/run.name,OUTPUT/'events'/run.name,
                ROOT/'output'/run.name,ROOT/'output/events'/run.name) if (p/'summary.md').is_file()),None)
            summary=published/'summary.md' if published else run/('summary.md' if (run/'summary.md').exists() else 'run.json')
            base_entry=dict(experiment_id=run.name,scheme=group,stage=stage,status=state.get('status','unknown'),
                   smoke=config.get('smoke',False),dataset_id=config.get('dataset_id'),split_id=config.get('split_id'),
                   source=relative(run),summary=relative(summary),best_epoch=state.get('best_epoch'),
                   config=relative(run/'config.yaml'),checkpoint=relative(run/'checkpoints/best.pt') if (run/'checkpoints/best.pt').exists() else None)
            scores=[]
            if stage=='event':
                continuous=read(run/'evaluations/continuous/metrics.json')
                if continuous.get('status')=='evaluated':
                    base_entry['status']='evaluated'
                    base_entry['summary']=relative(run/'evaluations/continuous/metrics.json')
                    scores.append(dict(eval_id='continuous',metric_scope='pp_val_landing_macro_f1',
                        f1=continuous['metrics']['macro_f1'],precision=None,recall=None,
                        e2e_fps=continuous.get('e2e_fps')))
                m=read(run/'evaluations/best/metrics.json')
                for part in ('val','test'):
                    if part in m:scores.append(dict(eval_id=f'best_{part}_anchors',metric_scope=f'{part}_sparse_event_classification',
                       f1=m[part].get('macro_f1'),precision=None,recall=None,e2e_fps=None))
                if 'public' in m:scores.append(dict(eval_id='best_public_landing',metric_scope='pp_public_landing',
                    f1=m['public'].get('macro_f1'),precision=None,recall=None,e2e_fps=m['public'].get('e2e_fps')))
                for path in sorted((run/'evaluations').glob('epoch*_public/public/metrics.json')):
                    m=read(path);scores.append(dict(eval_id=path.parent.parent.name,metric_scope='pp_public_landing_intermediate',
                       f1=m.get('macro_f1'),precision=None,recall=None,e2e_fps=None))
            else:
                m=read(run/'evaluations/best_val/metrics.json');bench=read(run/'evaluations/benchmark/metrics.json')
                for tol,metric in m.get('center',{}).items():
                    scores.append(dict(eval_id=f'best_val_{tol}px',metric_scope=f'val_ball_center_{tol}px',
                         f1=metric.get('f1'),precision=metric.get('precision'),recall=metric.get('recall'),e2e_fps=bench.get('e2e_fps')))
            if not scores:scores=[dict(eval_id='unscored',metric_scope='unscored',f1=None,precision=None,recall=None,e2e_fps=None)]
            entries.extend(dict(base_entry,**score) for score in scores)
    known={entry['experiment_id'] for entry in entries}
    for record in sorted((PROJECT/'experiments/records').glob('*')):
        state=read(record/'run.json')
        if record.name in known or state.get('status')!='launcher_failed':continue
        config=read(record/'config.yaml')
        entries.append(dict(experiment_id=record.name,scheme=scheme(config),stage='launcher',status='launcher_failed',
            smoke=False,dataset_id=config.get('dataset_id'),split_id=config.get('split_id'),source=relative(record),
            summary=relative(record/'run.json'),best_epoch=None,config=relative(record/'config.yaml'),checkpoint=None,
            eval_id='unscored',metric_scope='unscored',f1=None,precision=None,recall=None,e2e_fps=None))
    legacy=PROJECT/'experiments/comparison.csv'
    if legacy.exists():
        with legacy.open() as stream:
            for row in csv.DictReader(stream):
                def number(key):
                    value=row.get(key)
                    return float(value) if value not in (None,'','null','—') else None
                entries.append(dict(experiment_id=row['experiment_id'],scheme='legacy_'+row.get('model','unknown').replace(' ','_'),
                    stage=row.get('task'),status=row.get('status'),smoke=False,dataset_id=row.get('data_version'),split_id=row.get('split_id'),
                    source='experiments/records/'+row['experiment_id'],summary='experiments/records/'+row['experiment_id']+'/summary.md',
                    best_epoch=row.get('best_epoch'),config=row.get('config'),checkpoint=row.get('checkpoint'),eval_id=row.get('eval_id'),
                    metric_scope='legacy_'+row.get('protocol','unknown'),f1=number('center_f1') if row.get('task')=='ball_localisation' else number('touch_f1'),
                    precision=number('center_p'),recall=number('center_r'),e2e_fps=number('e2e_fps')))
    root=PROJECT/'experiments';root.mkdir(exist_ok=True)
    # Keep reviewed, portable result snapshots even when the original runtime
    # lives in a machine-local archive. Never reintroduce removed experiments
    # through a stale legacy CSV or compatibility link.
    curation=read(root/'curation.json')
    retained=curation.get('retained_results',[])
    replaced={row['experiment_id'] for row in retained}
    archived=set(curation.get('archived_experiment_ids',[]))
    entries=[row for row in entries if row['experiment_id'] not in archived|replaced
             and not (curation and row.get('smoke'))]
    entries=retained+entries
    save(root/'index.json',dict(schema=1,updated=now(),experiments=entries))
    fields=list(entries[0]) if entries else ['experiment_id','scheme','metric_scope','f1']
    with (root/'workspace_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(entries)
    def number(value):return '—' if value is None else f'{float(value):.4f}'
    lines=['# 方案与实验索引','', '保留完整版本与已评估对照；中断和预检记录已移出日常入口。指标按任务和评估协议分别展示，稀疏事件分类F1不能替代完整落点F1。', '',
           '[Quick intro](../docs/quick_intro.md) · [当前推荐结果](../result/README.md) · [后续计划](../docs/next_steps.md) · [CSV数据源](workspace_comparison.csv)', '']
    for group in sorted({e['scheme'] for e in entries}):
        lines.extend([f'## {group}', '', '| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |','|---|---|---|---|---:|---:|'])
        for row in entries:
            if row['scheme']!=group:continue
            lines.append(f'| [{row["experiment_id"]}](../{row["summary"]}) | {row["stage"]} | {row["status"]} | {row["metric_scope"]} | {number(row["f1"])} | {number(row["e2e_fps"])} |')
        lines.append('')
    (root/'index.md').write_text('\n'.join(lines).rstrip()+'\n')
    return root/'index.md'


def verify_protected(state=None):
    state=state or read(PROJECT/'out/workspace/current.json')
    errors=[]
    for record in state['protected_artifacts']:
        path=PROJECT/record['path']
        if not path.is_file() or path.stat().st_size!=record['size'] or sha(path)!=record['sha256']:
            errors.append(record['path'])
    if errors:raise ValueError(f'historical artifacts changed or missing: {errors}')
    return len(state['protected_artifacts'])


def archive_workspace():
    state_path=PROJECT/'out/workspace/current.json';state=read(state_path)
    if read(PROJECT/'out/workspace/frame_migration.json').get('status')!='completed':
        raise ValueError('frame migration must complete before archival')
    if not read(PROJECT/'out/workspace/validation.json').get('passed'):
        raise ValueError('model and checkpoint parity validation must pass before archival')
    # Do not move data underneath a live training/preparation process.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():continue
        try:
            if not (process/'comm').read_text().strip().startswith('python'):continue
            args=(process/'cmdline').read_bytes().split(b'\0')
            if any(b'ttnet.training.train'==a or b'ttnet.events.train'==a or b'ttnet.data_tools.workspace_migrate'==a for a in args):
                raise RuntimeError(f'workspace writer still active: PID {process.name}')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    verify_protected(state)
    dest=PROJECT/state['archive'];moves=state.get('moves',[])
    # Preserve every previously versioned runtime record as a byte-identical
    # lightweight snapshot, rather than leaving its only copy in ignored data.
    mapping_path=PROJECT/'experiments/legacy_record_paths.json'
    snapshots={r['source']:r for r in read(mapping_path).get('files',[])}
    indexed=subprocess.check_output(['git','ls-files','-z','ttnet/runs'],cwd=PROJECT).decode().split('\0')
    for name in filter(None,indexed):
        source=PROJECT/name
        if not source.is_file():raise FileNotFoundError(source)
        parts=Path(name).parts
        target=PROJECT/'experiments/records'/parts[2]/'history'/Path(*parts[3:])
        target.parent.mkdir(parents=True,exist_ok=True)
        checksum=sha(source)
        if target.exists() and sha(target)!=checksum:raise ValueError(f'historical snapshot collision: {target}')
        if not target.exists():shutil.copy2(source,target)
        snapshots[name]=dict(source=name,snapshot=relative(target),sha256=checksum)
    save(mapping_path,dict(schema=1,batch=state['batch'],files=sorted(snapshots.values(),key=lambda r:r['source'])))
    state['versioned_records_preserved']=len(snapshots)
    # Keep lightweight evidence in version-controlled experiment records even
    # when its original runtime directory is moved into the large-file archive.
    for run in sorted((ROOT/'runs').glob('*')):
        if not (run/'run.json').is_file():continue
        record=PROJECT/'experiments/records'/run.name
        record.mkdir(parents=True,exist_ok=True)
        for filename in ('config.yaml','run.json','best_selection.json'):
            source=run/filename;target=record/filename
            if not source.is_file():continue
            if target.exists() and sha(target)!=sha(source):raise ValueError(f'record collision: {target}')
            if not target.exists():shutil.copy2(source,target)
        evidence=[]
        for source in sorted((run/'evaluations').glob('*/metrics.json')):
            target=record/'evaluations'/f'{source.parent.name}.json'
            target.parent.mkdir(exist_ok=True)
            if not target.exists():shutil.copy2(source,target)
            evidence.append(relative(source))
        report=next((p for p in (ROOT/'output'/run.name/'summary.md',ROOT/'output/events'/run.name/'summary.md',run/'summary.md') if p.is_file()),run/'run.json')
        summary=record/'summary.md'
        if not summary.exists():summary.write_text(f'# {run.name}\n\n历史实验，原始配置、状态和产物保持原样。\n\n[完整报告](../../../{relative(report)}) · [原始运行目录](../../../{relative(run)})\n')
        save(record/'archive_provenance.json',dict(batch=state['batch'],source=relative(run),evidence=evidence))
    candidates=[PROJECT/'ttnet/data',PROJECT/'ttnet/runs',PROJECT/'ttnet/output',PROJECT/'out/datasets',
                PROJECT/'out/yolo_setup',PROJECT/'out/yolo-deps',PROJECT/'out/local_smoke',PROJECT/'sport-base_v1.tar']
    legacy_runs=sorted({Path(r['path']).parts[2] for r in state['protected_artifacts']
                        if r['path'].startswith('out/experiments/')})
    legacy_comparison=PROJECT/'experiments/comparison.csv'
    if legacy_comparison.is_file():
        with legacy_comparison.open() as stream:
            legacy_runs=sorted(set(legacy_runs)|{r['experiment_id'] for r in csv.DictReader(stream)})
    candidates += [RUNS/name for name in legacy_runs]
    for source in candidates:
        target=dest/source.relative_to(PROJECT)
        if source.is_symlink():continue
        if not source.exists():
            if target.exists():
                source.symlink_to(os.path.relpath(target,source.parent),target_is_directory=target.is_dir())
                item=dict(source=relative(source),archive=relative(target))
                if item not in moves:moves.append(item)
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        source.rename(target)
        source.symlink_to(os.path.relpath(target,source.parent),target_is_directory=target.is_dir())
        moves.append(dict(source=relative(source),archive=relative(target)))
        state.update(moves=moves,status='archiving');save(state_path,state)
    count=verify_protected(state)
    state.update(status='archived',finished=now(),verified_artifacts=count)
    save(state_path,state);save(dest/'inventory.json',state)
    lines=['# 历史工作区归档','',f'批次：{state["batch"]}。历史权重、状态和配置已校验；没有删除旧图片或训练数据。','',
           '旧路径保留兼容链接；新实验默认写入 out/experiments，新数据默认读取 data/datasets。','',
           '| 兼容路径 | 实体目录 |','|---|---|']
    lines.extend(f'| `{m["source"]}` | `{m["archive"]}` |' for m in moves)
    (dest/'README.md').write_text('\n'.join(lines)+'\n')
    refresh_index()
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('index','archive','verify'),default='index',nargs='?')
    args=parser.parse_args()
    result={'index':refresh_index,'archive':archive_workspace,'verify':verify_protected}[args.action]()
    print(json.dumps(result,indent=2) if isinstance(result,dict) else result)


if __name__=='__main__':main()
