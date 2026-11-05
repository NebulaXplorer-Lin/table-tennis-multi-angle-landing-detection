"""Run the approved mixed-data A/B experiment and all required evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import os
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from ttnet.common import PROJECT, DATA, RUNS, SRC, identity, jsonl, now, save, save_jsonl, sha
from ttnet.configuration import load_config


def execute(module,*args,log):
    cmd=[sys.executable,'-u','-m',module,*map(str,args)]
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    print('RUN '+' '.join(cmd),flush=True)
    with log.open('w') as stream:
        code=subprocess.call(cmd,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
    save(log.with_suffix('.exit.json'),dict(command=cmd,exit_code=code,finished=now()))
    if code:raise RuntimeError(f'command exited {code}: {log}')


def ready_evaluation(directory,marker):
    directory=Path(directory)
    if (directory/marker).is_file():return True
    if directory.exists():
        history=directory.parent/'.failed';history.mkdir(exist_ok=True)
        directory.rename(history/f'{directory.name}-{time.time_ns()}')
    return False


def prepare_sampling(task,dataset):
    from ttnet.training.mixed_sampling import MixedSourceSampler
    rows=[r for r in jsonl(Path(dataset)/'labels/train.jsonl') if r['label_status'] in ('positive','negative')]
    sampler=MixedSourceSampler(rows,42,16);root=Path(task)/'shared_sampling';root.mkdir(parents=True,exist_ok=True)
    manifests=[]
    for epoch in range(20):
        sampler.set_epoch(epoch);manifest=sampler.manifest;path=root/f'epoch_{epoch+1:03d}.jsonl'
        if not path.exists():save_jsonl(path,[dict(source_domain=rows[i]['source_domain'],video_id=rows[i]['video_id'],frame_id=rows[i]['frame_id']) for i in sampler])
        manifests.append(dict(manifest,path=path.name,file_sha256=sha(path)))
    save(root/'manifest.json',dict(dataset_id=json.loads((Path(dataset)/'metadata/dataset.json').read_text())['dataset_id'],
                                  seed=42,effective_batch=16,epochs=manifests))


def compatibility(global_checkpoint,local_checkpoint,output):
    """Use actual PP and event25 windows; do not update any saved parameters."""
    from ttnet.events.engine import EventSystem
    from ttnet.events.data import EventCorpus
    from ttnet.evaluation.ball_corpus import BallCorpus
    from ttnet.evaluation.native_ball import system_config
    cfg=system_config(global_checkpoint,local_checkpoint,checkpoint_thresholds=True)
    system=EventSystem(cfg,torch.device('cuda'));system.configure('joint',True)
    reports=[]
    for name,corpus in [('pp',BallCorpus(DATA)),('event25',EventCorpus(PROJECT/'data/datasets/openttgames_event25_v3'))]:
        if name=='pp':
            vid=next(v for v in corpus.split['val'] if corpus.videos[v]['meta']['n_frames']>=15)
            ids=list(range(1,14))
        else:
            vid=corpus.split['val'][0];center=corpus.samples['val'][0]['frame_id']
            ids=list(range(center-6,center+7))
        # Memory bounded while retaining descriptor autograd graphs for 13 frames.
        with torch.autocast('cuda'):
            features,records,_=system.forward_frames(corpus,vid,ids)
            visual=torch.stack([features[i] for i in ids])[None]
            motion=torch.zeros(1,13,9,device='cuda')
            motion[:,:,7]=torch.tensor([[records[i]['local_valid'] for i in ids]],device='cuda')
            logits=system.head(visual,motion,torch.ones(1,13,device='cuda',dtype=torch.bool))
        if visual.shape!=(1,13,256) or not torch.isfinite(visual).all():raise ValueError('feature interface')
        system.zero_grad(set_to_none=True);logits.float().square().mean().backward()
        gradients={stage:sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
                   for stage,model in [('global',system.global_model),('local',system.local_model)]}
        if not all(np.isfinite(v) and v>0 for v in gradients.values()):raise ValueError(f'feature gradient {name}: {gradients}')
        reports.append(dict(source=name,video_id=vid,frame_ids=ids,shape=list(visual.shape),gradient_abs_sum=gradients))
        del visual,features,logits
    save(output,dict(status='passed',cases=reports,global_sha256=sha(global_checkpoint),local_sha256=sha(local_checkpoint)))


def train_variant(variant,paths,preset,micro,task):
    for stage in ('global','local'):
        run=paths[stage]
        state_path=run/'run.json'
        attached=live_launchers(run)
        while live_launchers(run):
            print(f'ATTACH {variant} {stage}: existing trainer is running',flush=True)
            time.sleep(30)
        if attached:
            save(task/'logs'/f'{variant}_{stage}_attachment.json',dict(attached_pids=attached,finished=now(),
                 exit_code=None,note='coordinator handoff; completion checked from checkpoint/run state, original launcher exit code unavailable'))
        if state_path.is_file() and json.loads(state_path.read_text()).get('status') in ('trained_pending_eval','evaluated_pending_export','completed'):
            continue
        checkpoint_args=['--global-checkpoint',paths['global']/'checkpoints/best.pt'] if stage=='local' else []
        smoke=RUNS/(run.name+'_smoke')
        if smoke.exists() and json.loads((smoke/'run.json').read_text()).get('status')!='completed_smoke':
            raise RuntimeError(f'previous smoke did not pass: {smoke}')
        if not smoke.exists():
            execute('ttnet.cli','train','--config',preset,'--stage',stage,'--run',smoke,
                    '--set',f'micro_batch={micro}','--smoke',*checkpoint_args,log=task/'logs'/f'{variant}_{stage}_smoke.log')
        command=['train','--run',run,*checkpoint_args]
        if run.exists():command+=['--resume']
        else:command+=['--config',preset,'--stage',stage,'--set',f'micro_batch={micro}']
        execute('ttnet.cli',*command,log=task/'logs'/f'{variant}_{stage}_train.log')


def live_launchers(run):
    pids=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():continue
        try:
            argv=[x.decode() for x in (process/'cmdline').read_bytes().split(b'\0') if x]
            if not any(m in argv for m in ('ttnet.cli','ttnet.training.train')) or '--run' not in argv:continue
            if Path(argv[argv.index('--run')+1]).resolve()==run.resolve():pids.append(int(process.name))
        except (OSError,UnicodeError,ValueError,IndexError):continue
    return pids


def finalize_stage(run,global_checkpoint,local_checkpoint,task):
    checkpoint_args=['--local-checkpoint',local_checkpoint] if local_checkpoint else []
    if not ready_evaluation(run/'evaluations/best_val','metrics.json'):
        execute('ttnet.evaluation.evaluate','--run',run,'--dataset',DATA,'--global-checkpoint',global_checkpoint,
                *checkpoint_args,'--amp',log=task/'logs'/f'{run.name}_evaluate.log')
    if not ready_evaluation(run/'exports','manifest.json'):
        execute('ttnet.evaluation.export','--run',run,'--dataset',DATA,'--global-checkpoint',global_checkpoint,
                *checkpoint_args,log=task/'logs'/f'{run.name}_export.log')
    if not ready_evaluation(run/'evaluations/benchmark','metrics.json'):
        execute('ttnet.evaluation.benchmark','--run',run,'--dataset',DATA,'--global-checkpoint',global_checkpoint,
                *checkpoint_args,'--amp',log=task/'logs'/f'{run.name}_benchmark.log')


def report(task,paths,records):
    import csv
    dest=PROJECT/'out/results'/task.name;dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    sampling=json.loads((task/'shared_sampling/manifest.json').read_text())
    expected={entry['epoch']:entry for entry in sampling['epochs']}
    for variant,stages in paths.items():
        for stage,run in stages.items():
            for path in sorted((run/'logs').glob('attempt_*/epochs.jsonl')):
                for epoch in jsonl(path):
                    actual=epoch.get('sampling') or {};reference=expected[epoch['epoch']]
                    if any(actual.get(key)!=reference[key] for key in ('sample_sha256','count','sources')):
                        raise ValueError(f'unpaired sampling manifest: {run}:{epoch["epoch"]}')
            metric=json.loads((run/'evaluations/best_val/metrics.json').read_text())
            bench=json.loads((run/'evaluations/benchmark/metrics.json').read_text())
            state=json.loads((run/'run.json').read_text())
            exports=json.loads((run/'exports/manifest.json').read_text())
            if not all(x['parity']['passed'] for x in exports['models']):raise ValueError('missing ONNX parity')
            state.update(status='completed',completed=now());save(run/'run.json',state)
            save(records[variant][stage]/'run.json',state)
            for tol,m in metric['center'].items():
                rows.append(dict(variant=variant,stage=stage,eval_id='pp_val_independent_checkpoint_threshold',tolerance_px=tol,
                     precision=m['precision'],recall=m['recall'],f1=m['f1'],tp=m['tp'],fp=m['fp'],fn=m['fn'],
                     e2e_fps=bench['e2e_fps'],best_epoch=state['best_epoch'],run=str(run.relative_to(PROJECT))))
        best_epoch=json.loads((stages['local']/'run.json').read_text())['best_epoch']
        for part in ('val','test'):
            for tracker in ('independent','association'):
                metric=json.loads((dest/variant/f'native_{part}_{tracker}/metrics.json').read_text())
                for tol,m in metric['metric']['localization'].items():
                    rows.append(dict(variant=variant,stage='local',eval_id=f'opentt_{part}_{tracker}_threshold07',tolerance_px=tol,
                        precision=m['precision'],recall=m['recall'],f1=m['f1'],tp=m['tp'],fp=m['fp'],fn=m['fn'],
                        e2e_fps=None,best_epoch=best_epoch,run=str(stages['local'].relative_to(PROJECT))))
            selected=dest/variant/f'native_{part}_independent_checkpoint/metrics.json'
            fixed=dest/variant/f'native_{part}_independent/metrics.json'
            metric=json.loads((selected if selected.exists() else fixed).read_text())
            for tol,m in metric['metric']['localization'].items():
                rows.append(dict(variant=variant,stage='local',eval_id=f'opentt_{part}_independent_checkpoint',tolerance_px=tol,
                    precision=m['precision'],recall=m['recall'],f1=m['f1'],tp=m['tp'],fp=m['fp'],fn=m['fn'],
                    e2e_fps=None,best_epoch=best_epoch,run=str(stages['local'].relative_to(PROJECT))))
        from ttnet.evaluation.metrics import aggregate
        for tracker in ('independent','association'):
            directory=dest/variant/f'pp_{tracker}'
            scored=[r for path in sorted(directory.glob('*_predictions.jsonl')) for r in jsonl(path)]
            for tol in (10,50):
                m=aggregate([r['label'] for r in scored],[r.get('prediction') for r in scored],tol)
                rows.append(dict(variant=variant,stage='local',eval_id=f'pp_val_{tracker}_threshold07',tolerance_px=tol,
                    precision=m['precision'],recall=m['recall'],f1=m['f1'],tp=m['tp'],fp=m['fp'],fn=m['fn'],
                    e2e_fps=None,best_epoch=best_epoch,run=str(stages['local'].relative_to(PROJECT))))
    with (dest/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# PP + native OpenTT ball A/B comparison','','Both sources 50/50, effective batch16, seed42. PP selects checkpoints; game_5 is secondary. Historical PP-only batch8 is a separate reference.','',
           '| Variant | Stage | F1@10px | F1@50px | PP E2E FPS |','|---|---|---:|---:|---:|']
    for variant,stages in paths.items():
        for stage,run in stages.items():
            m=json.loads((run/'evaluations/best_val/metrics.json').read_text());b=json.loads((run/'evaluations/benchmark/metrics.json').read_text())
            lines.append(f'| {variant} | {stage} | {m["center"]["10"]["f1"]:.4%} | {m["center"]["50"]["f1"]:.4%} | {b["e2e_fps"]:.2f} |')
        lines += ['',f'- {variant}: [PP association errors]({variant}/pp_association/index.html), [native game_5 independent errors]({variant}/native_val_independent/index.html), [native game_5 association errors]({variant}/native_val_association/index.html)',
                  f'- {variant}: [feature compatibility]({variant}/feature_compatibility.json), [locked test independent]({variant}/native_test_independent/index.html), [locked test association]({variant}/native_test_association/index.html)']
    lines += ['','## Domain and tracker comparison','','| Variant | Evaluation | F1@10px | F1@50px |','|---|---|---:|---:|']
    for variant in paths:
        for scope in sorted({r['eval_id'] for r in rows if r['variant']==variant and 'checkpoint_threshold' not in r['eval_id']}):
            selected={str(r['tolerance_px']):r for r in rows if r['variant']==variant and r['eval_id']==scope}
            lines.append(f'| {variant} | {scope} | {selected["10"]["f1"]:.4%} | {selected["50"]["f1"]:.4%} |')
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    from ttnet.results import write_results
    write_results()
    return dest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args();task=args.task.resolve();task.mkdir(parents=True,exist_ok=True)
    calibration=json.loads(args.calibration.read_text());audit=json.loads(args.audit.read_text())
    if audit['status']!='passed_screening':raise ValueError('source overlap candidates need resolution before training')
    configured_dataset=Path(load_config(PROJECT/'configs/experiments/ball_mixed_baseline.json')[0]['data'])
    if sha(configured_dataset/'metadata/dataset.json')!=audit['source_manifest_sha256']:
        raise ValueError('source audit and training dataset identity differ')
    if calibration.get('implementation_sha256') and any(sha(SRC/name)!=value for name,value in calibration['implementation_sha256'].items()):
        raise ValueError('calibration implementation changed; repeat the selected-pair check')
    if (task/'task.json').exists():
        manifest=json.loads((task/'task.json').read_text())
    else:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
        manifest=dict(status='prepared',created=now(),calibration=calibration,source_audit=str(args.audit),
                      runs={v:{s:str(RUNS/f'{stamp}_mixed_ball_{v}_{s}') for s in ('global','local')} for v in ('A','B')})
        save(task/'task.json',manifest)
    paths={v:{s:Path(p) for s,p in stages.items()} for v,stages in manifest['runs'].items()}
    prepare_sampling(task,configured_dataset)
    presets={v:PROJECT/f'configs/experiments/ball_mixed_{"baseline" if v=="A" else "hr_se"}.json' for v in paths}
    records={v:{s:PROJECT/'experiments/records'/p.name for s,p in stages.items()} for v,stages in paths.items()}
    for v,stages in paths.items():
        for s,run in stages.items():
            record=records[v][s];record.mkdir(parents=True,exist_ok=True)
            if not (record/'purpose.md').exists():
                (record/'purpose.md').write_text(f'# {v} {s}\n\nPurpose: approved PP+OpenTT native ball A/B. Effective batch16, seed42.\nA residual FPN; B changes only Global high-resolution+SE. Matching feature shapes128/256.\n')
            if (run/'run.json').exists():
                save(record/'run.json',json.loads((run/'run.json').read_text()))
            elif not (record/'run.json').exists():
                save(record/'run.json',dict(status='queued',stage=s,variant=v,run=str(run),task=str(task),created=now()))
                planned,_=load_config(presets[v]);planned.update(stage=s,micro_batch=calibration['micro_batches'][v])
                save(record/'planned_config.json',planned)
    destination=PROJECT/'out/results'/task.name
    try:
        manifest.update(status='training',started=manifest.get('started',now()),coordinator_pid=os.getpid(),updated=now());save(task/'task.json',manifest)
        if calibration['parallel']:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(train_variant,v,paths[v],presets[v],calibration['micro_batches'][v],task) for v in paths]
                for future in futures:future.result()
        else:
            for v in paths:train_variant(v,paths[v],presets[v],calibration['micro_batches'][v],task)
        manifest.update(status='evaluating',updated=now());save(task/'task.json',manifest)
        from ttnet.evaluation.native_ball import evaluate_native,system_config
        from ttnet.evaluation.candidate_audit import audit as candidate_audit
        dataset=Path(load_config(presets['A'])[0]['data'])
        for v,stages in paths.items():
            g=stages['global']/'checkpoints/best.pt';l=stages['local']/'checkpoints/best.pt'
            for stage,run in stages.items():finalize_stage(run,g,l if stage=='local' else None,task)
            out=destination/v;out.mkdir(parents=True,exist_ok=True)
            if not (out/'feature_compatibility.json').exists():compatibility(g,l,out/'feature_compatibility.json')
            for tracker in ('independent','association'):
                target=out/f'pp_{tracker}'
                if not ready_evaluation(target,'summary.json'):candidate_audit(system_config(g,l,tracker),DATA,target,device='cuda')
                target=out/f'native_val_{tracker}'
                if not ready_evaluation(target,'metrics.json'):evaluate_native(dataset,g,l,target,'val',tracker)
            selected=system_config(g,l,checkpoint_thresholds=True)
            if (selected['global_threshold'],selected['local_threshold'])!=(.7,.7):
                target=out/'native_val_independent_checkpoint'
                if not ready_evaluation(target,'metrics.json'):evaluate_native(dataset,g,l,target,'val','independent',True)
        # All thresholds and models are frozen before either variant sees test scores.
        save(task/'selection_lock.json',dict(policy='PP-only checkpoint/threshold selection; no retraining after test',
             weights={v:{s:sha(run/'checkpoints/best.pt') for s,run in stages.items()} for v,stages in paths.items()}))
        for v,stages in paths.items():
            g=stages['global']/'checkpoints/best.pt';l=stages['local']/'checkpoints/best.pt'
            for tracker in ('independent','association'):
                target=destination/v/f'native_test_{tracker}'
                if not ready_evaluation(target,'metrics.json'):evaluate_native(dataset,g,l,target,'test',tracker)
            selected=system_config(g,l,checkpoint_thresholds=True)
            if (selected['global_threshold'],selected['local_threshold'])!=(.7,.7):
                target=destination/v/'native_test_independent_checkpoint'
                if not ready_evaluation(target,'metrics.json'):evaluate_native(dataset,g,l,target,'test','independent',True)
        result=report(task,paths,records)
        manifest.update(status='completed',completed=now(),result=str(result));save(task/'task.json',manifest)
        print(json.dumps(manifest),flush=True)
    except BaseException as error:
        manifest.update(status='failed',failed=now(),error=repr(error));save(task/'task.json',manifest)
        raise


if __name__=='__main__':main()
