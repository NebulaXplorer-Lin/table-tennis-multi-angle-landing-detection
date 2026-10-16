"""Durable single-GPU queue for the frozen A/B/C/D study, with fail-fast checks."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from ttnet.common import PROJECT, identity, save, sha
from ttnet.configuration import load_config


def prepare(root):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);steps=[]
    annotation=None
    for scheme in 'ABCD':
        cfg,_=load_config(f'configs/experiments/event_comparison_{scheme}.json')
        cfg['scheme']='event_comparison_'+scheme
        save(root/f'configs/{scheme}.json',cfg)
        annotation=cfg['pp_annotations']
        smoke=PROJECT/'out/experiments'/f'{root.name}_{scheme}_smoke'
        run=PROJECT/'out/experiments'/f'{root.name}_{scheme}'
        prefix=[sys.executable,'-u','-m','ttnet.cli','train','--config',str(root/f'configs/{scheme}.json'),'--device','cuda']
        steps.append(dict(name=f'{scheme}_smoke',kind='smoke',scheme=scheme,run=str(smoke),command=prefix+['--run',str(smoke),'--smoke']))
    # Extract A/C features once and run both heads before the longer B/D context.
    for scheme in 'ACBD':
        run=PROJECT/'out/experiments'/f'{root.name}_{scheme}'
        steps.append(dict(name=f'{scheme}_train',kind='train',scheme=scheme,run=str(run),command=[sys.executable,'-u','-m','ttnet.cli','train',
            '--config',str(root/f'configs/{scheme}.json'),'--device','cuda','--run',str(run)]))
    for scheme in 'ABCD':
        run=PROJECT/'out/experiments'/f'{root.name}_{scheme}'
        steps.append(dict(name=f'{scheme}_evaluate',kind='evaluate',scheme=scheme,run=str(run),command=[sys.executable,'-u','-m','ttnet.cli','evaluate','--run',str(run),'--device','cuda']))
    historical=PROJECT/'ttnet/output/events/20261014-181231_event25_frozen/checkpoints/best.pt'
    steps.append(dict(name='A0_evaluate',kind='evaluate',scheme='A0',command=[sys.executable,'-u','-m','ttnet.events.comparison','evaluate',
        '--scheme','A0','--checkpoint',str(historical),'--annotations',annotation,'--output',str(root/'A0_continuous'),'--device','cuda']))
    paths=[PROJECT/'out/experiments'/f'{root.name}_{k}'/'evaluations/continuous/metrics.json' for k in 'ABCD']+[root/'A0_continuous/metrics.json']
    steps.append(dict(name='comparison',kind='report',command=[sys.executable,'-u','-m','ttnet.events.comparison','report','--results',*map(str,paths),'--output',str(root/'comparison')]))
    # Only seed42 is automatic here; the multi-seed continuation depends on the
    # completed comparison and per-view review, not on an untrained score.
    sources=[p for folder in ('events','models','tracking') for p in (PROJECT/'ttnet/src'/folder).glob('*.py')]
    sources += [PROJECT/'ttnet/src/cli.py',PROJECT/'ttnet/src/configuration.py',Path(annotation)]
    plan=dict(status='ready',root=str(root),steps=steps,seed=42,annotation=annotation,annotation_sha256=sha(annotation),
        source_hashes={str(p):sha(p) for p in sources},
        remaining_after_seed42=['A and best new method seeds43/44','OpenTTGames test and PP public final generalization','final per-view and speed review'],
        metric_note='PP exhaustive Bounce confirmed by user; spatial reference is original event-frame ball GT, excluding missing coordinates; camera buckets are coarse visual metadata')
    save(root/'plan.json',plan);save(root/'status.json',dict(status='queued',next_step=steps[0]['name']))
    return plan


def run(root):
    root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text());completed=[]
    for step in plan['steps']:
        changed=[p for p,h in plan['source_hashes'].items() if sha(p)!=h]
        if changed:raise ValueError(f'study sources/labels changed after launch: {changed}')
        status=dict(status='running',pid=os.getpid(),step=step['name'],kind=step['kind'],run=step.get('run'),completed=completed)
        save(root/'status.json',status)
        print('START',step['name'],flush=True)
        log=root/f"logs/{step['name']}.log";log.parent.mkdir(exist_ok=True)
        with log.open('w') as stream:
            child=subprocess.Popen(step['command'],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
            save(root/'status.json',dict(status,child_pid=child.pid,log=str(log)))
            code=child.wait()
        save(root/f"steps/{step['name']}.json",dict(step,exit_code=code,finished=datetime.now().astimezone().isoformat()))
        if code:
            save(root/'status.json',dict(status='failed',step=step['name'],exit_code=code,log=str(log),completed=completed,pid=os.getpid()))
            raise RuntimeError(f"{step['name']} failed; see {log}")
        completed.append(step['name']);print('DONE',step['name'],flush=True)
    save(root/'status.json',dict(status='seed42_complete_pending_multiseed',completed=completed,pid=os.getpid()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    if a.action=='prepare':print(prepare(a.root)['root'])
    else:
        try:run(a.root)
        except BaseException as error:
            status_path=a.root/'status.json';state=json.loads(status_path.read_text()) if status_path.exists() else {}
            if state.get('status')!='failed':save(status_path,dict(state,status='failed',error=repr(error)))
            raise


if __name__=='__main__':main()
