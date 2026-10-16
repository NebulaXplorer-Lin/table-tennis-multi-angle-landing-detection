"""Run reproducible paired experiments; save subprocess logs and exit status."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from ttnet.common import CONFIGS, EVENT_DATA, PROJECT, RUNS, identity, now, save


def command(module,args,log):
    cmd=[sys.executable,'-u','-m',module,*map(str,args)]
    print(' '.join(cmd),flush=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    if log.exists():
        original=log
        index=2
        while log.exists():
            log=original.with_name(f'{original.stem}_{index:03d}{original.suffix}');index+=1
    with log.open('w') as stream:
        result=subprocess.run(cmd,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
    save(log.with_suffix('.exit.json'),dict(command=cmd,exit_code=result.returncode,finished=now()))
    if result.returncode:raise RuntimeError(f'{module} failed ({result.returncode}); see {log}')


def smoke_corpus(dataset,dest):
    # Smoke may use only already prepared game_1; its identity is explicitly pilot.
    game=dataset/'game_1'
    if not (game/'metadata.json').exists():raise ValueError('game_1 preparation is not complete')
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'game_1').symlink_to(game.resolve(),target_is_directory=True)
    reference=dataset/'frame_store.json'
    if reference.is_file():
        ref=json.loads(reference.read_text())
        ref['path']=os.path.relpath((dataset/ref['path']).resolve(),dest)
        save(dest/'frame_store.json',ref)
    split=json.loads((dataset/'split.json').read_text());save(dest/'split.json',split)
    save(dest/'dataset.json',dict(dataset_id=identity(dict(pilot=True,source=json.loads((game/'metadata.json').read_text()))),
                                  complete=False,pilot=True,split_id=split['split_id'],prepared=['game_1']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['smoke','train'],required=True)
    p.add_argument('--task',type=Path)
    p.add_argument('--dataset',type=Path,default=EVENT_DATA)
    p.add_argument('--config',type=Path,default=CONFIGS/'event.json')
    p.add_argument('--smoke-task',type=Path)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    stamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S')
    task=a.task or RUNS/f'{stamp}_event_{a.phase}_pipeline'
    task.mkdir(parents=True,exist_ok=a.resume)
    save(task/'run.json',dict(status='running',phase=a.phase,started=now()))
    try:
        if a.phase=='smoke':
            dataset=task/'pilot';smoke_corpus(a.dataset,dataset)
            for mode in ('frozen','joint'):
                command('ttnet.events.train',['--mode',mode,'--run',task/f'{mode}_smoke','--dataset',dataset,'--config',a.config,'--smoke'],task/f'{mode}.log')
            save(task/'run.json',dict(status='completed_smoke',finished=now()))
            print(f'Smoke passed: {task}',flush=True);return
        if not a.smoke_task or json.loads((a.smoke_task/'run.json').read_text())['status']!='completed_smoke':
            raise ValueError('--smoke-task must point to completed real-data smoke checks')
        # Wait for the independently running CPU preparation, without touching its files.
        deadline=time.monotonic()+7200
        while not (a.dataset/'dataset.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('dataset preparation did not complete in 2h')
            save(task/'run.json',dict(status='waiting_for_data',phase='train',updated=now()))
            time.sleep(15)
        if not json.loads((a.dataset/'dataset.json').read_text())['complete']:raise ValueError('incomplete dataset')
        runs=({k:Path(v) for k,v in json.loads((task/'runs.json').read_text()).items()}
              if a.resume and (task/'runs.json').exists() else
              {mode:RUNS/f'{stamp}_event25_{mode}' for mode in ('frozen','joint')})
        save(task/'runs.json',{k:str(v) for k,v in runs.items()})
        for mode,run in runs.items():
            current=json.loads((run/'run.json').read_text()) if (run/'run.json').exists() else {}
            if current.get('status')=='completed':continue
            save(task/'run.json',dict(status='training',mode=mode,run=str(run),updated=now()))
            if current.get('status')!='trained_pending_eval':
                options=['--mode',mode,'--run',run,'--dataset',a.dataset,'--config',a.config]
                if run.exists():
                    options.append('--resume')
                    if not (run/'checkpoints/last.pt').exists():options.append('--restart-incomplete')
                name=f'{mode}_train'+('_resume' if run.exists() else '')+'.log'
                command('ttnet.events.train',options,task/name)
            # Validation and independent test are evaluated after best selection; no feedback into training.
            save(task/'run.json',dict(status='evaluating',mode=mode,run=str(run),updated=now()))
            options=['--run',run]
            if (run/'evaluations/best').exists():options.append('--resume')
            name=f'{mode}_eval'+('_resume' if '--resume' in options else '')+'.log'
            command('ttnet.events.evaluate',options,task/name)
        save(task/'run.json',dict(status='completed',runs={k:str(v) for k,v in runs.items()},finished=now()))
    except BaseException as exc:
        save(task/'run.json',dict(status='failed',error=repr(exc),finished=now()))
        raise


if __name__=='__main__':main()
