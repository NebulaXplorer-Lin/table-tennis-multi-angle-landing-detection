"""Persistent serial dispatcher for the two authorized table training routes."""
from __future__ import annotations

import argparse
import fcntl
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from ttnet.common import PROJECT, save, sha
from ttnet.configuration import load_config


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def dispatch(task, device):
    task = task.resolve()
    marker = task/'task.json'
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest['status']=='completed':
            return
    else:
        task.mkdir(parents=True,exist_ok=True)
        manifest = dict(status='queued',created=timestamp(),device=device,policy='serial, shared_B then independent_cnn; no event-head training',
                        routes={name:dict(status='queued',config=str(PROJECT/f'configs/table/{name}.json'),
                                          run=str(task/name),log=str(task/f'{name}.log'))
                                for name in ('shared_B','independent_cnn')})
        for path in (PROJECT/'configs/experiments').glob('event_table_*.json'):
            config,_ = load_config(path)
            if config.get('table_frontend'):
                route = 'shared_B' if config['table_frontend']['source']=='shared_global' else 'independent_cnn'
                config['table_frontend']['checkpoint'] = str(task/route/'best.pt')
            save(task/'event_configs'/path.name,config)
    manifest.update(status='running',pid=os.getpid(),supervisor_status='active',updated=timestamp())
    save(marker,manifest)
    for name,route in manifest['routes'].items():
        if route.get('status')=='cancelled':
            continue
        run = Path(route['run'])
        state = json.loads((run/'run.json').read_text()) if (run/'run.json').exists() else {}
        pid = route.get('pid')
        # Reattach an orphaned B trainer without stopping or duplicating it.
        if route.get('status')=='running' and pid and pid!=os.getpid():
            command_path=Path(f'/proc/{pid}/cmdline')
            matches=command_path.exists() and str(run).encode() in command_path.read_bytes()
            if matches:
                route['reattached']=timestamp();save(marker,manifest)
                while command_path.exists():
                    try:
                        process_state=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
                    except FileNotFoundError:
                        break
                    if process_state=='Z':
                        break
                    time.sleep(5)
                state=json.loads((run/'run.json').read_text()) if (run/'run.json').exists() else {}
                if state.get('status')!='trained_evaluated':
                    route['status']='failed';manifest['status']='failed';save(marker,manifest)
                    raise RuntimeError(f'attached trainer {name} stopped before completion')
        if state.get('status')=='trained_evaluated':
            route.update(status='completed',best_sha256=sha(run/'best.pt'));save(marker,manifest)
            continue
        module='ttnet.scene.train_light' if name=='independent_cnn' else 'ttnet.scene.train_visual'
        command = [sys.executable,'-u','-m',module,'--config',route['config'],'--run',str(run),'--device',device,'--train']
        if (run/'last.pt').exists():
            command.append('--resume')
        route.update(status='running',command=command,started=timestamp())
        manifest['active_route']=name;manifest['updated']=timestamp();save(marker,manifest)
        with Path(route['log']).open('a') as stream:
            child = subprocess.Popen(command,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,
                                     env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1'))
            route['pid']=child.pid;save(marker,manifest)
            code = child.wait()
        route.update(exit_code=code,finished=timestamp(),status='completed' if code==0 else 'failed')
        if code:
            manifest.update(status='failed',updated=timestamp());save(marker,manifest)
            raise RuntimeError(f'{name} exited {code}; see {route["log"]}')
        route['best_sha256']=sha(run/'best.pt');save(marker,manifest)
    manifest.update(status='completed',active_route=None,completed=timestamp());save(marker,manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True);p.add_argument('--device',default='cuda')
    args=p.parse_args();args.task.mkdir(parents=True,exist_ok=True)
    with (args.task/'dispatcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dispatch(args.task,args.device)
