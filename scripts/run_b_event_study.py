"""Resume the recorded B-locator, three-arm frozen event study.

Run only one coordinator for a task. Existing F0 training can be adopted with
--wait-f0; all subsequent commands and exit codes are persisted in the task.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from ttnet.common import PROJECT, now, save


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def coordinate(task, wait_f0=False):
    task = task.resolve()
    document = task/'task.json'
    lock = (task/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(document)
    state['coordinator_started'] = now()
    state['coordinator_command'] = sys.argv
    save(document, state)

    def command(key, args):
        previous = state.setdefault('steps', {}).get(key, {})
        if previous.get('status') == 'completed':
            return
        # Incomplete attempts have a log but no receipt; preserve both kinds.
        existing = list((task/'logs').glob(key+'_*.log')) + list((task/'logs').glob(key+'_*.json'))
        attempts = max((int(p.stem.rsplit('_', 1)[1]) for p in existing), default=0) + 1
        log = task/'logs'/f'{key}_{attempts:03d}.log'
        log.parent.mkdir(exist_ok=True)
        cmd = [sys.executable, '-u', *map(str, args)]
        entry = dict(status='running', command=cmd, started=now(), log=str(log))
        state['status'] = key
        state['steps'][key] = entry
        save(document, state)
        print(f'{key}: {cmd}', flush=True)
        with log.open('w') as stream:
            code = subprocess.call(cmd, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
        entry.update(exit_code=code, finished=now(), status='completed' if code == 0 else 'failed')
        save(log.with_suffix('.json'), entry)
        save(document, state)
        if code:
            raise RuntimeError(f'{key} exited {code}; see {log}')

    try:
        f0 = PROJECT/state['runs']['F0']
        if wait_f0:
            print('Waiting for the already-running F0 training.', flush=True)
            while read(f0/'run.json').get('status') not in ('trained_pending_eval', 'failed'):
                time.sleep(10)
            if read(f0/'run.json').get('status') == 'failed':
                raise RuntimeError('existing F0 training failed')
            state['steps'].setdefault('F0_train', {}).update(status='completed', adopted=True, finished=now())
            save(document, state)
        else:
            args = ['-m', 'ttnet.cli', 'train', '--run', state['runs']['F0']]
            args += ['--resume'] if f0.exists() else ['--config', state['configs']['F0']]
            command('F0_train', args)

        for arm in ('geometry', 'concat'):
            smoke_run = PROJECT/(state['runs'][arm]+'_smoke')
            if read(smoke_run/'smoke.json').get('passed'):
                state['steps'][arm+'_smoke'] = dict(status='completed', adopted=True, run=str(smoke_run))
                save(document, state)
                continue
            if smoke_run.exists():
                number = 2
                while smoke_run.with_name(smoke_run.name+f'_retry{number:03d}').exists():
                    number += 1
                smoke_run = smoke_run.with_name(smoke_run.name+f'_retry{number:03d}')
            command(arm+'_smoke', ['-m', 'ttnet.cli', 'train', '--config', state['configs'][arm],
                                   '--run', smoke_run, '--smoke'])

        command('table_cache', ['-m', 'ttnet.scene.augment_event_cache', '--config', state['configs']['geometry'],
                                '--base', state['base_cache'], '--batch', str(state.get('table_cache_batch', 1)), '--device', 'cuda',
                                '--output', task/'table_cache.json'])
        for arm in ('geometry', 'concat'):
            run = PROJECT/state['runs'][arm]
            args = ['-m', 'ttnet.cli', 'train', '--run', run]
            args += ['--resume'] if run.exists() else ['--config', state['configs'][arm]]
            command(arm+'_train', args)

        for arm in ('F0', 'geometry', 'concat'):
            run = PROJECT/state['runs'][arm]
            output = run/'evaluations/continuous'
            completed = state['steps'].get(arm+'_evaluate', {})
            if completed.get('status') == 'completed':
                cmd = completed['command']
                output = Path(cmd[cmd.index('--output')+1])
                if not (output/'metrics.json').is_file():
                    raise FileNotFoundError(f'completed evaluation lost its metrics: {output}')
            if not (output/'metrics.json').exists():
                if output.exists():
                    count = len(list((run/'evaluations').glob('continuous*'))) + 1
                    output = output.with_name(f'continuous_attempt{count:03d}')
                command(arm+'_evaluate', ['-m', 'ttnet.events.comparison', 'evaluate',
                                          '--checkpoint', run/'checkpoints/best.pt',
                                          '--annotations', 'data/annotations/pp_bounce_confirmed_v1.json',
                                          '--output', output, '--scheme', arm, '--device', 'cuda',
                                          '--use-checkpoint-threshold'])
            state.setdefault('evaluations', {})[arm] = str(output/'metrics.json')
            save(document, state)
        state.update(status='evaluated_pending_analysis', finished=now())
        save(document, state)
    except BaseException as error:
        state.update(status='failed', error=repr(error), stopped=now())
        save(document, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--wait-f0', action='store_true')
    args = parser.parse_args()
    coordinate(args.task, args.wait_f0)
