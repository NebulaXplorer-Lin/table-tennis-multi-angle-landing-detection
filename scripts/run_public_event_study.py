"""Auditable PP154 training / PP-public validation experiments.

Old feature files may be copied only when their complete inference identity is
unchanged except for the corpus split identity. The original caches stay intact.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from ttnet.common import PROJECT, identity, jsonl, now, save, sha
from ttnet.configuration import load_config
from ttnet.events.continuous import attach_pp
from ttnet.events.data import EventCorpus
from ttnet.events.engine import EventSystem, build_cache, cache_identity, feature_cache_path
from ttnet.events.public_validation import attach_public_validation


def read(path):
    return json.loads(Path(path).read_text())


def prepare_cache(config, source, receipt):
    cfg, _ = load_config(config)
    torch.set_num_threads(cfg['torch_threads'])
    corpus = EventCorpus(cfg['dataset'])
    attach_pp(corpus, cfg['pp_annotations'])
    old_corpus_identity = corpus.identity
    attach_public_validation(corpus, cfg['public_validation_root'], cfg)
    split_dir = PROJECT/'experiments/splits'/corpus.split['split_id']
    split_dir.mkdir(parents=True,exist_ok=True)
    for name,value in (('split.json',corpus.split),('audit.json',corpus.public_split_audit)):
        path = split_dir/name
        if path.exists() and read(path) != value:
            raise ValueError('published public split changed: '+str(path))
        if not path.exists():save(path,value)
    system = EventSystem(cfg, torch.device('cuda'))
    expected = cache_identity(system, corpus)
    previous = read(source/'identity.json')
    old_expected = deepcopy(expected)
    old_expected['dataset_id'] = old_corpus_identity
    if previous != old_expected:
        changed = [k for k in previous.keys() | old_expected.keys() if previous.get(k) != old_expected.get(k)]
        raise ValueError('refusing cache reuse; changed inference identity: '+str(changed))
    dest = feature_cache_path(system, corpus)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'identity.json').exists() and read(dest/'identity.json') != expected:
        raise ValueError('destination identity changed')
    save(dest/'identity.json', expected)
    provenance_path = dest/'public_split_derivation.json'
    provenance = dict(schema='public_split_cache_derivation_v1', source=str(source.resolve()),
                      source_identity_sha256=sha(source/'identity.json'),
                      source_identity=previous, target_identity=expected,
                      split_audit=corpus.public_split_audit,
                      implementation_sha256=sha(__file__), reused={}, added={})
    videos = [vid for part in ('train','val') for vid in corpus.split[part]]
    for vid in videos:
        if vid.startswith('public_'):
            continue
        source_npz = source/f'{vid}.npz'
        with np.load(source_npz, allow_pickle=False) as arrays:
            ids = arrays['frame_ids'].tolist()
            if ids != sorted(corpus.videos[vid]['frames']):
                raise ValueError('reused cache frame coverage changed: '+vid)
        if [r['frame_id'] for r in jsonl(source/f'{vid}.jsonl')] != ids:
            raise ValueError('reused cache record coverage changed: '+vid)
        hashes = {}
        for suffix in ('.npz','.jsonl'):
            src, target = source/(vid+suffix), dest/(vid+suffix)
            digest = sha(src)
            if target.exists():
                if sha(target) != digest:
                    raise ValueError('cache reuse destination differs: '+str(target))
            else:
                temp = target.with_suffix(target.suffix+'.copying')
                shutil.copyfile(src,temp)
                if sha(temp) != digest:
                    raise ValueError('cache copy hash mismatch')
                temp.replace(target)
            hashes[suffix] = digest
        provenance['reused'][vid] = dict(frames=len(ids),sha256=hashes)
    provenance['status'] = 'extracting_public'
    save(provenance_path,provenance)
    # build_cache skips only the copied exact files; public inference uses no GT.
    build_cache(system,corpus,dest)
    for vid in videos:
        if not vid.startswith('public_'):
            continue
        with np.load(dest/f'{vid}.npz',allow_pickle=False) as arrays:
            if arrays['frame_ids'].tolist() != sorted(corpus.videos[vid]['frames']):
                raise ValueError('incomplete public cache: '+vid)
        provenance['added'][vid] = dict(frames=len(corpus.videos[vid]['frames']),
            sha256={s:sha(dest/(vid+s)) for s in ('.npz','.jsonl')})
    provenance.update(status='completed',finished=now())
    save(provenance_path,provenance)
    save(receipt,dict(status='completed',cache=str(dest),identity_sha256=sha(dest/'identity.json'),
                      derivation=str(provenance_path),derivation_sha256=sha(provenance_path)))
    print(json.dumps(dict(cache=str(dest),reused=len(provenance['reused']),public=len(provenance['added']))),flush=True)


def coordinate(task):
    task = task.resolve()
    lock = (task/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = read(task/'task.json')

    def command(key,args):
        previous = state.setdefault('steps',{}).get(key,{})
        if previous.get('status') == 'completed':
            return
        logs = task/'logs';logs.mkdir(exist_ok=True)
        index = len(list(logs.glob(key+'_*.json')))+1
        while (logs/f'{key}_{index:03d}.log').exists():
            index += 1
        log = logs/f'{key}_{index:03d}.log'
        argv = [sys.executable,'-u',*map(str,args)]
        entry = dict(status='running',started=now(),command=argv,log=str(log))
        state['status'] = key;state['steps'][key] = entry
        save(task/'task.json',state)
        print(key,flush=True)
        with log.open('w') as stream:
            code = subprocess.call(argv,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
        entry.update(status='completed' if code==0 else 'failed',exit_code=code,finished=now())
        save(log.with_suffix('.json'),entry);save(task/'task.json',state)
        if code:
            raise RuntimeError(f'{key} exited {code}: {log}')

    try:
        for name, spec in state['caches'].items():
            command('cache_'+name,['-m','scripts.run_public_event_study','prepare-cache',
                '--config',spec['config'],'--source',spec['source'],'--receipt',spec['receipt']])
        for spec in state['experiments']:
            key = spec['name'];run = PROJECT/spec['run']
            smoke = Path(str(run)+'_smoke')
            if (smoke/'smoke.json').exists() and read(smoke/'smoke.json').get('passed'):
                state.setdefault('steps',{}).setdefault(key+'_smoke',dict(status='completed',adopted=True,run=str(smoke)))
                save(task/'task.json',state)
            else:
                attempt=1
                while smoke.exists():
                    attempt+=1;smoke=Path(str(run)+f'_smoke_attempt{attempt:03d}')
                command(key+'_smoke',['-m','ttnet.cli','train','--config',spec['config'],'--run',smoke,'--smoke'])
            argv = ['-m','ttnet.cli','train','--run',run]
            argv += ['--resume'] if run.exists() else ['--config',spec['config']]
            command(key+'_train',argv)
            cache = read(state['caches'][spec['cache']]['receipt'])['cache']
            output=Path(spec.get('evaluation',str(run/'evaluations/public')))
            if state.get('steps',{}).get(key+'_evaluate',{}).get('status')!='completed':
                attempt=1
                while output.exists():
                    attempt+=1;output=run/f'evaluations/public_attempt{attempt:03d}'
                spec['evaluation']=str(output)
                save(task/'task.json',state)
            command(key+'_evaluate',['-m','scripts.evaluate_public_event',
                '--checkpoint',run/'checkpoints/best.pt','--output',output,
                '--public','data/pp_public_data','--cache',cache,'--device','cuda'])
            metric=read(output/'metrics.json')
            if not metric.get('cache_online_parity',{}).get('passed'):
                raise ValueError('public cache/online parity failed: '+str(output))
            command(key+'_benchmark',['-m','ttnet.cli','_event_benchmark','--run',run,'--device','cuda'])
        state.update(status='evaluated_pending_analysis',finished=now())
        save(task/'task.json',state)
    except BaseException as error:
        state.update(status='failed',error=repr(error),stopped=now())
        save(task/'task.json',state)
        raise


def main():
    os.environ.setdefault('YOLO_CONFIG_DIR',str(PROJECT/'out/yolo_setup/event_table_config'))
    os.environ.setdefault('YOLO_OFFLINE','true')
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('prepare-cache');c.add_argument('--config',type=Path,required=True)
    c.add_argument('--source',type=Path,required=True);c.add_argument('--receipt',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--task',type=Path,required=True)
    a=p.parse_args()
    if a.action=='prepare-cache':prepare_cache(a.config,a.source,a.receipt)
    else:coordinate(a.task)


if __name__=='__main__':main()
