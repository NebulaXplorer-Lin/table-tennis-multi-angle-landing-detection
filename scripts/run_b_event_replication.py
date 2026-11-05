"""Run the preregistered seed43/44 replications after seed42 passes its audit."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from ttnet.common import identity, now, save, sha
from scripts.report_b_event_study import ARMS, PROTOCOL_KEYS, resolve


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def current_cache_identity(config):
    """Call the official identity function without constructing models or a GPU."""
    from ttnet.events.engine import cache_identity
    corpus = SimpleNamespace(identity=config['dataset_id'])
    for name, backend in config.get('pixel_backends', {}).items():
        if name in ('frame_store', 'pp_frame_store'):
            setattr(corpus, name, SimpleNamespace(is_video=True, backend_identity=lambda value=backend:value))
    return cache_identity(SimpleNamespace(config=config), corpus)


class ReplicationAudit:
    def __init__(self):
        self.sources = {}
        self.mutable_source_exceptions = []
        self.cache_receipt_exceptions = []

    def digest(self, path):
        path = resolve(path)
        require(path.is_file(), 'missing audit artifact: '+str(path))
        if str(path) not in self.sources:
            self.sources[str(path)] = sha(path)
        return self.sources[str(path)]

    def report_sources(self, task, analysis, required):
        snapshots = {resolve(path):digest for path,digest in analysis.get('sources_sha256', {}).items()}
        require(bool(snapshots), 'report has no source hash snapshot: '+str(task))
        require(set(required) <= snapshots.keys(), 'report source snapshot omits required inputs: '+str(task))
        for path, expected in snapshots.items():
            actual = self.digest(path)
            if path == task/'task.json':
                self.mutable_source_exceptions.append(dict(
                    path=str(path),snapshot_sha256=expected,current_sha256=actual,
                    reason='Only this report-owned task.json may append orchestration state. Seeds, run/evaluation bindings and config files are checked separately.'))
            else:
                require(actual == expected, 'report source changed after audit: '+str(path))


def validate_table_cache(cache, audit):
    cache = resolve(cache); identity_path = cache/'identity.json'; derivation_path = cache/'derivation.json'
    ident = read(identity_path); derivation = read(derivation_path)
    identity_sha = audit.digest(identity_path); derivation_sha = audit.digest(derivation_path)
    require(derivation.get('status') == 'completed', 'table cache derivation is not completed')
    require(derivation.get('table_memory_format') == 'contiguous_nchw', 'retired or unknown table cache memory layout')
    require(type(derivation.get('table_inference_batch')) is int and derivation['table_inference_batch'] == 1,
            'table cache must use online-equivalent batch1')
    require(ident == derivation.get('target_identity'), 'table identity.json differs from derivation.target_identity')
    require(ident.get('schema') == 4 and cache.name == identity(ident)[:20], 'table cache identity/path hash mismatch')
    require(ident.get('config', {}).get('amp') is True and derivation.get('table_inference_precision') == 'AMP',
            'table cache precision differs from the fixed AMP experiment')
    require(derivation.get('implementation_sha256') == audit.digest(PROJECT/'ttnet/src/scene/augment_event_cache.py'),
            'table cache augmentation implementation is not the current source')
    visual = ident.get('table_visual', {}); implementations = visual.get('implementation', {})
    require(implementations.get('scene/visual_cache.py') == audit.digest(PROJECT/'ttnet/src/scene/visual_cache.py'),
            'table cache helper implementation is not the current source')
    for name,digest in implementations.items():
        require(audit.digest(PROJECT/'ttnet/src'/name) == digest, 'table visual implementation changed: '+name)
    base = resolve(derivation['base']); base_identity = read(base/'identity.json')
    require(base_identity == derivation.get('base_identity') and base.name == identity(base_identity)[:20],
            'table cache base identity/path mismatch')
    audit.digest(base/'identity.json')
    sources = derivation.get('sources', {}); completed = derivation.get('completed', {})
    require(len(sources) == 159 and set(sources) == set(completed), 'table cache must have all159 video completion markers')
    require(set(sources) == {p.stem for p in base.glob('*.npz')}, 'table cache derivation omits base video arrays')
    require(set(sources) == {p.stem for p in base.glob('*.jsonl')}, 'table cache derivation omits base video records')
    for vid,pair in sources.items():
        marker = completed[vid]
        require(marker.get('source') == pair and marker.get('frames', 0) > 0,
                'invalid table cache completion marker: '+vid)
        require(marker.get('ball_arrays_exact') is True and marker.get('ball_records_exact_excluding_table') is True,
                'table cache did not verify ball preservation: '+vid)
        for key,suffix in (('npz','.npz'),('records','.jsonl')):
            require(audit.digest(base/(vid+suffix)) == pair.get(key), 'base cache source changed: '+vid+suffix)
            require(audit.digest(cache/(vid+suffix)) == marker.get('outputs', {}).get(key),
                    'table cache output changed: '+vid+suffix)
    return dict(path=str(cache),identity=ident,identity_sha256=identity_sha,derivation_sha256=derivation_sha,
                base=str(base),videos=len(sources),table_memory_format='contiguous_nchw',table_inference_batch=1,table_inference_precision='AMP',
                augmentation_sha256=derivation['implementation_sha256'],helper_sha256=implementations['scene/visual_cache.py'])


def exact_original_F0(parent, run, data, audit):
    """The sole pre-receipt ball-only exception, backed by its saved exact run."""
    screen = read(parent/'task.json'); original = resolve(screen['runs']['F0'])
    require(run == original, 'missing feature-cache receipt is not the original seed42 F0')
    old_path = parent/'analysis.json'; old = read(old_path)['arms']['F0']; audit.digest(old_path)
    parity = old['parity']; totals = parity['totals']
    require(parity.get('status') == 'exact' and parity.get('fully_consistent') is True and totals.get('frames') == 6292,
            'original seed42 F0 lacks exact6292-frame cache/online evidence')
    require(data['parity'].get('fully_consistent') is True and data['parity']['totals'].get('frames') == 6292,
            'reused seed42 F0 report is not exact6292-frame evidence')
    for name in ('best', 'last'):
        require(old['checkpoints'][name]['sha256'] == data['checkpoints'][name]['sha256'],
                'reused F0 checkpoint differs from the original exact screen')
    audit.cache_receipt_exceptions.append(dict(seed=42,arm='F0',run=str(run),evidence=str(old_path),
        reason='Original ball-only training predates feature_cache.json. Both the original and reused report prove exact6292-frame parity and the checkpoint hashes are unchanged.'))


def reconstructed_geo42(task, state, run, config, data, cache, corrected, audit):
    """Explicit reconstruction for the one trainer started before receipts landed."""
    require(task == corrected and resolve(state['runs']['geometry']) == run,
            'missing receipt does not belong to the corrected seed42 geometry run')
    step = state.get('steps', {}).get('geometry_train', {}); command = step.get('command', [])
    cache_step = state.get('steps', {}).get('table_cache', {})
    require(step.get('status') == 'completed' and '--cache' not in command and not any(str(x).startswith('--cache=') for x in command),
            'geometry reconstruction requires a completed command without cache overrides')
    require('--run' in command and resolve(command[command.index('--run')+1]) == run,
            'geometry reconstruction command/run mismatch')
    require(cache_step.get('status') == 'completed' and cache_step.get('exit_code') == 0,
            'geometry reconstruction requires a successfully completed cache step')
    expected = current_cache_identity(config)
    require(expected == cache['identity'] and identity(expected)[:20] == Path(cache['path']).name,
            'official cache_identity does not map geometry config to the corrected cache')
    attempts = list(run.glob('attempts/*/epochs.jsonl'))
    require(len(attempts) == 1, 'geometry reconstruction is limited to its unique original training attempt')
    environment_path = attempts[0].parent/'environment.json'; environment = read(environment_path); audit.digest(environment_path)
    require(environment.get('code_hashes', {}).get('ttnet/src/events/engine.py') == expected.get('code_sha256')
            and bool(expected.get('code_sha256')), 'geometry training-time cache identity implementation differs')
    argv = environment.get('command', [])
    require('--cache' not in argv and not any(str(x).startswith('--cache=') for x in argv),
            'geometry trainer environment contains a cache override')
    require('--run' in argv and resolve(argv[argv.index('--run')+1]) == run,
            'geometry trainer environment/run mismatch')
    require(datetime.fromisoformat(cache_step['finished']) < datetime.fromisoformat(environment['started']),
            'geometry training started before corrected cache completion')
    parity = data['parity']
    require(parity.get('status') == 'exact' and parity.get('fully_consistent') is True
            and len(parity.get('per_video', {})) == 40 and parity['totals'].get('frames') == 6292,
            'geometry reconstruction requires exact full40-video/6292-frame probability and ball-record parity')
    require(cache['videos'] == 159, 'geometry reconstruction requires all159 source/output video hash pairs')
    audit.cache_receipt_exceptions.append(dict(seed=42,arm='geometry',run=str(run),
        evidence_type='reconstructed_evidence_not_an_original_receipt',environment=str(environment_path),
        official_identity=identity(expected),cache_finished=cache_step['finished'],trainer_started=environment['started'],
        reason='This unique corrected run started immediately before receipt logging was added. No cache override, official identity mapping, completed159-video hashes and exact full40-video parity reconstruct the accepted cache; no original receipt is invented.'))


def validate_training_cache(seed, arm, task, state, run, config, data, cache, corrected, parent, audit):
    attempts = [p.parent for p in run.glob('attempts/*/epochs.jsonl') if p.stat().st_size]
    require(bool(attempts), 'no completed training attempt for cache verification: '+str(run))
    receipts = [p/'feature_cache.json' for p in attempts]
    missing = [p for p in receipts if not p.is_file()]
    if missing:
        require(len(missing) == len(receipts), 'only some training attempts have cache provenance: '+str(run))
        if seed == 42 and arm == 'F0':
            exact_original_F0(parent, run, data, audit); return
        if seed == 42 and arm == 'geometry':
            reconstructed_geo42(task, state, run, config, data, cache, corrected, audit); return
        raise ValueError('missing actual training feature-cache receipt: '+str(missing[0]))
    expected_cache = resolve(state['base_cache']) if arm == 'F0' else Path(cache['path'])
    expected_identity = read(expected_cache/'identity.json')
    require(expected_cache.name == identity(expected_identity)[:20], 'training cache path/identity hash mismatch')
    require(expected_identity.get('dataset_id') == config.get('dataset_id'), 'training cache dataset differs from the trained config')
    for path in receipts:
        receipt = read(path); audit.digest(path)
        require(receipt.get('schema') == 'event_training_feature_cache_v1' and resolve(receipt['path']) == expected_cache,
                'actual training cache differs from the declared study cache: '+str(path))
        require(receipt.get('identity') == expected_identity and receipt.get('identity_sha256') == audit.digest(expected_cache/'identity.json'),
                'training cache receipt identity changed: '+str(path))
        expected_derivation = cache['derivation_sha256'] if arm != 'F0' else None
        require(receipt.get('derivation_sha256') == expected_derivation, 'training cache derivation receipt changed: '+str(path))


def validate_visual_quality(task, analysis, quality, audit):
    require(quality.get('status') == 'passed', 'complete video decoding failed: '+str(task))
    expected = {(arm,vid):path for arm in ARMS for vid,path in analysis['arms'][arm]['errors']['overlays'].items()}
    require(len(expected) == 120, 'visual QA must cover all3 arms ×40 videos')
    seen=set();total=0
    for item in quality.get('videos', []):
        key=(item['arm'],item['video_id'])
        require(key in expected and key not in seen, 'duplicate or unexpected visual QA video')
        seen.add(key);path=resolve(expected[key]); require(resolve(item['path']) == path, 'visual QA path differs from the report')
        frames=analysis['arms'][key[0]]['parity']['per_video'][key[1]]['frames']
        require(item.get('passed') is True and frames > 0 and frames == item.get('expected_frames') == item.get('decoded_frames') == item.get('declared_frames'),
                'visual QA frame coverage mismatch: '+str(key))
        require(audit.digest(path) == item.get('sha256'), 'overlay changed after complete video decoding: '+str(path))
        total += frames
    require(seen == set(expected) and total == quality.get('total_decoded_frames'), 'visual QA omits videos or misstates total frames')
    for path in (task/'training_curves.png',task/'review_cases/index.html'):
        audit.digest(path); require(path.stat().st_size > 0, 'missing visual review artifact: '+str(path))


def validate_benchmark(run, config, data, audit):
    path=run/'evaluations/benchmark/benchmark.json'; stats=read(path); digest=audit.digest(path)
    sample_path=path.with_name('benchmark_samples_ms.json'); samples=read(sample_path); audit.digest(sample_path)
    require(stats.get('warmup') == 50 and stats.get('measured') == 600 and stats.get('batch') == 1
            and isinstance(samples,list) and len(samples) == 600 and all(math.isfinite(v) and v > 0 for v in samples),
            'missing or incomplete benchmark protocol: '+str(run))
    require(stats.get('parameters', {}).get('head') == data['training']['head_parameters'], 'benchmark head does not match the trained arm')
    require(stats.get('amp') is config.get('amp') and str(stats.get('device','')).split(':')[0] == 'cuda',
            'benchmark device/precision differs from the experiment')
    for percentile,key in ((50,'head_forward_p50_ms'),(95,'head_forward_p95_ms')):
        require(math.isclose(float(np.percentile(samples,percentile)),stats[key],rel_tol=1e-9,abs_tol=1e-12),
                'benchmark summary differs from its600 raw samples')
    # The established head_benchmark schema records amp/device but omits backend.
    # It is a PyTorch-only producer; retain that explicit inference and its source SHA.
    backend=stats.get('backend','PyTorch'); precision='AMP' if stats['amp'] else 'FP32'
    signature=(stats.get('hardware'),backend,precision)
    performance=data['performance']
    require(signature == (performance.get('hardware'),performance.get('backend'),performance.get('precision')),
            'head benchmark and complete-video hardware/backend/precision differ')
    require(bool(signature[0]) and backend == 'PyTorch', 'unknown benchmark hardware/backend')
    return dict(path=str(path),sha256=digest,hardware=signature[0],backend=backend,precision=precision,
                backend_evidence='PyTorch-only ttnet.events.evaluate.head_benchmark producer; legacy schema omits backend',
                producer_sha256=audit.digest(PROJECT/'ttnet/src/events/evaluate.py'))


def validate_replication(manifest):
    require(manifest.get('seeds') == [42,43,44] and set(manifest.get('tasks', {})) == {'42','43','44'},
            'replication seeds must be exactly42/43/44 once each')
    tasks={int(seed):resolve(path) for seed,path in manifest['tasks'].items()}
    require(len(set(tasks.values())) == 3, 'replication tasks must be three independent identities')
    parent=resolve(manifest['parent']);screen=read(parent/'task.json');audit=ReplicationAudit();audit.digest(parent/'task.json')
    layout_path=Path(manifest.get('layout_fix','input_layout_fix.json'))
    layout_path=resolve(layout_path) if layout_path.is_absolute() else parent/layout_path
    layout=read(layout_path);audit.digest(layout_path);corrected=resolve(layout['corrected_seed42_task'])
    require(tasks[42] == corrected and corrected != parent, 'seed42 must use the corrected-layout task')
    retired={resolve(screen['runs'][arm]) for arm in ('geometry','concat')}
    analyses={seed:read(task/'analysis.json') for seed,task in tasks.items()}
    qualities={seed:read(task/'visual_qa.json') for seed,task in tasks.items()}
    states={seed:read(task/'task.json') for seed,task in tasks.items()}
    run_paths=[resolve(states[seed]['runs'][arm]) for seed in tasks for arm in ARMS]
    require(len(set(run_paths)) == 9, 'replication arms must use nine independent run identities')
    require(not set(run_paths) & retired, 'retired screen geometry/concat run mixed into replications')
    cache_paths={resolve(read(task/'table_cache.json')['cache']) for task in tasks.values()}
    require(len(cache_paths) == 1, 'replications did not use the identical corrected table cache')
    cache=validate_table_cache(next(iter(cache_paths)),audit)
    require(resolve(layout['original_cache']) != Path(cache['path']), 'retired table cache mixed into replications')
    require(layout.get('helper_sha256') == cache['helper_sha256'], 'layout-fix proof refers to another helper implementation')
    configs=[];protocols=set();benchmarks={};rows_by_seed={}
    for seed,task in tasks.items():
        state=states[seed];analysis=analyses[seed];audit.digest(task/'analysis.json');audit.digest(task/'visual_qa.json')
        audit.digest(task/'table_cache.json')
        require(state.get('seed') == seed == analysis.get('seed') and resolve(analysis['task']) == task,
                f'seed{seed}: manifest/task/report seed or task identity mismatch')
        require(analysis.get('status') == 'analyzed_internal_single_seed' and analysis.get('checks')
                and all(c.get('passed') is True for c in analysis['checks']), f'seed{seed}: accuracy/parity audit is incomplete')
        rows=analysis.get('comparison', []);require(len(rows) == 3 and {row['arm'] for row in rows} == set(ARMS), 'comparison must contain each arm exactly once')
        rows_by_seed[seed]={row['arm']:row for row in rows}
        required={task/'task.json',task/'planned_schedule.json',resolve(PROJECT/'scripts/report_b_event_study.py')}
        require(read(task/'planned_schedule.json').get('seed') == seed, f'seed{seed}: planned schedule seed mismatch')
        for arm in ARMS:
            run=resolve(state['runs'][arm]);config_path=run/'config.yaml';cfg=read(config_path);configs.append(cfg)
            data=analysis['arms'][arm];row=rows_by_seed[seed][arm]
            require(cfg.get('seed') == row.get('seed') == seed, f'seed{seed}/{arm}: actual config/comparison seed mismatch')
            evaluation=resolve(data['evaluation']);result=read(evaluation);protocols.add(result['protocol_id'])
            require(evaluation.is_relative_to(run/'evaluations') and resolve(row['metrics']) == evaluation,
                    'evaluation result is outside its independent run')
            require(arm in state.get('evaluations', {}) and resolve(state['evaluations'][arm]) == evaluation,
                    'task evaluation binding changed after its report')
            require(result.get('seed') == result.get('config', {}).get('seed') == seed, 'evaluation uses another seed')
            require(data['parity'].get('decisions_equal') is True, 'cache/online decisions differ in a replication arm')
            require(row['macro_f1'] == result['metrics']['macro_f1'] and all(row[k] == result['metrics']['landing'][k] for k in ('tp','fp','fn')),
                    'comparison scores differ from actual evaluation')
            require(set(data['checkpoints']) == {'best','last'}, 'replication must retain best and last checkpoints')
            for name,checkpoint in data['checkpoints'].items():
                path=resolve(checkpoint['path']);require(path == run/f'checkpoints/{name}.pt', 'checkpoint belongs to another run')
                require(audit.digest(path) == checkpoint['sha256'], 'checkpoint changed after evaluation: '+str(path));required.add(path)
            require(result['checkpoint_sha256'] == data['checkpoints']['best']['sha256'], 'evaluation used another checkpoint')
            required.update((config_path,evaluation,run/'predictions/best_pp_continuous.json'))
            validate_training_cache(seed,arm,task,state,run,cfg,data,cache,corrected,parent,audit)
            benchmarks[f'{seed}/{arm}']=validate_benchmark(run,cfg,data,audit)
        audit.report_sources(task,analysis,required)
        validate_visual_quality(task,analysis,qualities[seed],audit)
    keys=[k for k in PROTOCOL_KEYS if k != 'seed']
    require(all({k:cfg.get(k) for k in keys} == {k:configs[0].get(k) for k in keys} for cfg in configs),
            'cross-seed training controls differ beyond seed')
    require(len(protocols) == 1, 'scoring protocol differs across replications')
    require(len({(b['hardware'],b['backend'],b['precision']) for b in benchmarks.values()}) == 1,
            'benchmark hardware/backend/precision differs across the nine runs')
    validation=dict(status='passed',table_cache=cache,benchmarks=benchmarks,
                    mutable_source_exceptions=audit.mutable_source_exceptions,
                    cache_receipt_exceptions=audit.cache_receipt_exceptions,sources_sha256=audit.sources,
                    policy='Only each report-owned task.json may append orchestration state. All other report-source hashes are immutable. Two narrowly identified pre-receipt seed42 exceptions are disclosed rather than fabricated as original receipts.')
    return tasks,analyses,qualities,protocols,cache,validation


def aggregate(manifest):
    tasks,analyses,qualities,protocols,cache,validation = validate_replication(manifest)
    seeds = sorted(tasks); scores = {}; rows = []
    for arm in ARMS:
        by_seed = {seed:next(r for r in analyses[seed]['comparison'] if r['arm']==arm) for seed in seeds}
        values = [by_seed[s]['macro_f1'] for s in seeds]
        scores[arm] = dict(by_seed=by_seed, macro_f1_mean=float(np.mean(values)),
                           macro_f1_std=float(np.std(values)), mean_fp=float(np.mean([r['fp'] for r in by_seed.values()])),
                           by_view_mean={v:float(np.mean([r[v+'_f1'] for r in by_seed.values()])) for v in ('side','diagonal','front')},
                           e2e_fps_by_seed={s:by_seed[s]['e2e_fps'] for s in seeds})
        rows.append(dict(arm=arm, **{f'seed{s}':by_seed[s]['macro_f1'] for s in seeds},
                         mean=scores[arm]['macro_f1_mean'], std_population=scores[arm]['macro_f1_std'],
                         mean_fp=scores[arm]['mean_fp'], **{v+'_mean':x for v,x in scores[arm]['by_view_mean'].items()}))
    paired = {}
    for left,right in (('F0','geometry'),('F0','concat'),('geometry','concat')):
        changes = {s:scores[right]['by_seed'][s]['macro_f1']-scores[left]['by_seed'][s]['macro_f1'] for s in seeds}
        paired[left+'_to_'+right] = dict(by_seed=changes, mean=float(np.mean(list(changes.values()))),
                                        std_population=float(np.std(list(changes.values()))),
                                        positive_seeds=sum(v>0 for v in changes.values()))
    history = {}
    for seed in seeds:
        base = PROJECT/'result/tcn_frames_lr0.0003_d0.1'
        path = base/'metrics.json' if seed==42 else base/f'seeds/seed{seed}/metrics.json'
        data = read(path)
        if data['protocol_id'] not in protocols:
            raise ValueError('historical reference uses another scoring protocol')
        history[seed] = dict(macro_f1=data['metrics']['macro_f1'], path=str(path), sha256=sha(path))
    candidate = max(ARMS,key=lambda arm:(scores[arm]['macro_f1_mean'],-scores[arm]['mean_fp'],-ARMS.index(arm)))
    output = resolve(manifest['parent'])
    result = dict(schema='B_event_replication_summary_v1',status='replicated_internal_pending_human_review',
                  created=now(),seeds=seeds,arms=scores,paired=paired,historical=history,
                  historical_macro_f1_mean=float(np.mean([x['macro_f1'] for x in history.values()])),
                  highest_mean_internal_candidate=candidate,table_cache=cache['path'],validation=validation,
                  total_decoded_overlay_frames=sum(q['total_decoded_frames'] for q in qualities.values()),
                  reports={seed:str(task/'comparison.md') for seed,task in tasks.items()},
                  limitations=['Three seeds on the same internal validation set; no independent generalization claim or significance test.',
                               'Historical comparison changes locators, table and event settings; it is not a locator-only ablation.',
                               'Concat changes visual information and projection capacity. FPS is descriptive local full-video timing, not a controlled speed attribution.'])
    save(output/'replication_analysis.json', result)
    with (output/'replication_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 新B定位器三种子配对训练', '', '| 方案 | seed42 | seed43 | seed44 | 均值 ± 标准差 | 侧/斜/正平均F1 |', '|---|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['arm']} | {row['seed42']:.2%} | {row['seed43']:.2%} | {row['seed44']:.2%} | {row['mean']:.2%} ± {row['std_population']*100:.2f}pp | {row['side_mean']:.2%}/{row['diagonal_mean']:.2%}/{row['front_mean']:.2%} |")
    lines += ['', '标准差使用三种子的总体标准差（ddof=0），宏F1先按各轮完整40视频的三视角评分，再跨种子平均。', '', '## 配对差值', '']
    for name,data in paired.items():
        detail=', '.join(f'seed{s} {value*100:+.2f}pp' for s,value in data['by_seed'].items())
        lines.append(f"- {name}：{detail}；平均{data['mean']*100:+.2f}pp，{data['positive_seeds']}/3种子为正。")
    lines += ['', f"内部验证均值最高：{candidate}。历史主基线三种子均值为{result['historical_macro_f1_mean']:.2%}，两组配方同时有多项变化，不能作定位器单变量归因。", '', '## 完整结果与错误视频', '']
    for seed,task in tasks.items():
        link=os.path.relpath(task/'comparison.md',output)
        lines.append(f'- [seed{seed}完整审计、三组错误入口和权重]({link})；[曲线]({os.path.relpath(task/"training_curves.png",output)})；[配对案例]({os.path.relpath(task/"review_cases/index.html",output)})。')
    lines += ['', f"九组完整叠加视频共逐帧解码{result['total_decoded_overlay_frames']}帧；全部单种子协议、冻结权重、逐轮采样、缓存/在线决策及错误覆盖审计通过。", '', '三任务/九运行独立，缓存159组源与输出哈希、C-order/batch1身份、真实训练缓存凭证及九组benchmark已复核。仅各自task.json允许追加状态，其他报告来源哈希必须保持不变。']
    for exception in validation['cache_receipt_exceptions']:
        lines.append(f"- seed{exception['seed']} {exception['arm']}缓存凭证历史例外：{exception['reason']}")
    lines += ['', '仅内部验证与三个种子，不作独立外测或统计显著性结论。未使用public/test调参。各组真实轮数按早停不同；全部保留best/last与逐轮记录。']
    (output/'replication_summary.md').write_text('\n'.join(lines)+'\n')
    return result


def run(path):
    path=resolve(path);state=read(path);parent=resolve(state['parent'])
    lock=(parent/'replication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
             YOLO_CONFIG_DIR=str(parent/'replication_yolo_config'),MPLCONFIGDIR=str(parent/'replication_matplotlib_config'))
    def command(key,args):
        if state.setdefault('steps',{}).get(key,{}).get('status')=='completed':return
        logs=parent/'replication_logs';logs.mkdir(exist_ok=True)
        number=max((int(p.stem.rsplit('_',1)[1]) for p in logs.glob(key+'_*.log')),default=0)+1
        log=logs/f'{key}_{number:03d}.log';cmd=[sys.executable,'-u',*map(str,args)]
        entry=dict(status='running',command=cmd,started=now(),log=str(log));state['status']=key;state['steps'][key]=entry;save(path,state)
        with log.open('w') as stream:
            child=subprocess.Popen(cmd,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,env=env)
            entry['pid']=child.pid;save(path,state);code=child.wait()
        entry.update(status='completed' if code==0 else 'failed',exit_code=code,finished=now());save(log.with_suffix('.json'),entry);save(path,state)
        if code:raise RuntimeError(f'{key} exited{code}: {log}')
    try:
        print('Waiting for seed42 accuracy/parity/physical artifact checks.',flush=True)
        while True:
            quality=read(parent/'quality_driver.json')
            if quality['status']=='completed':break
            if quality['status'] in ('failed','blocked_by_upstream_failure'):
                raise RuntimeError('seed42 final artifact checks failed; investigate before replication')
            time.sleep(10)
        revised_screen = resolve(state['tasks']['42']) != parent
        if not revised_screen and read(parent/'analysis.json')['status']!='analyzed_internal_single_seed':
            raise RuntimeError('seed42 parity/integrity audit needs review before replication')
        for seed in ((42,43,44) if revised_screen else (43,44)):
            task=resolve(state['tasks'][str(seed)])
            args=['-m','scripts.run_b_event_study','--task',task]
            if seed==42 and revised_screen:args.append('--wait-f0')
            command(f'seed{seed}_train_evaluate',args)
            command(f'seed{seed}_report',['-m','scripts.report_b_event_study','--task',task])
            command(f'seed{seed}_visual_review',['-m','scripts.review_b_event_artifacts','--task',task])
            if read(task/'analysis.json')['status']!='analyzed_internal_single_seed':
                raise RuntimeError(f'seed{seed}: online/cache audit requires review')
            for arm,run_path in read(task/'task.json')['runs'].items():
                benchmark=resolve(run_path)/'evaluations/benchmark/benchmark.json'
                if seed==42 and arm=='F0' and benchmark.is_file():
                    state['steps'][f'seed{seed}_{arm}_benchmark']=dict(status='completed',adopted=True,
                        path=str(benchmark),sha256=sha(benchmark),reason='Unchanged F0 reused from exact-parity screen; preserve original benchmark.')
                    save(path,state)
                    continue
                command(f'seed{seed}_{arm}_benchmark',['-m','ttnet.cli','_event_benchmark','--run',run_path,'--device','cuda'])
        result=aggregate(state);state.update(status='completed_pending_review',finished=now(),candidate=result['highest_mean_internal_candidate']);save(path,state)
        print(json.dumps(dict(status=state['status'],candidate=state['candidate'])),flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error),stopped=now());save(path,state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    run(parser.parse_args().manifest)
