"""Package one verified checkpoint selected from the completed public study.

Selection is highest three-seed mean arm, then highest public-F1 seed, fewer FP,
and lower seed. Original runs are read-only. Existing destinations are refused.
This verifies artifacts and strict CPU loading; it does not rerun inference.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tarfile
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from ttnet.common import save, sha

DEFAULT_TASK = PROJECT/'out/experiments/20261105-092319_public_event_study'


def require(passed, message):
    if not passed:
        raise ValueError(message)


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT/path).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def close(left, right):
    return (type(left) in (int, float) and type(right) in (int, float)
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12))


def select_checkpoint(analysis):
    """Recompute arm means and deterministic seed ranking from report rows."""
    arms, seeds = analysis['arms'], analysis['seeds']
    require(len(arms) == len(set(arms)) and len(seeds) == len(set(seeds)) == 3,
            'expected distinct arms and exactly three seeds')
    require(len(analysis['runs']) == len(arms)*3, 'unexpected number of reported runs')
    summaries = {}
    for arm in arms:
        runs = [(name, result) for name, result in analysis['runs'].items()
                if result['row']['arm'] == arm]
        require(len(runs) == 3 and {r['row']['seed'] for _, r in runs} == set(seeds),
                'incomplete three-seed arm: '+arm)
        for name, result in runs:
            row = result['row']
            require(row['name'] == name and row['status'] == 'completed'
                    and type(row['macro_f1']) in (int, float)
                    and math.isfinite(row['macro_f1']) and 0 <= row['macro_f1'] <= 1
                    and type(row['fp']) is int and row['fp'] >= 0,
                    'invalid run row: '+name)
        mean = statistics.mean(r['row']['macro_f1'] for _, r in runs)
        fp = statistics.mean(r['row']['fp'] for _, r in runs)
        require(close(mean, analysis['arm_summary'][arm]['macro_f1_mean']),
                'arm mean does not match its three runs: '+arm)
        summaries[arm] = dict(mean=mean, mean_fp=fp, runs=runs)
    arm = max(arms, key=lambda a:(summaries[a]['mean'], -summaries[a]['mean_fp'], -arms.index(a)))
    require(analysis['highest_mean_validation_candidate'] == arm,
            'reported winning arm differs from recomputed ranking')
    name, selected = min(summaries[arm]['runs'],
                         key=lambda item:(-item[1]['row']['macro_f1'], item[1]['row']['fp'], item[1]['row']['seed']))
    return arm, name, selected


def verify_sources(task, analysis, completion, selected):
    reports = ('analysis.json', 'comparison.csv', 'summary.md', 'training_curves.png')
    require(completion.get('schema') == 'public_event_completion_v1'
            and completion.get('status') == analysis.get('status') == 'completed'
            and completion.get('validation_set') == analysis.get('validation_set') == 'pp_public'
            and completion.get('all_checks_passed') is True
            and resolve(completion['task']) == resolve(analysis['task']) == task,
            'study completion or validation identity is invalid')
    checks = analysis.get('checks', [])
    require(bool(checks) and all(c.get('passed') is True for c in checks)
            and completion['machine_audit_checks'] == len(checks), 'not all study checks passed')
    require(completion['runs'] == len(analysis['runs'])
            and completion['sources_sha256'] == analysis['sources_sha256'],
            'study completion does not bind the analyzed runs and sources')
    verified = {}
    for name in reports:
        path = task/name
        expected = completion['reports_sha256'][name]
        require(sha(path) == expected, 'changed report: '+str(path))
        verified[str(path)] = expected
    verified[str(task/'completion.json')] = sha(task/'completion.json')
    row = selected['row']; run = resolve(row['checkpoint']).parent.parent
    require(run.name == row['experiment_id'], 'selected run/checkpoint identity differs')
    registry = {str(resolve(path)):digest for path, digest in analysis['sources_sha256'].items()}
    evaluation = resolve(selected['evaluation'])
    require(evaluation.is_relative_to(run/'evaluations')
            and resolve(row['metrics']) == evaluation/'metrics.json',
            'selected evaluation is not bound to its run')
    required = [run/'config.yaml', run/'run.json', run/'best_selection.json',
                run/'checkpoints/best.pt', run/'public_split_audit.json',
                evaluation/'metrics.json', evaluation/'performance.json', evaluation/'identity.json',
                evaluation/'cache_parity.json', evaluation/'overlay_verification.json',
                evaluation/'per_video.json', evaluation/'source_identity.json',
                run/'evaluations/benchmark/benchmark.json',
                run/'evaluations/benchmark/benchmark_samples_ms.json']
    for path in required:
        require(str(path) in registry, 'critical source missing from study audit: '+str(path))
    # task.json is the sole mutable orchestration snapshot. Every other source
    # bound by the completed study must still match, including other seeds,
    # public GT, configurations, caches, source videos and implementations.
    mutable_task = task/'task.json'
    for source, expected in registry.items():
        path = Path(source)
        if path == mutable_task:
            continue
        require(path.is_file() and sha(path) == expected, 'changed audited source: '+source)
        verified[source] = expected
    return run, evaluation, verified


def verify_checkpoint(task, analysis, selected, run, evaluation, verified):
    import torch
    row = selected['row']; config = read(run/'config.yaml')
    metric = read(evaluation/'metrics.json'); performance = read(evaluation/'performance.json')
    selection = read(run/'best_selection.json'); checkpoint = run/'checkpoints/best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    digest = sha(checkpoint)
    require(config == state['config'] == selected['config']
            and sha(run/'config.yaml') == row['config_sha256']
            and digest == row['best_sha256'] == metric['checkpoint_sha256']
            and state['epoch'] == selection['epoch'] == row['best_epoch']
            and state.get('step') == 0
            and state['thresholds'] == selection['thresholds'] == metric['thresholds']
            and close(state['thresholds'][0], row['threshold']),
            'checkpoint/config/epoch/threshold binding failed')
    require(metric.get('status') == 'evaluated' and metric.get('validation_set') == 'pp_public'
            and metric.get('threshold_source') == 'checkpoint thresholds'
            and metric.get('ground_truth_input_to_inference') is False
            and close(metric['macro_f1'], row['macro_f1'])
            and close(selection['continuous']['macro_f1'], row['macro_f1'])
            and all(metric['overall'][k] == selection['continuous']['landing'][k] == row[k]
                    for k in ('tp', 'fp', 'fn'))
            and close(performance['inference_e2e_fps'], metric['e2e_fps']),
            'selected validation scores do not match the checkpoint evidence')
    parity = metric.get('cache_online_parity', {})
    receipt = read(evaluation/'cache_parity.json')
    require(all(parity.get(k) is True and receipt.get(k) is True
                for k in ('passed', 'decisions_equal', 'landing_decisions_equal')),
            'selected cache/online decisions did not pass parity')
    record = PROJECT/'experiments/records'/run.name
    marker_path = record/'public_completion.json'; marker = read(marker_path)
    reference = marker['evaluation']
    require(marker.get('schema') == 'public_event_run_completion_v1'
            and marker.get('status') == 'completed' and marker.get('experiment_id') == run.name
            and resolve(marker['study_completion']) == task/'completion.json'
            and marker['study_completion_sha256'] == sha(task/'completion.json')
            and reference.get('status') == 'completed' and reference.get('validation_set') == 'pp_public'
            and resolve(reference['metrics']) == evaluation/'metrics.json'
            and reference['metrics_sha256'] == sha(evaluation/'metrics.json')
            and resolve(reference['checkpoint']) == checkpoint
            and reference['checkpoint_sha256'] == digest and reference['best_epoch'] == state['epoch'],
            'selected per-run completion receipt does not bind its artifacts')
    verified[str(marker_path)] = sha(marker_path)
    return config, state, metric, performance, marker_path


def strict_cpu_load(config, state):
    import torch
    from ttnet.events.engine import EventSystem
    from ttnet.models.table_encoder import ultralytics_api
    os.environ.setdefault('YOLO_CONFIG_DIR', str(PROJECT/'out/yolo_setup/event_table_config'))
    os.environ.setdefault('YOLO_OFFLINE', 'true')
    torch.set_num_threads(config.get('torch_threads', 4))
    api = ultralytics_api()
    with torch.random.fork_rng(devices=[]):
        system = EventSystem(deepcopy(config), torch.device('cpu'), initialize=False)
        loaded = system.load_state_dict(state['system'], strict=True)
        system.configure(config['mode'], False)
    require(not loaded.missing_keys and not loaded.unexpected_keys,
            'strict CPU load returned missing or unexpected keys')
    components = sorted({key.split('.')[0] for key in state['system']})
    required = {'global_model', 'local_model', 'head'}
    if config.get('table_frontend'):
        required.add('table_encoder')
    if config.get('table_checkpoint') or config.get('table_head'):
        required.add('table_head')
    require(required <= set(components), 'full checkpoint is missing a model component')
    result = dict(passed=True, device='cpu', initialize=False, strict=True,
        missing_keys=list(loaded.missing_keys), unexpected_keys=list(loaded.unexpected_keys),
        embedded_components=components, model_parameters=sum(p.numel() for p in system.parameters()),
        state_tensor_count=len(state['system']), torch_version=torch.__version__,
        ultralytics_version=api.__version__, dependency_path=str(PROJECT/'out/yolo-deps'),
        verification_scope='Artifact identity and strict CPU loading only; accuracy/inference not rerun.')
    del system
    return result


def verify_archive(archive, output, inventory):
    with tarfile.open(archive, 'r:gz') as stream:
        files = {member.name:member for member in stream.getmembers() if member.isfile()}
        expected = {output.name+'/'+name:digest for name,digest in inventory.items()}
        require(set(files) == set(expected), 'archive file list differs from the package')
        for name, member in files.items():
            digest = hashlib.sha256()
            with stream.extractfile(member) as content:
                for block in iter(lambda:content.read(1024*1024), b''):
                    digest.update(block)
            require(digest.hexdigest() == expected[name], 'archive content changed: '+name)


def package(task, output=None):
    task = resolve(task)
    analysis, completion = read(task/'analysis.json'), read(task/'completion.json')
    arm, name, selected = select_checkpoint(analysis)
    run, evaluation, verified = verify_sources(task, analysis, completion, selected)
    config, state, metric, performance, marker = verify_checkpoint(task, analysis, selected, run, evaluation, verified)
    verified_cpu = strict_cpu_load(config, state)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    output = resolve(output) if output else PROJECT/'out/weights'/f'{stamp}_public_event_best'
    archive = output.with_name(output.name+'.tar.gz')
    archive_hash = archive.with_name(archive.name+'.sha256')
    require(not any(path.exists() for path in (output, archive, archive_hash)),
            'output directory/archive already exists; choose a new unique --output')
    sources = {'best.pt':run/'checkpoints/best.pt', 'config.yaml':run/'config.yaml',
        'evidence/metrics.json':evaluation/'metrics.json',
        'evidence/performance.json':evaluation/'performance.json',
        'evidence/best_selection.json':run/'best_selection.json',
        'evidence/public_split_audit.json':run/'public_split_audit.json',
        'evidence/evaluation_identity.json':evaluation/'identity.json',
        'evidence/source_identity.json':evaluation/'source_identity.json',
        'evidence/cache_parity.json':evaluation/'cache_parity.json',
        'evidence/overlay_verification.json':evaluation/'overlay_verification.json',
        'evidence/per_video.json':evaluation/'per_video.json',
        'evidence/benchmark.json':run/'evaluations/benchmark/benchmark.json',
        'evidence/benchmark_samples_ms.json':run/'evaluations/benchmark/benchmark_samples_ms.json',
        'evidence/study_completion.json':task/'completion.json',
        'evidence/public_completion.json':marker,
        'evidence/study_summary.md':task/'summary.md'}
    # Nothing is published until the report, sources and CPU model all pass.
    output.mkdir(parents=True, exist_ok=False)
    for target, source in sources.items():
        path = output/target; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        require(sha(path) == verified[str(source)], 'copied evidence differs: '+target)
    row = selected['row']
    manifest = dict(schema='public_event_best_package_v1', created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        selection=dict(arm=arm, run=name, seed=row['seed'], best_epoch=state['epoch'],
            arm_three_seed_mean=analysis['arm_summary'][arm]['macro_f1_mean'],
            rule='highest three-seed mean arm; highest public macro F1 seed; fewer FP; lower seed',
            arm_ties='fewer mean FP, then preregistered arm order'),
        validation_set='pp_public', validation_role=analysis['validation_role'],
        public_macro_f1=metric['macro_f1'], overall=metric['overall'], by_angle=metric['by_angle'],
        thresholds=state['thresholds'], thresholds_order=['Bounce','Net'],
        locator_thresholds={key:config.get(key, config['localization_threshold']) for key in ('global_threshold','local_threshold')},
        dataset_id=config['dataset_id'], split_id=config['split_id'],
        checkpoint=dict(path='best.pt', source=str(run/'checkpoints/best.pt'), sha256=sha(output/'best.pt'),
                        bytes=(output/'best.pt').stat().st_size, embedded_components=verified_cpu['embedded_components']),
        original_run=str(run), study=str(task), study_completion_sha256=sha(task/'completion.json'),
        runtime_dependencies=dict(repository_code_required=True, repository=str(PROJECT),
            ultralytics='8.3.225', isolated_dependency='out/yolo-deps',
            note='Full model parameters are embedded. Repository implementations and the pinned runtime remain required; original pretraining checkpoint paths are provenance.'),
        original_artifacts_modified=False, accuracy_reevaluated=False,
        evidence_sources={target:str(source) for target,source in sources.items()})
    save(output/'manifest.json', manifest)
    save(output/'evidence/source_verification.json',dict(scope='All recorded study sources and core reports; task.json is the sole mutable orchestration snapshot.',
        mutable_source_exceptions=[str(task/'task.json')],
        verified_files=len(verified), sources_sha256=verified))
    save(output/'verification.json',dict(passed=True, strict_cpu_load=verified_cpu,
        checkpoint_byte_identical=True, config_epoch_thresholds_match=True,
        study_all_checks_passed=True, selected_source_hashes_verified=len(verified),
        all_model_parameters_embedded=True, accuracy_reevaluated=False,
        original_artifacts_modified=False))
    stability = ''
    if arm == 'gated':
        paired = analysis['event_analysis']['paired']['concat_to_gated']
        stability = (f"gated比concat的均值高{paired['mean_delta']*100:.2f}pp，只有{paired['positive_seeds']}/3种子改善，"
                     "尚不能声称稳定优于concat。\n")
    readme = f'''# PP public 验证选定候选权重

选择：三种子验证均值最高的 {arm}，其中 public 宏F1最高的 seed{row['seed']}，best epoch {state['epoch']}。
所选单次宏F1 {metric['macro_f1']:.4%}；阈值 Bounce/Net={state['thresholds']}。
所选方案三种子均值 {analysis['arm_summary'][arm]['macro_f1_mean']:.2%} ± {analysis['arm_summary'][arm]['macro_f1_std_population']*100:.2f}pp。
{stability}

PP154用于训练；pp_public用于验证和选型，不是独立测试。完整参数已包含Global、Local、事件头及配置启用的桌面前端。
仍需本仓库代码、PyTorch及 out/yolo-deps 中固定的 ultralytics 8.3.225。原配置中的预训练路径仅用于追溯；initialize=False不会读取这些原始权重。

```python
from pathlib import Path
import torch
from ttnet.events.engine import EventSystem

state = torch.load(Path("{output}/best.pt"), map_location="cpu", weights_only=False)
system = EventSystem(state["config"], torch.device("cpu"), initialize=False)
system.load_state_dict(state["system"], strict=True)
system.configure(state["config"]["mode"], False)
thresholds = state["thresholds"]
```

在本目录执行 `sha256sum -c SHA256SUMS` 校验包内文件。已核对研究完成收据、所选run与实现代码SHA、配置/轮次/阈值，以及严格CPU加载。本次未重跑推理或精度评估。
metrics/performance为完整视频评估；benchmark的p50/p95只测事件头，不能解释为完整系统延迟。原实验及权重保持原样。
'''
    (output/'README.md').write_text(readme)
    for source, expected in verified.items():
        require(sha(source) == expected, 'source changed during packaging: '+source)
    inventory = {str(path.relative_to(output)):sha(path) for path in sorted(output.rglob('*')) if path.is_file()}
    (output/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in inventory.items()))
    inventory['SHA256SUMS'] = sha(output/'SHA256SUMS')
    with tarfile.open(archive, 'x:gz') as stream:
        stream.add(output, arcname=output.name)
    verify_archive(archive, output, inventory)
    digest = sha(archive)
    archive_hash.write_text(f'{digest}  {archive.name}\n')
    return dict(status='completed', output=str(output), archive=str(archive), archive_sha256=digest,
                arm=arm, seed=row['seed'], epoch=state['epoch'], macro_f1=metric['macro_f1'],
                checkpoint_sha256=manifest['checkpoint']['sha256'], strict_cpu_load=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, default=DEFAULT_TASK)
    parser.add_argument('--output', type=Path, help='New unique directory; existing directories/archives are refused')
    args = parser.parse_args()
    print(json.dumps(package(args.task, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
