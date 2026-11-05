"""CPU-only replication integrity tests; all generated artifacts are temporary.

Fixtures contain small hashable stand-ins, not trained neural networks. They
exercise provenance, seed/layout isolation and stage completeness without GPU
access, inference, or mutation of the live study.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('replication_audit',PROJECT/'scripts/run_b_event_replication.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def file(path,value=b'fixture'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value);return path


class ReplicationFixture:
    def __init__(self,root):
        self.root=Path(root);self.parent=self.root/'screen';self.parent.mkdir()
        self.tasks={seed:self.root/f'task{seed}' for seed in (42,43,44)}
        self.manifest=dict(parent=str(self.parent),seeds=[42,43,44],tasks={str(seed):str(task) for seed,task in self.tasks.items()},layout_fix='input_layout_fix.json')
        source_identity=dict(schema=3,dataset_id='fixture',config=dict(amp=True))
        self.base=self.root/'cache'/module.identity(source_identity)[:20];save(self.base/'identity.json',source_identity)
        helper=module.sha(PROJECT/'ttnet/src/scene/visual_cache.py')
        target=dict(schema=4,dataset_id='fixture',config=dict(amp=True),code_sha256=module.sha(PROJECT/'ttnet/src/events/engine.py'),
                    table_visual=dict(implementation={'scene/visual_cache.py':helper}))
        self.cache=self.root/'cache'/module.identity(target)[:20];save(self.cache/'identity.json',target)
        sources={};completed={}
        for i in range(159):
            vid=f'v{i:03d}';pair={};output={}
            for key,suffix in [('npz','.npz'),('records','.jsonl')]:
                pair[key]=module.sha(file(self.base/(vid+suffix),b'ball:'+vid.encode()))
                output[key]=module.sha(file(self.cache/(vid+suffix),b'ball+table:'+vid.encode()))
            sources[vid]=pair;completed[vid]=dict(source=pair,outputs=output,frames=1,ball_arrays_exact=True,ball_records_exact_excluding_table=True)
        save(self.cache/'derivation.json',dict(status='completed',base=str(self.base),base_identity=source_identity,
            target_identity=target,implementation_sha256=module.sha(PROJECT/'ttnet/src/scene/augment_event_cache.py'),
            table_memory_format='contiguous_nchw',table_inference_batch=1,table_inference_precision='AMP',sources=sources,completed=completed))
        save(self.parent/'input_layout_fix.json',dict(corrected_seed42_task=str(self.tasks[42]),original_cache=str(self.root/'old_cache'),helper_sha256=helper))
        save(self.parent/'task.json',dict(runs=dict(F0=str(self.parent/'oldF0'),geometry=str(self.parent/'old_geometry'),concat=str(self.parent/'old_concat'))))
        self.immutable=file(self.root/'immutable.txt',b'data-version1')
        history=PROJECT/'result/tcn_frames_lr0.0003_d0.1/metrics.json'
        self.protocol=module.read(history)['protocol_id'] if history.is_file() else 'fixture_protocol'
        self.runs={};self.configs={};self.data={}
        for seed,task in self.tasks.items():
            runs={arm:self.root/f'run{seed}_{arm}' for arm in module.ARMS};self.runs[seed]=runs
            state=dict(seed=seed,base_cache=str(self.base),runs={arm:str(run) for arm,run in runs.items()},
                       evaluations={arm:str(run/'evaluations/continuous/metrics.json') for arm,run in runs.items()})
            save(task/'task.json',state);save(task/'planned_schedule.json',dict(seed=seed))
            save(task/'table_cache.json',dict(cache=str(self.cache),batch=1,derivation=str(self.cache/'derivation.json')))
            analysis=dict(task=str(task),seed=seed,status='analyzed_internal_single_seed',checks=[dict(name='fixture',passed=True)],arms={},comparison=[],sources_sha256={})
            quality=dict(status='passed',videos=[],total_decoded_frames=120)
            required=[task/'task.json',task/'planned_schedule.json',PROJECT/'scripts/report_b_event_study.py',self.immutable]
            for arm,run in runs.items():
                cfg=dict(seed=seed,amp=True,dataset_id='fixture',table_fusion=dict(mode='none' if arm=='F0' else arm))
                self.configs[seed,arm]=cfg;save(run/'config.yaml',cfg)
                checkpoints={}
                for name in ('best','last'):
                    path=file(run/f'checkpoints/{name}.pt',f'{seed}:{arm}:{name}'.encode())
                    checkpoints[name]=dict(path=str(path),sha256=module.sha(path));required.append(path)
                prediction=file(run/'predictions/best_pp_continuous.json',b'fixture-predictions')
                evalpath=run/'evaluations/continuous/metrics.json'
                metric=dict(macro_f1=.7,landing=dict(tp=100,fp=10,fn=89))
                result=dict(seed=seed,config=cfg,protocol_id=self.protocol,metrics=metric,checkpoint_sha256=checkpoints['best']['sha256'])
                save(evalpath,result);required.extend([run/'config.yaml',evalpath,prediction])
                row=dict(arm=arm,seed=seed,macro_f1=.7,tp=100,fp=10,fn=89,side_f1=.7,diagonal_f1=.7,front_f1=.7,e2e_fps=70.,metrics=str(evalpath))
                analysis['comparison'].append(row)
                overlays={};pervideo={}
                for i in range(40):
                    vid=f'02_{i:03d}';path=file(evalpath.parent/f'{vid}_overlay.mp4',f'{seed}:{arm}:{vid}'.encode())
                    overlays[vid]=str(path);pervideo[vid]=dict(frames=1)
                    quality['videos'].append(dict(arm=arm,video_id=vid,path=str(path),sha256=module.sha(path),declared_frames=1,expected_frames=1,decoded_frames=1,passed=True))
                data=dict(checkpoints=checkpoints,evaluation=str(evalpath),training=dict(head_parameters=10),
                          parity=dict(status='exact',fully_consistent=True,decisions_equal=True,totals=dict(frames=40),per_video=pervideo),
                          errors=dict(overlays=overlays),performance=dict(hardware='Fixture GPU',backend='PyTorch',precision='AMP'))
                analysis['arms'][arm]=data;self.data[seed,arm]=data
                file(run/'attempts/001/epochs.jsonl',b'{"epoch":1}\n')
                selected=self.base if arm=='F0' else self.cache
                save(run/'attempts/001/feature_cache.json',dict(schema='event_training_feature_cache_v1',path=str(selected),identity=module.read(selected/'identity.json'),
                    identity_sha256=module.sha(selected/'identity.json'),derivation_sha256=None if arm=='F0' else module.sha(self.cache/'derivation.json')))
                save(run/'evaluations/benchmark/benchmark.json',dict(hardware='Fixture GPU',parameters=dict(head=10),amp=True,device='cuda:0',batch=1,warmup=50,measured=600,head_forward_p50_ms=1.,head_forward_p95_ms=1.))
                save(run/'evaluations/benchmark/benchmark_samples_ms.json',[1.]*600)
            analysis['sources_sha256']={str(path):module.sha(path) for path in required}
            save(task/'analysis.json',analysis);save(task/'visual_qa.json',quality)
            file(task/'training_curves.png');file(task/'review_cases/index.html',b'<html>fixture</html>');file(task/'comparison.md')


class ReplicationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.fixture=ReplicationFixture(self.temporary.name)

    def validate(self):return module.validate_replication(self.fixture.manifest)

    def test_correct_three_tasks_nine_runs_and_stages_pass(self):
        tasks,analyses,qualities,protocols,cache,proof=self.validate()
        self.assertEqual(sorted(tasks),[42,43,44]);self.assertEqual(len(proof['benchmarks']),9)
        self.assertEqual(cache['videos'],159);self.assertEqual(proof['cache_receipt_exceptions'],[])
        self.assertEqual(len(proof['mutable_source_exceptions']),3)
        before={path:module.sha(path) for path in (task/'task.json' for task in tasks.values())}
        if (PROJECT/'result/tcn_frames_lr0.0003_d0.1/metrics.json').is_file():
            output=module.aggregate(self.fixture.manifest)
            self.assertEqual(output['validation']['status'],'passed')
            self.assertTrue((self.fixture.parent/'replication_summary.md').is_file())
        self.assertEqual(before,{path:module.sha(path) for path in before})

    def test_duplicate_task_and_missing_seed_are_rejected(self):
        self.fixture.manifest['tasks']['43']=self.fixture.manifest['tasks']['42']
        with self.assertRaisesRegex(ValueError,'independent identities'):self.validate()
        self.fixture.manifest['seeds']=[42,43]
        with self.assertRaisesRegex(ValueError,'exactly42/43/44'):self.validate()

    def test_config_seed_cannot_hide_behind_manifest_label(self):
        path=self.fixture.runs[43]['F0']/'config.yaml';cfg=module.read(path);cfg['seed']=42;save(path,cfg)
        with self.assertRaisesRegex(ValueError,'actual config/comparison seed mismatch'):self.validate()

    def test_report_seed_cannot_hide_behind_manifest_label(self):
        path=self.fixture.tasks[44]/'analysis.json';data=module.read(path);data['seed']=42;save(path,data)
        with self.assertRaisesRegex(ValueError,'manifest/task/report seed'):self.validate()

    def test_retired_screen_run_is_rejected(self):
        path=self.fixture.tasks[42]/'task.json';state=module.read(path)
        state['runs']['geometry']=module.read(self.fixture.parent/'task.json')['runs']['geometry'];save(path,state)
        with self.assertRaisesRegex(ValueError,'retired screen'):self.validate()

    def test_old_layout_cache_is_rejected(self):
        path=self.fixture.cache/'derivation.json';data=module.read(path);data.pop('table_memory_format');save(path,data)
        with self.assertRaisesRegex(ValueError,'memory layout'):self.validate()

    def test_identity_path_hash_and_actual_outputs_are_checked(self):
        path=self.fixture.cache/'identity.json';ident=module.read(path);ident['unexpected']=True;save(path,ident)
        with self.assertRaisesRegex(ValueError,'differs from derivation'):self.validate()
        derivation=module.read(self.fixture.cache/'derivation.json');derivation['target_identity']=ident;save(self.fixture.cache/'derivation.json',derivation)
        with self.assertRaisesRegex(ValueError,'identity/path hash'):self.validate()

    def test_source_and_output_hashes_are_real_not_only_completion_flags(self):
        file(self.fixture.cache/'v000.npz',b'corrupted')
        with self.assertRaisesRegex(ValueError,'table cache output changed'):self.validate()

    def test_current_augmentation_source_is_required(self):
        path=self.fixture.cache/'derivation.json';data=module.read(path);data['implementation_sha256']='old';save(path,data)
        with self.assertRaisesRegex(ValueError,'augmentation implementation'):self.validate()

    def test_cache_precision_cannot_be_changed(self):
        path=self.fixture.cache/'derivation.json';data=module.read(path);data['table_inference_precision']='FP32';save(path,data)
        with self.assertRaisesRegex(ValueError,'cache precision'):self.validate()

    def test_immutable_report_source_drift_is_rejected(self):
        self.fixture.immutable.write_bytes(b'data-version2')
        with self.assertRaisesRegex(ValueError,'report source changed'):self.validate()

    def test_only_own_task_state_may_append_after_report(self):
        path=self.fixture.tasks[43]/'task.json';state=module.read(path);state['final_note']='reviewed';save(path,state)
        self.assertEqual(self.validate()[-1]['status'],'passed')

    def test_new_run_cannot_omit_training_cache_receipt(self):
        (self.fixture.runs[43]['geometry']/'attempts/001/feature_cache.json').unlink()
        with self.assertRaisesRegex(ValueError,'missing actual training'):self.validate()

    def test_training_receipt_cannot_point_at_another_cache(self):
        path=self.fixture.runs[44]['concat']/'attempts/001/feature_cache.json';data=module.read(path);data['path']=str(self.fixture.base);save(path,data)
        with self.assertRaisesRegex(ValueError,'actual training cache differs'):self.validate()

    def test_benchmark_must_exist_and_match_hardware(self):
        path=self.fixture.runs[43]['concat']/'evaluations/benchmark/benchmark.json';stats=module.read(path);stats['hardware']='Another GPU';save(path,stats)
        with self.assertRaisesRegex(ValueError,'hardware/backend/precision differ'):self.validate()
        path.unlink()
        with self.assertRaises(FileNotFoundError):self.validate()

    def test_visual_decode_receipt_cannot_skip_one_video(self):
        path=self.fixture.tasks[43]/'visual_qa.json';quality=module.read(path);quality['videos'].pop();save(path,quality)
        with self.assertRaisesRegex(ValueError,'omits videos'):self.validate()

    def test_geo42_reconstruction_is_narrow_and_not_an_original_receipt(self):
        task=self.fixture.tasks[42];run=self.fixture.runs[42]['geometry'];receipt=run/'attempts/001/feature_cache.json';receipt.unlink()
        cfg=self.fixture.configs[42,'geometry'];data=deepcopy(self.fixture.data[42,'geometry']);data['parity']['totals']['frames']=6292
        state=module.read(task/'task.json');state['steps']=dict(
            geometry_train=dict(status='completed',command=['python','--run',str(run)]),
            table_cache=dict(status='completed',exit_code=0,finished='2026-11-04T19:00:00+08:00'))
        save(run/'attempts/001/environment.json',dict(started='2026-11-04T19:01:00+08:00',command=['train.py','--run',str(run)],
            code_hashes={'ttnet/src/events/engine.py':module.sha(PROJECT/'ttnet/src/events/engine.py')}))
        audit=module.ReplicationAudit();cache=module.validate_table_cache(self.fixture.cache,audit)
        with patch.object(module,'current_cache_identity',return_value=cache['identity']):
            module.validate_training_cache(42,'geometry',task,state,run,cfg,data,cache,task,self.fixture.parent,audit)
            self.assertEqual(audit.cache_receipt_exceptions[0]['evidence_type'],'reconstructed_evidence_not_an_original_receipt')
            self.assertFalse(receipt.exists())
            state['steps']['geometry_train']['command']+=['--cache','other']
            with self.assertRaisesRegex(ValueError,'without cache overrides'):
                module.validate_training_cache(42,'geometry',task,state,run,cfg,data,cache,task,self.fixture.parent,audit)

    def test_original_F0_exception_checks_exact_frames_and_checkpoint_identity(self):
        task=self.fixture.tasks[42];run=self.fixture.runs[42]['F0'];(run/'attempts/001/feature_cache.json').unlink()
        screen=module.read(self.fixture.parent/'task.json');screen['runs']['F0']=str(run);save(self.fixture.parent/'task.json',screen)
        data=deepcopy(self.fixture.data[42,'F0']);data['parity']['totals']['frames']=6292
        save(self.fixture.parent/'analysis.json',dict(arms={'F0':data}))
        audit=module.ReplicationAudit();cache=module.validate_table_cache(self.fixture.cache,audit)
        module.validate_training_cache(42,'F0',task,module.read(task/'task.json'),run,self.fixture.configs[42,'F0'],data,cache,task,self.fixture.parent,audit)
        self.assertEqual(audit.cache_receipt_exceptions[0]['arm'],'F0')
        data['checkpoints']['best']['sha256']='different'
        with self.assertRaisesRegex(ValueError,'checkpoint differs'):
            module.validate_training_cache(42,'F0',task,module.read(task/'task.json'),run,self.fixture.configs[42,'F0'],data,cache,task,self.fixture.parent,audit)


if __name__=='__main__':unittest.main()
