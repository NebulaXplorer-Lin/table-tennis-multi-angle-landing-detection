"""Reporting checks use temporary outputs and existing historical PP artifacts."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

PROJECT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('report_b_event_study',PROJECT/'scripts/report_b_event_study.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def lines(path,values):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(''.join(json.dumps(v)+'\n' for v in values))


class ReportHelpers(unittest.TestCase):
    def test_recovered_evaluation_path_is_stable_and_missing_receipt_fails(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);metric=root/'continuous_attempt002/metrics.json';save(metric,{})
            state=dict(runs={'F0':str(root)},steps={'F0_evaluate':dict(status='completed',command=['python','--output',str(metric.parent)])},evaluations={'F0':str(metric)})
            self.assertEqual(report.evaluation_path(state,'F0'),metric)
            save(root/'evaluations/continuous/metrics.json',{'unrelated':True})
            metric.unlink()
            with self.assertRaises(report.ReportError):report.evaluation_path(state,'F0')

    def test_sampling_recovery_keeps_duplicate_evidence_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);planned=dict(epochs=[dict(epoch=1,sha256='wanted',samples=10,sources={'pp':5},pp_positive=2)])
            row=dict(epoch=1,schedule_sha256='wanted',sampling_counts=dict(total=10,pp=5,pp_positive=2),thresholds=[.4,.5],continuous={})
            lines(root/'attempts/001/epochs.jsonl',[row]);lines(root/'attempts/002/epochs.jsonl',[dict(row,elapsed_sec=1)])
            audit=report.Audit();epochs,evidence=report.collect_epochs(root,planned,audit)
            self.assertEqual(list(epochs),[1]);self.assertEqual(evidence['duplicate_receipts'],[1]);self.assertTrue(all(x['passed'] for x in audit.checks))
            lines(root/'attempts/002/epochs.jsonl',[dict(row,schedule_sha256='unexpected')])
            audit=report.Audit();report.collect_epochs(root,planned,audit)
            self.assertFalse(all(x['passed'] for x in audit.checks))

    def test_sub_tolerance_probability_flip_is_a_decision_difference(self):
        annotation=dict(view='side',reviewed_intervals=[[0,2]],unknown_intervals=[],events=[dict(kind='bounce',frame_id=1,x=10.,y=20.)])
        rows=[dict(frame_id=i,source_frame_id=i,timestamp_sec=i/30,probabilities=[.1,.01],prediction=dict(x=10.,y=20.,valid=True,confidence=.8)) for i in range(3)]
        cached={'v':deepcopy(rows)};online={'v':deepcopy(rows)}
        cached['v'][1]['probabilities'][0]=.4-1e-7;online['v'][1]['probabilities'][0]=.4+1e-7
        result,_=report.compare_cached_online(cached,online,{'v':annotation},.4,.08)
        self.assertLess(result['probability_max_abs'],result['probability_atol'])
        self.assertFalse(result['fully_consistent']);self.assertEqual(result['status'],'decision_difference')
        self.assertEqual(result['totals']['bounce_threshold_decision_differences'],1)
        self.assertEqual(result['cache_metrics']['landing']['tp'],0);self.assertEqual(result['online_metrics']['landing']['tp'],1)

    def test_equal_event_counts_do_not_hide_peak_frame_change(self):
        annotation=dict(view='side',reviewed_intervals=[[0,3]],unknown_intervals=[],events=[dict(kind='bounce',frame_id=1,x=0.,y=0.)])
        rows=[dict(frame_id=i,source_frame_id=i,timestamp_sec=i/30,probabilities=[.1,.01],prediction=dict(x=0.,y=0.,valid=True)) for i in range(4)]
        a={'v':deepcopy(rows)};b={'v':deepcopy(rows)}
        a['v'][1]['probabilities'][0]=.7;a['v'][2]['probabilities'][0]=.70001
        b['v'][1]['probabilities'][0]=.70001;b['v'][2]['probabilities'][0]=.7
        result,_=report.compare_cached_online(a,b,{'v':annotation},.4,.08)
        self.assertEqual(result['per_video']['v']['cache_events'],result['per_video']['v']['online_events'])
        self.assertEqual(result['status'],'decision_difference');self.assertTrue(result['event_decision_differences'])

    def test_incomplete_frame_coverage_is_rejected(self):
        with self.assertRaises(report.ReportError):report.frame_rows([dict(frame_id=1,probabilities=[.2,.1])],'broken')

    def test_sub_tolerance_timestamp_difference_is_not_exact(self):
        annotation=dict(view='side',reviewed_intervals=[[0,0]],unknown_intervals=[],events=[])
        row=dict(frame_id=0,source_frame_id=0,timestamp_sec=0.,probabilities=[.1,.01],prediction=None)
        result,_=report.compare_cached_online({'v':[row]},{'v':[dict(row,timestamp_sec=1e-8)]},{'v':annotation},.4,.08)
        self.assertTrue(result['decisions_equal']);self.assertFalse(result['fully_consistent'])
        self.assertEqual(result['totals']['timestamp_unequal_frames'],1)
        self.assertEqual(result['totals']['timestamp_outside_atol_frames'],0)

    def test_historical189_pairing_and_coordinate_drift(self):
        source=PROJECT/'result/tcn_frames_lr0.0003_d0.1/metrics.json'
        diag=PROJECT/'out/experiments/20261104-182418_B_event_exploration/pretraining_diagnostics.json'
        if not source.is_file() or not diag.is_file():self.skipTest('historical study artifacts not available')
        metrics=report.read(source)['metrics'];diagnostics=report.read(diag)
        result=report.paired_events(metrics,{arm:deepcopy(metrics) for arm in report.ARMS},diagnostics)
        self.assertEqual(result['buckets']['old45FN']['events'],45)
        self.assertEqual(result['buckets']['old144TP']['arms']['F0']['fn'],0)
        self.assertEqual(result['buckets']['six_exact_frame_repairs']['events'],6)
        self.assertEqual(result['buckets']['no_roi_entire_13window']['events'],40)
        self.assertEqual(result['buckets']['group22']['events'],46)
        damaged=deepcopy(metrics);damaged['per_video']['02_001']['ground_truth'][0]['x']+=1
        with self.assertRaises(report.ReportError):report.paired_events(metrics,dict(F0=damaged,geometry=metrics,concat=metrics),diagnostics)
        # Preserve every aggregate count while exchanging which GT was matched.
        altered=deepcopy(metrics);score=altered['per_video']['02_001'];score['landing']['matches'][0]['ground_truth']=2
        score['landing']['false_negatives']=[0]
        self.assertEqual(altered['landing'],metrics['landing'])
        self.assertNotEqual(report.metric_signature(altered),report.metric_signature(metrics))

    def test_error_index_must_cover_scored_fn_even_if_overlay_exists(self):
        metric=dict(per_video={'v':dict(ground_truth=[dict(frame_id=1)],predictions=[],detected=[],temporal_ground_truth=[dict(frame_id=1)],
                                       landing=dict(false_positives=[],false_negatives=[0]),continuous_bounce=dict(false_positives=[],false_negatives=[0]))})
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'v_overlay.mp4').write_bytes(b'test-existence-only');(root/'index.html').write_text('fixture');lines(root/'error_index.jsonl',[])
            audit=report.Audit();result=report.audit_errors(root,metric,{'v':[dict(frame_id=1)]},audit,'fixture')
            self.assertEqual(len(result['missing_event_errors']),2);self.assertFalse(audit.checks[-1]['passed'])
            lines(root/'error_index.jsonl',[dict(video_id='v',frame_id=1,kinds=['landing_false_negative','continuous_false_negative'],visualization='v_overlay.mp4')])
            audit=report.Audit();report.audit_errors(root,metric,{'v':[dict(frame_id=1)]},audit,'fixture');self.assertTrue(audit.checks[-1]['passed'])


class FullHistoricalFixture(unittest.TestCase):
    def test_three_arm_report_is_repeatable_read_only_and_negative_results_survive(self):
        """Real189 GT/predictions, synthetic frozen states/schedule in a temp task.

        This validates report plumbing, not training or the new models' accuracy.
        """
        historical=PROJECT/'result/tcn_frames_lr0.0003_d0.1/metrics.json'
        diagnostic=PROJECT/'out/experiments/20261104-182418_B_event_exploration/pretraining_diagnostics.json'
        oldrun=PROJECT/'out/experiments/20261016_table_ar/tcn_frames_lr0.0003_d0.1'
        if not all(p.is_file() for p in (historical,diagnostic,oldrun/'predictions/best_pp_continuous.json')):
            self.skipTest('historical study artifacts not available')
        with tempfile.TemporaryDirectory() as name:
            task=Path(name);baseline=report.read(historical);diag=report.read(diagnostic)
            pretrained={};locked={}
            for stage in ('global','local','table'):
                path=task/'models'/stage/'best.pt';path.parent.mkdir(parents=True)
                value=torch.tensor([1.,2.]);torch.save(dict(model={'weight':value}),path)
                pretrained[stage]=path;locked[stage]=report.sha(path)
            save(pretrained['table'].parent/'metrics.json',dict(checkpoint_sha256=locked['table']))
            lock=task/'selection_lock.json';save(lock,dict(weights={'B':{s:locked[s] for s in ('global','local')}}))
            annotations=PROJECT/'data/annotations/pp_bounce_confirmed_v1.json'
            diag['sources_sha256']={str(historical):report.sha(historical),str(annotations):report.sha(annotations),str(lock):report.sha(lock)}
            save(task/'pretraining_diagnostics.json',diag)
            planned=report.read(PROJECT/'out/experiments/20261104-182418_B_event_exploration/planned_schedule.json');planned['epochs']=planned['epochs'][:2];save(task/'planned_schedule.json',planned)
            state=dict(runs={},configs={},evaluations={},steps={})
            for arm in report.ARMS:
                preset=report.read(PROJECT/f'configs/experiments/event_B_yolo_{arm}.json')
                # Resolve component references before relocating the fixture preset.
                for key,value in preset['components'].items():preset['components'][key]=str((PROJECT/'configs/experiments'/value).resolve())
                preset['dataset']=str(PROJECT/'configs/datasets/openttgames_event25_v3.json')
                for stage in ('global','local'):preset['parameters'][stage+'_checkpoint']=str(pretrained[stage])
                preset['parameters'].update(epochs=2,patience=1)
                if arm!='F0':preset['parameters']['table_frontend']['checkpoint']=str(pretrained['table'])
                preset_path=task/f'{arm}.json';save(preset_path,preset)
                cfg,_=report.load_config(preset_path);cfg.update({key:planned[key] for key in ('seed','dataset_id','train_samples_sha256','val_samples_sha256')})
                cfg.update(split_id='openttgames_event25_match_v1',global_sha256=locked['global'],local_sha256=locked['local'])
                run=task/arm;save(run/'config.yaml',cfg);save(run/'run.json',dict(status='trained_pending_eval',epochs_completed=2,stop_reason='early_stopping'))
                records=[]
                for e in planned['epochs']:
                    records.append(dict(epoch=e['epoch'],schedule_sha256=e['sha256'],sampling_counts=dict(total=e['samples'],pp=e['sources']['pp'],pp_positive=e['pp_positive']),thresholds=[.4,.5],continuous=baseline['metrics']))
                lines(run/'attempts/001/epochs.jsonl',records);save(run/'best_selection.json',records[0])
                tensors={stage+'_model.weight':torch.tensor([1.,2.]) for stage in ('global','local')}
                if arm!='F0':tensors['table_encoder.weight']=torch.tensor([1.,2.])
                for key,epoch in [('best',1),('last',2)]:
                    path=run/f'checkpoints/{key}.pt';path.parent.mkdir(exist_ok=True);torch.save(dict(system=tensors,config=cfg,epoch=epoch,thresholds=[.4,.5]),path)
                (run/'predictions').mkdir();(run/'predictions/best_pp_continuous.json').symlink_to(oldrun/'predictions/best_pp_continuous.json')
                ev=run/'evaluations/continuous_attempt002';ev.mkdir(parents=True)
                result=deepcopy(baseline);result.update(config=cfg,epoch=1,checkpoint_sha256=report.sha(run/'checkpoints/best.pt'),scheme=arm,threshold_sweep=None)
                save(ev/'metrics.json',result)
                source=oldrun/'evaluations/continuous'
                for path in source.iterdir():
                    if path.name!='metrics.json' and path.is_file():(ev/path.name).symlink_to(path)
                state['runs'][arm]=str(run);state['configs'][arm]=str(preset_path);state['evaluations'][arm]=str(ev/'metrics.json')
                state['steps'][arm+'_evaluate']=dict(status='completed',command=['python','--output',str(ev)])
            save(task/'task.json',state);before=report.sha(task/'task.json');out=task/'report'
            result=report.report(task,out,historical)
            failed=[x for x in result['checks'] if not x['passed']]
            self.assertEqual(failed,[])
            self.assertEqual(result['internal_online_candidate'],'F0')
            self.assertEqual(result['paired']['buckets']['old45FN']['arms']['concat']['tp'],0)
            for filename in ('comparison.csv','comparison.md','analysis.json','summary.md'):self.assertTrue((out/filename).is_file())
            report.report(task,out,historical)
            self.assertEqual(before,report.sha(task/'task.json'))
            # Sampling drift invalidates adoption without suppressing actual scores.
            path=task/'concat/attempts/001/epochs.jsonl';values=report.jsonl(path);values[0]['schedule_sha256']='corrupted';lines(path,values)
            result=report.report(task,out,historical)
            self.assertEqual(result['status'],'audit_failed');self.assertIsNone(result['internal_online_candidate'])
            self.assertEqual(result['comparison'][2]['macro_f1'],baseline['metrics']['macro_f1'])
            self.assertEqual(before,report.sha(task/'task.json'))


if __name__=='__main__':unittest.main()
