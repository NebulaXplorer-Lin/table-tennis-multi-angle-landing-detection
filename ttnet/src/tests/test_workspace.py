"""Configuration and storage contracts that protect historical experiments."""
import contextlib
import io
import json
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ttnet.common import PROJECT,identity,save,save_jsonl,sha
from ttnet.configuration import load_config,merge
from ttnet.data_tools.frame_store import FrameStore,pack_video,image_rgb
from ttnet.data_tools.storage_check import verify_store


def worker_read(args):
    dataset,frame_id=args
    image=FrameStore(dataset).get_rgb('video',frame_id)
    return int(image[0,0,0]),image.flags.writeable


class FrameStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.dataset=Path(self.temp.name)/'dataset';self.root=Path(self.temp.name)/'store'
        self.dataset.mkdir()
        self.rows=[dict(frame_id=fid,source_frame_id=fid*5,timestamp_sec=fid*.04,video_id='video') for fid in (0,1,4,5,6)]
        self.images={r['frame_id']:np.full((48,64,3),r['frame_id'],np.uint8) for r in self.rows}

    def publish(self,reader=None):
        result=pack_video(self.root,'video',self.rows,reader or (lambda row:self.images[row['frame_id']]),
                          chunk_frames=2,source_identity='fixed')
        save(self.root/'manifest.json',dict(status='completed',storage_id='store',videos=['video']))
        save(self.dataset/'frame_store.json',dict(path='../store',storage_id='store'))
        return result

    def test_sparse_cross_chunk_readonly_and_missing(self):
        self.publish();store=FrameStore(self.dataset,max_open_chunks=1)
        for row in self.rows:
            image=store.row_rgb(row)
            np.testing.assert_array_equal(image,self.images[row['frame_id']])
            self.assertFalse(image.flags.writeable)
        with self.assertRaises(KeyError):store.get_rgb('video',2)
        with self.assertRaises(ValueError):image[0,0,0]=100
        self.assertEqual(verify_store(self.dataset,full=True)['frames'],5)

    def test_interruption_resume_and_identity_rejection(self):
        def failing(row):
            if row['frame_id']==4:raise OSError('interrupted')
            return self.images[row['frame_id']]
        with self.assertRaises(OSError):self.publish(failing)
        self.assertTrue((self.root/'video/000000.json').exists())
        self.assertFalse((self.root/'video/manifest.json').exists())
        self.publish()
        self.assertEqual(verify_store(self.dataset,full=True)['chunks'],3)
        with self.assertRaises(ValueError):
            pack_video(self.root,'video',self.rows,lambda row:self.images[row['frame_id']],chunk_frames=2,source_identity='changed')

    def test_pickle_and_spawn_workers_have_private_mappings(self):
        self.publish();store=FrameStore(self.dataset)
        store.get_rgb('video',0)
        restored=pickle.loads(pickle.dumps(store))
        self.assertEqual(len(restored._chunks),0)
        np.testing.assert_array_equal(restored.get_rgb('video',6),self.images[6])
        with multiprocessing.get_context('spawn').Pool(2) as pool:
            result=pool.map(worker_read,[(str(self.dataset),fid) for fid in (0,4,6)])
        self.assertEqual(result,[(0,False),(4,False),(6,False)])

    def test_corruption_and_incomplete_manifest_are_rejected(self):
        self.publish();path=self.root/'video/000000.npy'
        with path.open('r+b') as stream:
            stream.seek(-1,2);stream.write(b'X')
        with self.assertRaises(ValueError):verify_store(self.dataset,full=True)
        path.write_bytes(b'truncated')
        with self.assertRaises(ValueError):FrameStore(self.dataset).get_rgb('video',0)
        save(self.root/'manifest.json',dict(status='preparing',storage_id='store'))
        with self.assertRaises(ValueError):FrameStore(self.dataset)

    def test_png_and_jpeg_decoder_pixels_are_preserved(self):
        random=np.random.default_rng(42)
        for extension in ('png','jpg'):
            source=self.dataset/f'source.{extension}'
            cv2.imwrite(str(source),random.integers(0,256,(48,64,3),dtype=np.uint8))
            pixels=image_rgb(source)
            root=self.root/extension
            pack_video(root,'video',[self.rows[0]],lambda row:image_rgb(source),source_identity=extension)
            np.testing.assert_array_equal(np.load(root/'video/000000.npy')[0],pixels)

    def test_remigration_reads_shared_arrays_when_old_images_are_removed(self):
        from ttnet.data_tools.workspace_migrate import migrate_video
        self.publish()
        target=Path(self.temp.name)/'remigrated'
        vid,frames,_=migrate_video((str(self.dataset),str(target),'ball','video',self.rows,'new',2))
        self.assertEqual((vid,frames),('video',5))
        np.testing.assert_array_equal(np.load(target/'video/000000.npy')[0],self.images[0])

    def test_legacy_reader_remains_available(self):
        image=self.images[4];cv2.imwrite(str(self.dataset/'4.png'),image[:,:,::-1])
        store=FrameStore(self.dataset)
        np.testing.assert_array_equal(store.get_rgb('video',4,path='4.png'),image)

    def test_array_pilot_keeps_external_store_reference(self):
        from ttnet.events.pipeline import smoke_corpus
        from ttnet.events.data import EventCorpus
        # Use a true sparse event descriptor with a shared external NPY store.
        root=Path(self.temp.name)/'events';game=root/'game_1';game.mkdir(parents=True)
        rows=[dict(frame_id=i,timestamp_sec=i*.04,path=f'frames/{i}.jpg',label_status='unknown',xy=None) for i in range(3)]
        pack_video(self.root,'game_1',rows,lambda row:np.full((48,64,3),row['frame_id'],np.uint8))
        save(self.root/'manifest.json',dict(status='completed',storage_id='events',videos=['game_1']))
        save(root/'frame_store.json',dict(path='../store',storage_id='events'))
        save(game/'metadata.json',dict(n_frames=3,width=64,height=48))
        save_jsonl(game/'frames.jsonl',rows);save_jsonl(game/'events.jsonl',[]);save_jsonl(game/'samples.jsonl',[])
        save(root/'split.json',dict(train=['game_1'],val=[],test=[],split_id='fixture'))
        pilot=root/'pilot';smoke_corpus(root,pilot)
        corpus=EventCorpus(pilot)
        self.assertEqual(int(corpus.image('game_1',2)[0,0,0]),2)


class ConfigurationTests(unittest.TestCase):
    def test_presets_and_overrides_are_explicit(self):
        for name in ('ball_fixed3','event_tcn_frozen','event_tcn_joint','event_tcn_reference'):
            config,meta=load_config(f'configs/experiments/{name}.json')
            self.assertEqual(meta['name'],name)
            self.assertEqual(config['temporal_mode'],'fixed3')
        config,_=load_config('configs/experiments/event_tcn_joint.json',['training.lr=0.0001','seed=7'])
        self.assertEqual(config['lr'],.0001);self.assertEqual(config['seed'],7)
        self.assertTrue(Path(config['dataset']).is_absolute())
        config,_=load_config('configs/experiments/ball_fixed3.json',['global_train_scales=[[384,216],[512,288]]'])
        self.assertEqual(len(config['global_train_scales']),2)

    def test_result_index_uses_completed_evidence_and_respects_archival(self):
        from ttnet.workspace import refresh_index
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs=root/'out/experiments'
            for name,smoke in [('complete',False),('archived',False),('smoke',True)]:
                run=runs/name
                save(run/'config.yaml',dict(stage='event',scheme='fixture',smoke=smoke))
                save(run/'run.json',dict(status='trained_pending_eval'))
                save(run/'evaluations/continuous/metrics.json',dict(status='evaluated',
                    metrics=dict(macro_f1=.8),e2e_fps=75))
            # A stale legacy CSV must not bring an archived run back into view.
            (root/'experiments').mkdir()
            (root/'experiments/comparison.csv').write_text(
                'experiment_id,model,task,status\narchived,legacy,event,failed\n')
            save(root/'experiments/curation.json',dict(archived_experiment_ids=['archived']))
            with patch('ttnet.workspace.PROJECT',root),patch('ttnet.workspace.ROOT',root/'ttnet'), \
                 patch('ttnet.workspace.RUNS',runs),patch('ttnet.workspace.OUTPUT',root/'out/results'):
                refresh_index()
                rows=json.loads((root/'experiments/index.json').read_text())['experiments']
                self.assertEqual([r['experiment_id'] for r in rows],['complete'])
                self.assertEqual(rows[0]['status'],'evaluated')
                self.assertEqual(rows[0]['f1'],.8)
                self.assertEqual(rows[0]['metric_scope'],'pp_val_landing_macro_f1')
                # A retained snapshot remains readable after its runtime is moved.
                retained=dict(rows[0],experiment_id='retained',scheme='retained_reference')
                save(root/'experiments/curation.json',dict(archived_experiment_ids=['archived'],
                    retained_results=[retained]))
                refresh_index()
                rows=json.loads((root/'experiments/index.json').read_text())['experiments']
                self.assertEqual([r['experiment_id'] for r in rows],['retained','complete'])
            self.assertEqual(json.loads((runs/'complete/run.json').read_text())['status'],
                             'trained_pending_eval')

    def public_index_fixture(self,root):
        run=root/'out/experiments/public_run'
        record=root/'experiments/records/public_run'
        save(run/'config.yaml',dict(stage='event',scheme='public_fixture',smoke=False,
            dataset_id='pp154_public15',split_id='pp154_public15_event_v1',
            public_validation_root=str(root/'public')))
        save(run/'run.json',dict(status='trained_pending_eval'))
        checkpoint=run/'checkpoints/best.pt';checkpoint.parent.mkdir()
        checkpoint.write_bytes(b'fixed-checkpoint')
        # The report references a later successful attempt. Earlier output must
        # not be selected by directory naming or score.
        save(run/'evaluations/public/metrics.json',dict(status='evaluated',macro_f1=.99))
        save(run/'evaluations/continuous/metrics.json',dict(status='evaluated',
            metrics=dict(macro_f1=.98),e2e_fps=90))
        metrics=run/'evaluations/public_attempt003/metrics.json'
        save(metrics,dict(status='evaluated',validation_set='pp_public',macro_f1=.61,
            checkpoint_sha256=sha(checkpoint),e2e_fps=45,
            cache_online_parity=dict(passed=True,decisions_equal=True,landing_decisions_equal=True)))
        study=root/'out/study/completion.json'
        save(study,dict(status='completed',validation_set='pp_public',all_checks_passed=True))
        reference=dict(status='completed',validation_set='pp_public',eval_id='public',
            metrics=str(metrics),metrics_sha256=sha(metrics),checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),best_epoch=7)
        completion=record/'public_completion.json'
        save(completion,dict(schema='public_event_run_completion_v1',status='completed',
            experiment_id=run.name,evaluation=reference,study_completion=str(study),
            study_completion_sha256=sha(study)))
        summary=record/'summary.md';summary.write_text('# Completed public result\n')
        return dict(run=run,record=record,metrics=metrics,checkpoint=checkpoint,
                    completion=completion,study=study,summary=summary)

    def public_index_rows(self,root):
        from ttnet.workspace import refresh_index
        with patch.multiple('ttnet.workspace',PROJECT=root,ROOT=root/'ttnet',
                RUNS=root/'out/experiments',OUTPUT=root/'out/results'):
            refresh_index()
        return json.loads((root/'experiments/index.json').read_text())['experiments']

    def test_public_index_uses_bound_completion_attempt_and_preserves_training_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);files=self.public_index_fixture(root)
            state_hash=sha(files['run']/'run.json')
            rows=self.public_index_rows(root)
            self.assertEqual(len(rows),1)
            row=rows[0]
            self.assertEqual(row['status'],'completed')
            self.assertEqual(row['metric_scope'],'pp_public_validation_landing_macro_f1')
            self.assertEqual(row['eval_id'],'public')
            self.assertEqual(row['f1'],.61)
            self.assertEqual(row['e2e_fps'],45)
            self.assertEqual(row['best_epoch'],7)
            self.assertEqual(row['summary'],'experiments/records/public_run/summary.md')
            self.assertEqual(sha(files['run']/'run.json'),state_hash)

    def test_public_index_does_not_promote_missing_failed_or_changed_evidence(self):
        cases=('missing_receipt','failed_parity','changed_metrics','changed_checkpoint',
               'failed_study','missing_summary','wrong_run','outside_run','malformed_receipt')
        for case in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);files=self.public_index_fixture(root)
                receipt=json.loads(files['completion'].read_text())
                if case=='missing_receipt':
                    files['completion'].unlink()
                elif case=='failed_parity':
                    metric=json.loads(files['metrics'].read_text())
                    metric['cache_online_parity']['landing_decisions_equal']=False
                    save(files['metrics'],metric)
                    receipt['evaluation']['metrics_sha256']=sha(files['metrics'])
                    save(files['completion'],receipt)
                elif case=='changed_metrics':
                    files['metrics'].write_text(files['metrics'].read_text()+'\n')
                elif case=='changed_checkpoint':
                    files['checkpoint'].write_bytes(b'replaced-checkpoint')
                elif case=='failed_study':
                    save(files['study'],dict(status='completed',validation_set='pp_public',all_checks_passed=False))
                    receipt['study_completion_sha256']=sha(files['study'])
                    save(files['completion'],receipt)
                elif case=='missing_summary':
                    files['summary'].unlink()
                elif case=='wrong_run':
                    receipt['experiment_id']='another-run';save(files['completion'],receipt)
                elif case=='outside_run':
                    outside=root/'different-run/metrics.json'
                    save(outside,json.loads(files['metrics'].read_text()))
                    receipt['evaluation']['metrics']=str(outside)
                    receipt['evaluation']['metrics_sha256']=sha(outside)
                    save(files['completion'],receipt)
                elif case=='malformed_receipt':
                    save(files['completion'],[])
                rows=self.public_index_rows(root)
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['metric_scope'],'unscored')
                self.assertIsNone(rows[0]['f1'])
                self.assertEqual(rows[0]['status'],'trained_pending_eval')

    def test_unknown_unimplemented_and_mistyped_overrides_fail(self):
        for override in ('lrr=0.1','event_head.type=invalid','tracker.type=invalid','seed=true','window=25'):
            with self.subTest(override=override),self.assertRaises(ValueError):
                load_config('configs/experiments/event_tcn_frozen.json',[override])

    def test_merge_replaces_lists_and_preserves_legacy_config(self):
        self.assertEqual(merge({'x':[1,2],'a':{'b':1}},{'x':[3],'a':{'b':2}}),{'x':[3],'a':{'b':2}})
        path=PROJECT/'ttnet/src/configs/fixed3.json'
        config,meta=load_config(path)
        self.assertEqual(config,json.loads(path.read_text()));self.assertTrue(meta['legacy'])

    def test_cli_dry_run_does_not_create_run_or_records(self):
        from ttnet.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'dry-run-never-created'
            record=PROJECT/'experiments/records'/run.name
            before=record.exists()
            with patch('sys.argv',['ttnet.cli','train','--config','configs/experiments/event_tcn_frozen.json',
                                  '--run',str(run),'--set','seed=8','--dry-run']),contextlib.redirect_stdout(io.StringIO()) as stream:
                main()
            plan=json.loads(stream.getvalue())
            self.assertEqual(plan['config']['seed'],8)
            self.assertIn('ttnet.events.train',plan['commands'][0])
            self.assertFalse(run.exists());self.assertEqual(record.exists(),before)

    def test_cli_subprocess_logs_provenance_and_failed_launches(self):
        from types import SimpleNamespace
        from ttnet.cli import run_experiment
        from ttnet.workspace import refresh_index
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'dataset';dataset.mkdir()
            source=root/'source.json';source.write_text('{}')
            for success in (True,False):
                run=root/'out/experiments'/('success' if success else 'failure')
                record=root/'experiments/records'/run.name
                code=("import json,sys;from pathlib import Path;p=Path(sys.argv[1]);p.mkdir(parents=True);"
                      "(p/'run.json').write_text(json.dumps({'status':'completed_smoke'}));"
                      "(p/'config.yaml').write_text(json.dumps({'stage':'global','scheme':'fixture'}));print('fixture done')") if success else "print('fixture failed');raise SystemExit(4)"
                plan=dict(dataset=str(dataset),run=str(run),record=str(record),scheme='fixture',config={},
                    provenance=dict(config_hash=identity({}),sources=[str(source)]),commands=[[sys.executable,'-c',code,str(run)]])
                args=SimpleNamespace(command='train',resume=False,dry_run=False)
                with patch('ttnet.cli.PROJECT',root),patch('ttnet.cli.execution_plan',return_value=plan),patch('ttnet.workspace.refresh_index'),contextlib.redirect_stdout(io.StringIO()):
                    if success:run_experiment(args)
                    else:
                        with self.assertRaises(subprocess.CalledProcessError):run_experiment(args)
                attempt=root/'out/launches'/run.name/'attempt_001'
                self.assertTrue((attempt/'console.log').is_file())
                self.assertEqual(json.loads((attempt/'exit.json').read_text())['exit_code'],0 if success else 4)
                if success:
                    provenance=json.loads((record/'provenance.json').read_text())
                    self.assertEqual(provenance['effective_config_hash'],identity(json.loads((run/'config.yaml').read_text())))
                else:self.assertEqual(json.loads((record/'run.json').read_text())['status'],'launcher_failed')
            with patch.multiple('ttnet.workspace',PROJECT=root,ROOT=root/'ttnet',RUNS=root/'out/experiments',OUTPUT=root/'out/results'):
                refresh_index()
            rows=json.loads((root/'experiments/index.json').read_text())['experiments']
            self.assertTrue(any(r['status']=='launcher_failed' for r in rows))


class InputCacheTests(unittest.TestCase):
    def test_array_input_and_crop_caches_match_direct_preprocessing(self):
        from ttnet.training.cache import ensure_global_cache,ensure_local_cache
        from ttnet.training.dataset import BallDataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'dataset'
            generator=np.random.default_rng(42);maps={}
            for split,video in [('train','a'),('val','b')]:
                rows=[];maps[split]={}
                for fid in range(3):
                    path=dataset/'frames'/video/f'{fid}.png';path.parent.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(path),generator.integers(0,256,(64,64,3),dtype=np.uint8))
                    rows.append(dict(video_id=video,frame_id=fid,timestamp_sec=fid*.04,frame_path=str(path.relative_to(dataset)),
                       resolution=dict(width=64,height=64),label_status='positive',ball_centers=[[32.,32.]]))
                    maps[split][video,fid]=dict(x=32.,y=32.,valid=True,confidence=1.)
                save_jsonl(dataset/'labels'/f'{split}.jsonl',rows)
            save(dataset/'metadata/dataset.json',dict(dataset_id='fixture'))
            config=dict(workers=1,local_input=[32,32],temporal_mode='fixed3')
            with patch('ttnet.training.cache.CACHE',root/'cache'):
                global_cache=ensure_global_cache(dataset,[(32,32)],workers=1)
                local_cache=ensure_local_cache(dataset,maps,'global-hash',config)
            for stage in ('global','local'):
                options=dict(stage=stage,input_size=(32,32),temporal_mode='fixed3',predictions=maps['train'])
                direct=BallDataset(dataset,'train',**options)
                cached=BallDataset(dataset,'train',**options,input_cache=global_cache,local_cache=local_cache)
                for gt in (False,True):
                    direct.gt_crops=cached.gt_crops=gt
                    for i in range(3):
                        np.random.seed(42);x0,y0,v0,_=direct[i]
                        np.random.seed(42);x1,y1,v1,_=cached[i]
                        for key in x0:np.testing.assert_array_equal(x0[key],x1[key])
                        np.testing.assert_array_equal(y0,y1);self.assertEqual(float(v0),float(v1))
                neutral=BallDataset(dataset,'train',**options,augment=True,
                    augmentation=dict(horizontal_flip=0.,brightness_scale=[1.,1.],brightness_offset=[0.,0.],gt_crop_jitter_fraction=0.))
                direct.gt_crops=False
                x0,y0,_,_=direct[1];x1,y1,_,_=neutral[1]
                for key in x0:np.testing.assert_array_equal(x0[key],x1[key])
                np.testing.assert_array_equal(y0,y1)


if __name__=='__main__':unittest.main()
