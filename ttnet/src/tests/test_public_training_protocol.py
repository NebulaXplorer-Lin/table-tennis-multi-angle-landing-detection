"""Public routing, supervised-loss isolation and explicit cache migration."""
from argparse import Namespace
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_public_event_study as study
from ttnet.cli import execution_plan
from ttnet.common import save, save_jsonl, sha
from ttnet.configuration import load_config
from ttnet.events.continuous import mixed_samples
from ttnet.events.engine import load_cache
from ttnet.events.public_validation import attach_public_validation
from ttnet.events.train import validate
from ttnet.tests import test_public_validation as public_fixture


class PublicTrainingProtocolTests(unittest.TestCase):
    def arguments(self, run):
        return Namespace(command='evaluate',run=run,config=None,set=[],resume=False,
            dataset=None,global_checkpoint=None,local_checkpoint=None,stage=None,
            device='cpu',smoke=False)

    def test_saved_public_run_routes_to_public_evaluator_without_legacy_pp_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'saved-run'
            config,_=load_config('configs/experiments/event_public_B_concat_seed42.json')
            save(run/'config.yaml',dict(config,stage='event',scheme='public_concat'))
            args=self.arguments(run)
            command=execution_plan(args)['commands'][0]
            self.assertEqual(command[3],'scripts.evaluate_public_event')
            self.assertEqual(command[command.index('--public')+1],config['public_validation_root'])
            self.assertEqual(command[command.index('--checkpoint')+1],str(run/'checkpoints/best.pt'))
            self.assertNotIn('--annotations',command)
            args.dataset=Path(temp)/'different-dataset'
            with self.assertRaisesRegex(ValueError,'saved in the run'):
                execution_plan(args)
            args.dataset=None
            # Existing saved runs continue to evaluate their original PP40 split.
            config.pop('public_validation_root')
            save(run/'config.yaml',dict(config,stage='event',scheme='legacy_A'))
            self.assertEqual(execution_plan(args)['commands'][0][3],'ttnet.events.comparison')

    def test_public_gt_and_old_pp_val_do_not_enter_auxiliary_validation_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            corpus,_=public_fixture.PublicValidationTests().fixture(Path(temp))
            for rows in corpus.samples.values():
                for sample in rows:
                    sample.update(original=True,anchor=sample['frame_id'],hard_target=sample['target'],
                                  source='pp' if sample['video_id'].startswith('pp_') else 'opentt')
            # Give the training scheduler all three sparse anchor categories.
            for index,kind in enumerate(('bounce','net','empty_event')):
                target=[float(kind=='bounce'),float(kind=='net')]
                corpus.samples['train'].append(dict(video_id='game_1',frame_id=index,anchor=index,
                    group=kind,target=target,hard_target=target,original=True,supervised=[True,True]))
            attach_public_validation(corpus,temp)
            system=SimpleNamespace(device=torch.device('cpu'),config=dict(amp=False,validation_batch=8))
            visited=[]
            def forward(_system,_corpus,_cache,sample):
                visited.append(sample['video_id'])
                return torch.tensor([[2.,-1.]])
            cache={vid:(None,None) for vid in corpus.split['val']}
            with patch('ttnet.events.train.cached_forward',side_effect=forward):
                first,rows=validate(system,corpus,cache)
                corpus.public_ground_truth['public_v'][0].update(frame_id=9,x=63.,y=47.)
                second,after=validate(system,corpus,cache)
            self.assertEqual(first,second)
            self.assertEqual(rows,after)
            self.assertEqual(visited,['game_5','game_5'])
            self.assertEqual([r['video_id'] for r in rows],['game_5'])
            schedule=mixed_samples(corpus.samples['train'],42,0)
            self.assertTrue({'pp_a','pp_b'} <= {s['video_id'] for s in schedule})
            self.assertFalse(any(s['video_id'].startswith('public_') for s in schedule))
            self.assertFalse(set(corpus.split['train']) & set(corpus.split['val']))
            self.assertEqual(sum(s.get('source')=='pp' for s in schedule),len(schedule)//2)
            corpus.image.cache_clear()

    def test_old_split_cache_is_rejected_by_normal_loader_before_reading_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            save(root/'identity.json',dict(dataset_id='pp114_pp40',amp=True))
            corpus=SimpleNamespace(split=dict(train=['missing'],val=['public_v']))
            with self.assertRaisesRegex(ValueError,'feature cache identity mismatch'):
                load_cache(corpus,root,expected=dict(dataset_id='pp154_public15',amp=True))

    def cache_migration(self, root, source_identity):
        source,dest=root/'source',root/'target'
        save(source/'identity.json',source_identity)
        np.savez_compressed(source/'pp_a.npz',frame_ids=np.array([0]),features=np.zeros((1,256),np.float32))
        save_jsonl(source/'pp_a.jsonl',[dict(frame_id=0,prediction=None)])
        corpus=SimpleNamespace(identity='old-corpus',
            split=dict(split_id='new-split',train=['pp_a'],val=['public_v'],test=[]),
            videos={vid:dict(frames={0:{}}) for vid in ('pp_a','public_v')})
        def attach_public(corpus,*_args):
            corpus.identity='new-corpus';corpus.public_split_audit=dict(public_training_samples=0)
        def build(_system,_corpus,path):
            np.savez_compressed(path/'public_v.npz',frame_ids=np.array([0]),features=np.ones((1,256),np.float32))
            save_jsonl(path/'public_v.jsonl',[dict(frame_id=0,prediction=None)])
        expected=dict(dataset_id='new-corpus',amp=True,global_sha256='fixed-locator')
        cfg=dict(torch_threads=1,dataset='fixture',pp_annotations='fixture',public_validation_root='fixture')
        stack=ExitStack()
        for target,value in [('PROJECT',root),('load_config',lambda _path:(cfg,{})),
                ('EventCorpus',lambda _path:corpus),('attach_pp',lambda *_args:None),
                ('attach_public_validation',attach_public),('EventSystem',lambda *_args:object()),
                ('cache_identity',lambda *_args:copy.deepcopy(expected)),
                ('feature_cache_path',lambda *_args:dest)]:
            stack.enter_context(patch.object(study,target,value))
        builder=stack.enter_context(patch.object(study,'build_cache',side_effect=build))
        stack.enter_context(patch.object(study.torch,'set_num_threads'))
        return stack,builder,source,dest

    def test_explicit_migration_copies_exact_old_features_and_records_new_public_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            stack,builder,source,dest=self.cache_migration(root,
                dict(dataset_id='old-corpus',amp=True,global_sha256='fixed-locator'))
            with stack:
                study.prepare_cache(root/'config.json',source,root/'receipt.json')
            builder.assert_called_once()
            self.assertEqual(sha(source/'pp_a.npz'),sha(dest/'pp_a.npz'))
            self.assertEqual(sha(source/'pp_a.jsonl'),sha(dest/'pp_a.jsonl'))
            provenance=json.loads((dest/'public_split_derivation.json').read_text())
            self.assertEqual(provenance['status'],'completed')
            self.assertEqual(set(provenance['reused']),{'pp_a'})
            self.assertEqual(set(provenance['added']),{'public_v'})
            self.assertEqual(provenance['target_identity']['dataset_id'],'new-corpus')
            self.assertEqual(provenance['source_identity']['dataset_id'],'old-corpus')

    def test_migration_rejects_any_non_split_inference_change_without_building_public_cache(self):
        for changed in ('amp','global_sha256','dataset_id'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as temp:
                previous=dict(dataset_id='old-corpus',amp=True,global_sha256='fixed-locator')
                previous[changed]=False if changed=='amp' else 'different'
                root=Path(temp)
                stack,builder,source,dest=self.cache_migration(root,previous)
                with stack,self.assertRaisesRegex(ValueError,'refusing cache reuse'):
                    study.prepare_cache(root/'config.json',source,root/'receipt.json')
                builder.assert_not_called()
                self.assertFalse(dest.exists())


if __name__=='__main__':
    unittest.main()
