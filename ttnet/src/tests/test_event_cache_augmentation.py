"""Ball-cache immutability and interrupted table-cache recovery."""
from copy import deepcopy
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ttnet.common import jsonl, save, save_jsonl, sha
from ttnet.scene.augment_event_cache import augment_cache, verify_base_identity
from ttnet.scene.visual_cache import restore_visual, table_image_batch


class EventCacheAugmentationTests(unittest.TestCase):
    def test_rgb_transposed_views_become_online_equivalent_contiguous_nchw(self):
        images = [np.arange(32*48*3,dtype=np.uint8).reshape(32,48,3) for _ in range(2)]
        # A plain stack of these CHW views is physically channel-last.
        old = torch.from_numpy(np.stack([im.transpose(2,0,1) for im in images]))
        self.assertFalse(old.is_contiguous())
        actual = table_image_batch(images,torch.device('cpu'))
        online = torch.cat([torch.from_numpy(im.transpose(2,0,1).copy()).float()[None]/255. for im in images])
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(actual.stride(),online.stride())
        torch.testing.assert_close(actual,online,atol=0,rtol=0)

    def test_compressed_arrays_are_read_once_independent_of_frame_count(self):
        class CountReads(Mapping):
            def __init__(self, archive):
                self.archive, self.reads = archive, Counter()
            def __iter__(self):return iter(self.archive.files)
            def __len__(self):return len(self.archive.files)
            def __getitem__(self, key):
                self.reads[key] += 1
                return self.archive[key]
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'compressed.npz'; count = 32
            arrays = dict(table_roi=np.arange(count*32,dtype=np.float32).reshape(count,32),
                          table_context=np.ones((count,2),np.float32),
                          table_roi_valid=np.ones(count,bool),table_context_valid=np.ones(count,bool))
            np.savez_compressed(path, **arrays)
            records = {i:dict(table=None) for i in range(count)}
            with np.load(path) as archive:
                counted = CountReads(archive)
                restore_visual(counted,list(range(count)),records,2)
                self.assertEqual(counted.reads,Counter({key:1 for key in arrays}))
            for i in range(count):
                np.testing.assert_array_equal(records[i]['table']['visual']['roi'].numpy(),arrays['table_roi'][i])
                self.assertTrue(records[i]['table']['visual']['context_valid'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base, self.dest = self.root/'base', self.root/'derived'
        self.base.mkdir()
        self.previous = dict(schema=3, dataset_id='frozen-data', global_sha256='global-B',
            local_sha256='paired-local', locator_thresholds=dict(global_threshold=.7,local_threshold=.7),
            implementation=dict(decoder='frozen-code'), table=dict(checkpoint_sha256=None,source='predicted'))
        self.expected = dict(self.previous, schema=4, table_visual=dict(checkpoint_sha256='yolo',channels=2))
        save(self.base/'identity.json', self.previous)
        self.features = np.arange(512,dtype=np.float32).reshape(2,256)
        np.savez_compressed(self.base/'v.npz',frame_ids=np.array([0,1]),features=self.features)
        self.records = [dict(frame_id=i, prediction=dict(valid=True,x=i,y=3,confidence=.9),
            crop_transform=dict(left=i,side=20),local_valid=bool(i),table=None) for i in range(2)]
        save_jsonl(self.base/'v.jsonl', self.records)
        self.corpus = SimpleNamespace(split=dict(train=['v'],val=[]),videos=dict(v=dict(frames={0:{},1:{}})))
        self.system = SimpleNamespace(device=torch.device('cpu'),config=dict(amp=False,table_frontend=dict(source='independent_yolo',channels=2)),
                                      configure=lambda *args:None)
        self.payload = {i:dict(valid=True,bbox_xyxy=[0,0,8,8],score=.9,source='predicted',
            visual=dict(roi=torch.full((32,),float(i)),context=torch.tensor([1.,2.]),roi_valid=True,context_valid=True))
            for i in range(2)}

    def run_augmentation(self):
        with patch('ttnet.scene.augment_event_cache.cache_identity',return_value=self.expected), \
             patch('ttnet.scene.augment_event_cache.infer_tables',return_value=self.payload) as inference:
            augment_cache(self.system,self.corpus,self.base,self.dest,batch=2)
        return inference.call_count

    def test_changed_ball_threshold_or_implementation_is_rejected(self):
        verify_base_identity(self.previous,self.expected)
        for key, value in [('locator_thresholds',dict(global_threshold=.7,local_threshold=.9)),
                           ('implementation',dict(decoder='changed-code'))]:
            changed = deepcopy(self.expected);changed[key] = value
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                verify_base_identity(self.previous,changed)

    def test_roundtrip_preserves_ball_data_and_complete_resume_skips_inference(self):
        before = [sha(self.base/f'v{suffix}') for suffix in ('.npz','.jsonl')]
        self.assertEqual(self.run_augmentation(),1)
        with np.load(self.dest/'v.npz') as arrays:
            np.testing.assert_array_equal(arrays['features'],self.features)
            self.assertEqual(arrays['table_roi'].shape,(2,32))
        actual = jsonl(self.dest/'v.jsonl')
        for original, result in zip(self.records, actual):
            self.assertEqual({k:v for k,v in original.items() if k!='table'},
                             {k:v for k,v in result.items() if k!='table'})
        self.assertEqual([sha(self.base/f'v{suffix}') for suffix in ('.npz','.jsonl')],before)
        self.assertEqual(self.run_augmentation(),0)

    def test_uncommitted_pair_and_corrupt_outputs_are_rebuilt(self):
        self.run_augmentation()
        path = self.dest/'derivation.json'
        marker = json.loads(path.read_text());marker['completed'] = {};save(path,marker)
        # Both files exist, but completion was interrupted: they must be rebuilt.
        self.assertEqual(self.run_augmentation(),1)
        (self.dest/'v.npz').write_bytes(b'corrupt')
        self.assertEqual(self.run_augmentation(),1)
        with np.load(self.dest/'v.npz') as arrays:
            np.testing.assert_array_equal(arrays['features'],self.features)

    def test_source_mutation_is_rejected_even_after_completion(self):
        self.run_augmentation()
        changed = deepcopy(self.records);changed[0]['prediction']['x'] = 123
        save_jsonl(self.base/'v.jsonl',changed)
        with self.assertRaisesRegex(ValueError,'derivation changed on resume: sources'):
            self.run_augmentation()

    def test_resume_rejects_changed_batch(self):
        self.run_augmentation()
        with patch('ttnet.scene.augment_event_cache.cache_identity',return_value=self.expected):
            with self.assertRaisesRegex(ValueError,'table_inference_batch'):
                augment_cache(self.system,self.corpus,self.base,self.dest,batch=8)


if __name__ == '__main__':
    unittest.main()
