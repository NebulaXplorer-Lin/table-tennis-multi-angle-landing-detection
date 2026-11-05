"""Preparation contracts; no optimizer steps, training jobs or saved trained weights."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ttnet.configuration import load_config
from ttnet.common import PROJECT
from ttnet.data_tools.geometry import letterbox_transform
from ttnet.data_tools.prepare_table import sample_epoch, transform_frame
from ttnet.events.data import assemble_window
from ttnet.events.engine import EventSystem, cache_identity, build_cache, load_cache
from ttnet.models.event import TableFusionEventHead
from ttnet.models.table_encoder import SharedGlobalTableHead, select_table, spatial_visual
from ttnet.scene.table import geometry
from ttnet.scene.visual_cache import batch_windows, serialize_visual, restore_visual
from ttnet.tests.test_events import TinyCorpus


class TableVisualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_sampling_rotates_and_rejects_unknown_or_validation(self):
        rows = [dict(video_id=f'{g}_v',frame_id=i,split='train',table_status='positive',
                     table_bbox_xyxy=[0,0,50,2 if i<3 else 30],resolution=dict(height=100,width=100))
                for g in ('g1','g2') for i in range(30)]
        groups = {'g1_v':'g1','g2_v':'g2'}
        first = sample_epoch(rows,groups,42,0,12)
        self.assertEqual(first,sample_epoch(rows,groups,42,0,12))
        self.assertNotEqual(first,sample_epoch(rows,groups,42,1,12))
        keys = {(r['video_id'],r['frame_id']) for e in range(20) for r in sample_epoch(rows,groups,42,e,12)}
        self.assertEqual(len(keys),60)
        self.assertEqual(len({(r['video_id'],r['frame_id']) for r in first}),12)
        with self.assertRaises(ValueError):
            sample_epoch([dict(rows[0],split='val')],groups,42,0)
        with self.assertRaises(ValueError):
            sample_epoch([dict(rows[0],table_status='unknown')],groups,42,0)

    def test_flip_preserves_pixel_centres_and_box_edges(self):
        rgb = np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
        actual,b,_,_ = transform_frame(rgb,[1,2,4,6],[8,8],np.random.default_rng(3),
                                       dict(horizontal_flip=1.,translate=0.,scale=[1.,1.]))
        np.testing.assert_array_equal(actual,rgb[:,::-1])
        np.testing.assert_allclose(b,[.5,.25,.875,.75])

    def test_dense_shared_head_supervision_and_ordered_boxes(self):
        head = SharedGlobalTableHead()
        spatial = torch.randn(2,32,12,20,requires_grad=True)
        prediction,returned = head(spatial)
        self.assertIs(returned,spatial)
        targets = torch.tensor([[.1,.3,.9,.4],[.2,.2,.7,.8]])
        loss,parts = head.loss(prediction,targets)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(set(parts),{'center','l1','giou'})
        loss.backward()
        self.assertGreater(float(spatial.grad.abs().sum()),0.)
        candidates = head.candidates(prediction,confidence=0.,input_size=(160,96))
        for c in candidates:
            self.assertTrue((c[:,2:4]>c[:,:2]).all())

    def test_roi_alignment_layout_and_gradient_without_coordinate_gradient(self):
        spatial = torch.arange(4).float().repeat(4,1)[None].requires_grad_()
        transform = letterbox_transform(32,32,32,32)
        table = dict(valid=True,bbox_xyxy=[0,0,32,32])
        payload = spatial_visual(spatial,table,transform,dict(width_scale=1.,height_scale=1.,min_height=.01))
        torch.testing.assert_close(payload['roi'].reshape(4,4),spatial[0])
        (payload['roi'].sum()+payload['context'].sum()).backward()
        self.assertTrue(torch.isfinite(spatial.grad).all());self.assertGreater(float(spatial.grad.sum()),0)
        missing = spatial_visual(spatial,dict(valid=False),transform)
        self.assertFalse(missing['roi_valid']);self.assertTrue(missing['context_valid'])
        self.assertEqual(float(missing['roi'].abs().sum()),0.)

    def test_prediction_selection_never_needs_gt(self):
        candidates = torch.tensor([[1,1,10,10,.9,0],[0,0,20,20,.9,0],[float('nan'),0,2,2,1,0]])
        table = select_table(candidates,letterbox_transform(32,32,32,32))
        self.assertEqual(table['bbox_xyxy'],[0.,0.,20.,20.])
        self.assertFalse(select_table(torch.empty(0,6),letterbox_transform(32,32))['valid'])

    def test_table_masking_zero_control_and_event_gradient(self):
        f,m,v = torch.randn(2,13,256),torch.zeros(2,13,18),torch.ones(2,13,dtype=torch.bool)
        m[:,:,7]=1;m[:,:,15]=1
        table = dict(roi=torch.randn(2,13,32,requires_grad=True),context=torch.randn(2,13,2,requires_grad=True),
                     roi_valid=torch.ones(2,13,dtype=torch.bool),context_valid=torch.ones(2,13,dtype=torch.bool))
        model = TableFusionEventHead(2,'concat').eval()
        model(f,m,v,table).square().sum().backward()
        self.assertGreater(float(table['roi'].grad.abs().sum()),0)
        zero = TableFusionEventHead(2,'zero').eval()
        original = zero(f,m,v,table)
        changed = dict(table,roi=table['roi'].detach()*100,context=table['context'].detach()*100)
        torch.testing.assert_close(original,zero(f,m,v,changed),atol=0,rtol=0)
        invalid = {k:torch.full_like(value,float('nan')) if value.dtype.is_floating_point else torch.zeros_like(value) for k,value in table.items()}
        self.assertTrue(torch.isfinite(model(f,m,v,invalid)).all())
        v[:]=False;f[:]=float('nan');m[:]=float('nan')
        self.assertTrue(torch.isfinite(model(f,m,v,invalid)).all())

    def test_geometry_bounds_thin_boxes_without_masking_ball(self):
        table = dict(valid=True,bbox_xyxy=[20,30,80,30.1])
        p = dict(valid=True,x=50,y=90)
        g = geometry(table,p,100,100,stable=True)
        self.assertEqual(g[5],4.);self.assertEqual(g[6],1);self.assertEqual(g[7],1)

    def test_cache_roundtrip_and_structured_batching(self):
        payload = dict(roi=torch.arange(32).float(),context=torch.tensor([2.,3.]),roi_valid=True,context_valid=True)
        records = [dict(frame_id=4,table=dict(valid=True,bbox_xyxy=[1,2,3,4],visual=payload))]
        arrays,clean = serialize_visual(records,2)
        self.assertIn('visual',records[0]['table']);self.assertNotIn('visual',clean[0]['table'])
        json.dumps(clean)
        restored = {4:clean[0]};restore_visual(arrays,[4],restored,2)
        torch.testing.assert_close(restored[4]['table']['visual']['roi'],payload['roi'])
        inp = (torch.ones(1,13,256),torch.zeros(1,13,18),torch.ones(1,13,dtype=torch.bool),
               dict(roi=torch.zeros(1,13,32),context=torch.ones(1,13,2)))
        actual = batch_windows([inp,inp],'cpu')
        self.assertEqual(actual[0].shape,(2,13,256));self.assertEqual(actual[3]['context'].shape,(2,13,2))

    def test_shared_B_runs_Global_once_and_batch_parity(self):
        from ttnet.events.batched_features import independent_features
        cfg = dict(tracker_type='independent',temporal_mode='fixed3',global_localizer_type='high_resolution_se',
                   global_input=[64,32],local_input=[32,32],local_crop_side_1080=256,localization_threshold=.1,
                   global_threshold=.1,local_threshold=.9,sigma=1.5,amp=False,
                   table_frontend=dict(source='shared_global',checkpoint='unused.pt',channels=32,input=[64,32],
                                       confidence=0.,nms_iou=.7,roi=dict(width_scale=1.1,height_scale=1.5,min_height=.08)))
        system = EventSystem(cfg,torch.device('cpu'),initialize=False);system.configure('frozen',False)
        corpus = TinyCorpus()
        with torch.inference_mode(),patch.object(system.global_model,'forward',wraps=system.global_model.forward) as call:
            a,ar,_ = system.forward_frames(corpus,'v',[5])
            self.assertEqual(call.call_count,1)
            b,br,_ = independent_features(system,corpus,'v',[5])
        torch.testing.assert_close(a[5],b[5])
        torch.testing.assert_close(ar[5]['table']['visual']['roi'],br[5]['table']['visual']['roi'])
        self.assertFalse(ar[5]['table']['visual']['roi'].requires_grad)
        with self.assertRaisesRegex(ValueError,'later stage'):
            system.configure('joint',True)

    def test_presets_and_cache_threshold_identity(self):
        configs = [load_config(p)[0] for p in (PROJECT/'configs/experiments').glob('event_table_*.json')]
        self.assertEqual(len(configs),9)
        for cfg in configs:
            self.assertEqual(cfg['mode'],'frozen');self.assertIsNone(cfg['table_checkpoint'])
            self.assertEqual((cfg['global_threshold'],cfg['local_threshold']),(.7,.9))
        cfg = load_config('configs/experiments/event_table_shared_B_concat.json')[0]
        with patch('ttnet.events.engine.sha',return_value='hash'):
            original = cache_identity(SimpleNamespace(config=cfg),SimpleNamespace(identity='data'))
            changed = deepcopy(cfg);changed['local_threshold']=.1
            altered = cache_identity(SimpleNamespace(config=changed),SimpleNamespace(identity='data'))
        self.assertNotEqual(original,altered)


if __name__ == '__main__':
    unittest.main()
