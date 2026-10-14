import copy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from ttnet.common import save_jsonl
from ttnet.models.localizer import BallLocalizer,LocalizationLoss
from ttnet.models.features import FeatureWindowCache
from ttnet.temporal import sample_window
from ttnet.training.dataset import BallDataset


class TemporalSamplingTests(unittest.TestCase):
    def test_fixed3_stays_three_native_frames_at_all_fps(self):
        for fps in [25,30,60,120]:
            w=sample_window(np.arange(11)/fps,5,'fixed3')
            self.assertEqual(w['frame_ids'],[4,5,6]);self.assertEqual(w['frame_valid'],[True]*3)
            self.assertAlmostEqual(w['actual_span_sec'],2/fps)

    def test_masked9_keeps_center_and_actual_time_span(self):
        for fps,n in [(25,3),(30,3),(60,5),(120,9)]:
            w=sample_window(np.arange(11)/fps,5,'fps_mask9')
            self.assertEqual(w['frame_ids'][4],5);self.assertEqual(sum(w['frame_valid']),n)
        self.assertEqual(sample_window([0,.04],0,'fixed3')['frame_ids'],[None,0,1])
        self.assertEqual(sample_window([0],0,'fixed3')['frame_ids'],[None,0,None])

    def test_event_cache_obeys_both_modes(self):
        for mode,n in [('fixed3',3),('fps_mask9',9)]:
            c=FeatureWindowCache('d','g','l',temporal_mode=mode);c.reset('v')
            for i in range(7):c.append(i,i*.04,np.ones(128)*i,np.ones(128),True,{'frame_id':i})
            w=c.window(3)
            self.assertEqual(w['features'].shape,(1,n,256))
            self.assertEqual(w['center_frame_id'],3)
            self.assertEqual(int(w['frame_valid'].sum()),3)


class TemporalModelTests(unittest.TestCase):
    def test_masked_images_and_times_cannot_affect_prediction(self):
        torch.set_num_threads(2)
        for mode,t in [('fixed3',3),('fps_mask9',9)]:
            m=BallLocalizer(mode).eval()
            x=torch.randn(1,t,3,64,96);valid=torch.zeros(1,t,dtype=torch.bool)
            valid[:,t//2]=True
            times=torch.zeros(1,t)
            with torch.inference_mode():a=m(x,valid,times)
            changed=x.clone();changed[:,~valid[0]]=torch.randn_like(changed[:,~valid[0]])*100
            changed_times=times.clone();changed_times[:,~valid[0]]=100
            with torch.inference_mode():b=m(changed,valid,changed_times)
            for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
            self.assertEqual(a[0].shape,(1,1,32,48));self.assertEqual(a[1].shape,(1,128))

    def test_context_changes_center_output_and_receives_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        model=BallLocalizer('fixed3')
        x=torch.randn(1,3,3,64,96,requires_grad=True)
        valid=torch.ones(1,3,dtype=torch.bool);times=torch.tensor([[-.04,0.,.04]])
        heat,_=model(x,valid,times)
        heat.square().mean().backward()
        self.assertTrue(all(float(x.grad[:,i].abs().sum())>0 for i in range(3)))
        with torch.inference_mode():
            changed=x.detach().clone();changed[:,0]=0
            other,_=model(changed,valid,times)
            self.assertGreater(float((other-heat.detach()).abs().max()),1e-6)


class TemporalDatasetTests(unittest.TestCase):
    def test_neighbor_frames_unknown_labels_and_shared_local_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'frames').mkdir();rows=[]
            for i in range(5):
                im=np.full((144,256,3),20+i*20,np.uint8)
                cv2.imwrite(str(root/'frames'/f'{i}.png'),im)
                rows.append(dict(video_id='v',frame_id=i,timestamp_sec=i*.04,resolution=dict(width=256,height=144),
                                 frame_path=f'frames/{i}.png',label_status='positive' if i==2 else 'unknown',
                                 ball_centers=[[100.,70.]] if i==2 else [],ball_boxes=[]))
            save_jsonl(root/'labels/train.jsonl',rows)
            ds=BallDataset(root,'train',temporal_mode='fixed3',input_size=(128,72))
            x,y,v,index=ds[0]
            self.assertEqual(x['images'].shape,(3,3,72,128))
            self.assertEqual(x['frame_valid'].tolist(),[True]*3)
            self.assertEqual(ds.temporal_index(ds.rows[0])['frame_ids'],[1,2,3])
            self.assertEqual(float(v),1.)
            self.assertLess(float(x['images'][0].mean()),float(x['images'][1].mean()))
            self.assertLess(float(x['images'][1].mean()),float(x['images'][2].mean()))
            pred={('v',i):dict(x=100.,y=70.,valid=True) for i in range(5)}
            local=BallDataset(root,'train','local',predictions=pred,temporal_mode='fixed3',input_size=(64,64))
            lx,ly,lv,_=local[0]
            self.assertEqual(lx['images'].shape,(3,3,64,64))
            # All three crops use the center transform; constant images stay ordered.
            self.assertLess(float(lx['images'][0].mean()),float(lx['images'][2].mean()))
            edge=BallDataset(root,'train',all_frames=True,temporal_mode='fixed3',input_size=(128,72))
            ex,_,_,_=edge[0]
            self.assertEqual(ex['frame_valid'].tolist(),[False,True,True])
            self.assertEqual(float(ex['images'][0].abs().sum()),0.)
