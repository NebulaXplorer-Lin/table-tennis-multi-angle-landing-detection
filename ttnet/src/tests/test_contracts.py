import unittest
import numpy as np
import torch

from ttnet.data_tools.geometry import letterbox,map_point,crop_transform,crop_image,crop_point,window_indices
from ttnet.models.features import FeatureWindowCache
from ttnet.models.localizer import BallLocalizer,LocalizationLoss
from ttnet.evaluation.metrics import aggregate,decode_heatmap,select_threshold
from ttnet.training.dataset import gaussian


class GeometryTests(unittest.TestCase):
    def test_original_coordinate_roundtrip(self):
        for w,h in [(1266,720),(1280,720),(1920,1080)]:
            image,t=letterbox(np.zeros((h,w,3),np.uint8))
            self.assertEqual(image.shape,(288,512,3))
            for point in [(0,0),(w-1,h-1),(w*.6,h*.7)]:
                np.testing.assert_allclose(map_point(*map_point(*point,t),t,inverse=True),point,atol=1e-6)

    def test_boundary_crop_preserves_center_and_pads(self):
        for w,h in [(1266,720),(1280,720),(1920,1080)]:
            t=crop_transform(0,0,w,h)
            crop=crop_image(np.full((h,w,3),255,np.uint8),t)
            self.assertEqual(crop.shape,(256,256,3))
            self.assertEqual(int(crop[0,0,0]),0)
            np.testing.assert_allclose(crop_point(*crop_point(0,0,t),t,inverse=True),(0,0))

    def test_fps_window_slots(self):
        for fps,n,span in [(25,3,.08),(30,3,2/30),(60,5,4/60),(120,9,8/120)]:
            window=window_indices(np.arange(21)/fps,10)
            self.assertEqual(sum(window['frame_valid']),n)
            self.assertEqual(window['frame_ids'][4],10)
            self.assertAlmostEqual(window['actual_span_sec'],span)
            self.assertEqual(len(set(i for i in window['frame_ids'] if i is not None)),n)

    def test_edges_short_video_and_vfr(self):
        w=window_indices([0],0)
        self.assertEqual(sum(w['frame_valid']),1)
        self.assertEqual(w['frame_ids'][4],0)
        first=window_indices([0,.04,.081,.121],0)
        self.assertFalse(any(first['frame_valid'][:4]))
        self.assertAlmostEqual(first['time_offsets_sec'][5],.04)
        with self.assertRaises(ValueError):window_indices([0,.04,.04],1)


class FeatureTests(unittest.TestCase):
    def test_local_mask_and_video_reset(self):
        c=FeatureWindowCache('data','global','local');c.reset('video-a')
        for i in range(3):
            c.append(i,i*.04,np.ones(128),np.ones(128),i!=1,{'frame_id':i})
        w=c.window(1)
        self.assertEqual(w['features'].shape,(1,9,256))
        self.assertFalse(w['local_valid'][0,4])
        self.assertEqual(float(w['features'][0,4,128:].sum()),0.)
        self.assertEqual(float(w['features'][0,0].sum()),0.)
        c.reset('video-b');self.assertEqual(len(c.frames),0)


class ModelTests(unittest.TestCase):
    def test_export_pooling_is_equivalent(self):
        torch.set_num_threads(2)
        model=BallLocalizer().eval()
        with torch.inference_mode():
            for h,w in [(288,512),(256,256)]:
                model.export_pool_kernel=None
                x=torch.randn(2,3,h,w)
                expected=model(x)
                model.export_pool_kernel=(h//4,w//4)
                actual=model(x)
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b)

    def test_shapes_features_gradients_and_ignore_loss(self):
        torch.set_num_threads(2)
        model=BallLocalizer()
        for shape in [(1,3,288,512),(1,3,256,256)]:
            h,f=model(torch.zeros(shape))
            self.assertEqual(h.shape,(1,1,shape[2]//2,shape[3]//2))
            self.assertEqual(f.shape,(1,128))
        criterion=LocalizationLoss()
        heat,features=model(torch.zeros((1,3,256,256)))
        target=torch.from_numpy(gaussian(128,128,(62.2,71.7)))[None]
        self.assertEqual(float(criterion(heat,target,torch.zeros(1))),0.)
        loss=criterion(heat,target,torch.ones(1));loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


class MetricTests(unittest.TestCase):
    def row(self,fid,status='positive'):
        return dict(video_id='v',frame_id=fid,label_status=status,ball_centers=[[10.,10.]],resolution={'width':100,'height':100})

    def test_ignore_breaks_miss_runs_and_location_error_counts_fp_fn(self):
        rows=[self.row(0),self.row(1,'unknown'),self.row(2)]
        predictions=[None,dict(x=30,y=30,confidence=1.,valid=True),dict(x=30,y=30,confidence=1.,valid=True)]
        result=aggregate(rows,predictions,10)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(0,1,2))
        self.assertEqual(result['max_consecutive_missed'],1)

    def test_independent_peaks_are_not_averaged(self):
        heat=gaussian(64,64,(10,10))[0]+.8*gaussian(64,64,(50,50))[0]
        x,y,c=decode_heatmap(heat)
        self.assertAlmostEqual(x,20,places=3);self.assertAlmostEqual(y,20,places=3)

    def test_local_threshold_fallback(self):
        rows=[self.row(0)]
        glob=[dict(x=10,y=10,confidence=.9,source='global')]
        local=[dict(x=70,y=70,confidence=.2,source='local',crop_transform={})]
        selected=select_threshold(rows,local,10,glob,.5)
        self.assertEqual(selected['metrics']['f1'],1.)
        self.assertEqual(selected['predictions'][0]['source'],'global')


if __name__=='__main__':unittest.main()
