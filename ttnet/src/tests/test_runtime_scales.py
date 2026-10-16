import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ttnet.common import DATA,jsonl,save_jsonl
from ttnet.data_tools.geometry import map_point,crop_point
from ttnet.training.dataset import BallDataset,normalize,read_rgb
from ttnet.training.scales import MultiScaleBatchSampler,input_size,training_scales
from ttnet.evaluation.inference import predict


class RuntimeScaleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'frames').mkdir()
        self.rows=[]
        for i,(w,h) in enumerate([(1266,720),(1280,720),(1920,1080)]):
            rgb=np.zeros((h,w,3),np.uint8);x,y=w*.55,h*.45
            cv2.circle(rgb,(round(x),round(y)),5,(120,200,250),-1)
            path=self.root/'frames'/f'{i}.png';cv2.imwrite(str(path),rgb)
            self.rows.append(dict(video_id='fixture',frame_id=i,timestamp_sec=i*.04,fps=25.,
                                 resolution=dict(width=w,height=h),ball_centers=[[x,y]],ball_boxes=[[x-3,y-3,x+3,y+3]],
                                 label_status='positive',frame_path=str(path.relative_to(self.root))))
        save_jsonl(self.root/'labels/train.jsonl',self.rows)

    def tearDown(self):self.temp.cleanup()

    def test_source_only_labels_work_at_three_scales_without_mutation(self):
        ds=BallDataset(self.root,'train');original=copy.deepcopy(ds.rows)
        for size in [(384,216),(512,288),(640,360)]:
            for i,row in enumerate(self.rows):
                x,y,valid,index=ds[(i,size)]
                self.assertEqual(tuple(x.shape),(3,size[1],size[0]))
                self.assertEqual(tuple(y.shape),(1,size[1]//2,size[0]//2))
                self.assertEqual(float(valid),1.)
                expected=map_point(*row['ball_centers'][0],ds.transform_for_row(row,size))
                iy,ix=np.unravel_index(int(y.argmax()),y.shape[-2:])
                np.testing.assert_allclose([ix*2,iy*2],expected,atol=1.01)
        self.assertEqual(ds.rows,original)

    def test_local_crop_field_is_independent_of_output_scale(self):
        predictions={(r['video_id'],r['frame_id']):dict(x=r['ball_centers'][0][0],y=r['ball_centers'][0][1],valid=True) for r in self.rows}
        ds=BallDataset(self.root,'train','local',predictions=predictions)
        for i,row in enumerate(self.rows):
            sides=set()
            for size in [(192,192),(256,256),(320,320)]:
                x,y,valid,index=ds[(i,size)]
                self.assertEqual(tuple(x.shape),(3,size[1],size[0]))
                t=ds.transform_for_row(row,size);sides.add(t['side'])
                p=crop_point(*row['ball_centers'][0],t)
                np.testing.assert_allclose(crop_point(*p,t,inverse=True),row['ball_centers'][0],atol=1e-6)
            self.assertEqual(len(sides),1)

    def test_multiscale_batches_are_collatable(self):
        ds=BallDataset(self.root,'train')
        torch.manual_seed(0)
        sampler=MultiScaleBatchSampler(len(ds),2,[(384,216),(640,360)])
        batches=list(DataLoader(ds,batch_sampler=sampler,num_workers=0))
        ids=[]
        for x,y,valid,indices in batches:
            self.assertIn(tuple(x.shape[-2:]),[(216,384),(360,640)])
            self.assertEqual(tuple(y.shape[-2:]),tuple(s//2 for s in x.shape[-2:]))
            ids.extend(indices.tolist())
        self.assertEqual(sorted(ids),list(range(len(ds))))

    def test_shape_validation(self):
        self.assertEqual(input_size({},'global'),(512,288))
        self.assertEqual(training_scales({'global_train_scales':[[384,216],[640,360]]},'global'),[(384,216),(640,360)])
        with self.assertRaises(ValueError):input_size({'local_input':[256,192]},'local')
        with self.assertRaises(ValueError):input_size({'global_input':[511,288]},'global')

    def test_prediction_uses_runtime_inverse_transform(self):
        save_jsonl(self.root/'labels/val.jsonl',self.rows)
        class FixedPeak(torch.nn.Module):
            def forward(self,x):
                b,_,h,w=x.shape
                heat=torch.zeros((b,1,h//2,w//2))
                heat[:,:,h//4,w//4]=1.
                return heat,torch.zeros((b,128))
        for size in [(384,216),(640,360)]:
            rows,candidates,_=predict(FixedPeak(),self.root,'val',torch.device('cpu'),workers=0,batch=2,input_size=size)
            ds=BallDataset(self.root,'val',input_size=size)
            for r,p in zip(rows,candidates):
                expected=map_point(size[0]/2,size[1]/2,ds.transform_for_row(r),inverse=True)
                np.testing.assert_allclose([p['x'],p['y']],expected,atol=1e-6)


class MaterializedDataTests(unittest.TestCase):
    def test_v2_contains_no_model_input_fields_and_shares_original_frames(self):
        if not (DATA/'metadata/dataset.json').exists():self.skipTest('materialized dataset not present')
        meta=json.loads((DATA/'metadata/dataset.json').read_text())
        self.assertEqual(meta['schema'],'ttnet_ball_v2.0')
        self.assertEqual(meta['status'],'completed')
        if (DATA/'frame_store.json').exists():
            from ttnet.data_tools.frame_store import FrameStore
            self.assertIsNotNone(FrameStore(DATA).root)
        else:self.assertTrue((DATA/'frames').is_symlink())
        row=jsonl(DATA/'labels/val.jsonl')[0]
        self.assertFalse({'global_resolution','global_transform','global_input_path'} & row.keys())
        self.assertTrue({'fps','resolution','timestamp_sec','frame_path'} <= row.keys())
        self.assertFalse((DATA/'global_inputs').exists())

    def test_legacy_global_images_equal_runtime_resize(self):
        from ttnet.common import ROOT
        legacy=ROOT/'data/datasets/ball_v1'
        if not legacy.exists():self.skipTest('legacy dataset absent')
        if not next((legacy/'global_inputs').glob('*/*.png'),None):
            self.skipTest('obsolete PNG resize cache was removed; runtime resize remains covered')
        seen=set()
        for split in ['train','val']:
            ds=BallDataset(legacy,split,all_frames=True)
            for i,row in enumerate(ds.rows):
                key=tuple(row['resolution'].values())
                if key in seen:continue
                seen.add(key)
                image,_,_,_=ds[i]
                expected=normalize(read_rgb(legacy/row['global_input_path']))
                np.testing.assert_array_equal(image.numpy(),expected)
        self.assertEqual(len(seen),3)


if __name__=='__main__':unittest.main()
