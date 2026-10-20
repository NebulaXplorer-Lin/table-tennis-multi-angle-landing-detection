import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ttnet.events.engine import EventSystem
from ttnet.events.joint import heat_target, joint_forward, teacher_probability
from ttnet.events.data import assemble_window
from ttnet.events.train import state_digest
from ttnet.models.event import event_loss
from ttnet.models.localizer import LocalizationLoss
from ttnet.data_tools.geometry import crop_transform


class Corpus:
    def __init__(self):
        self.videos={'pp_v':dict(meta=dict(n_frames=13,width=96,height=64),frames={
            i:dict(timestamp_sec=i*.04,label_status='unknown' if i==2 else 'positive',xy=None if i==2 else [48.,32.])
            for i in range(13)})}
        self.images={i:np.random.default_rng(i).integers(0,256,(64,96,3),dtype=np.uint8) for i in range(13)}
    def image(self,vid,fid):return self.images[fid]
    def window_ids(self,vid,center):return [i if 0<=i<13 else None for i in range(center-6,center+7)]


def config():
    return dict(initialization='random',table_head=True,train_table_head=True,
                global_input=[64,32],local_input=[32,32],local_crop_side_1080=256,
                tracker_type='association',context_head=True,scene_mode='box',head_dropout=0.,
                localization_threshold=0.,sigma=1.5,amp=False,joint_frame_batch=2,
                gt_crop_full_epochs=3,gt_crop_end_epoch=10,table_smoothing='window_ema',table_smoothing_tau_sec=.15)


class ScratchTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)

    def test_random_never_loads_and_all_modules_update_at_first_step(self):
        cfg=config()
        with patch('torch.load',side_effect=AssertionError('pretrained weights read')):
            system=EventSystem(cfg,torch.device('cpu'))
        system.configure('joint',True)
        modules=[system.global_model,system.local_model,system.table_head,system.head]
        before=[state_digest(m) for m in modules]
        self.assertTrue(all(p.requires_grad for m in modules for p in m.parameters()))
        sample=dict(video_id='pp_v',frame_id=6)
        with patch('ttnet.events.joint.annotations',return_value=({('v',i):[10,20,85,60] for i in range(13)},{})):
            logits,losses=joint_forward(system,Corpus(),sample)
        event=event_loss(logits,torch.tensor([[1.,0.]]),torch.tensor([[True,False]]))
        for m in modules[:2]:
            grad=torch.autograd.grad(event,next(m.encoder.parameters()),retain_graph=True)[0]
            self.assertGreater(float(grad.abs().sum()),0.)
        self.assertGreater(float(losses['table']),0.)
        opt=torch.optim.AdamW(system.parameters(),lr=3e-4)
        (event+sum(losses.values())).backward();opt.step()
        self.assertTrue(all(a!=state_digest(m) for a,m in zip(before,modules)))
        cfg['global_checkpoint']='forbidden.pt'
        with self.assertRaisesRegex(ValueError,'random initialization'):EventSystem(cfg,torch.device('cpu'))

    def test_unknown_localization_mask_and_empty_crop(self):
        crop=crop_transform(48,32,96,64,32,256)
        target,known=heat_target(dict(label_status='unknown',xy=[48,32]),crop,32,32,1.5,True)
        self.assertFalse(known);self.assertEqual(target.sum(),0)
        target,known=heat_target(dict(label_status='positive',xy=[0,0]),crop,32,32,1.5,True)
        self.assertTrue(known);self.assertEqual(target.sum(),0)
        p=torch.full((2,1,16,16),.2,requires_grad=True)
        LocalizationLoss()(p,torch.zeros_like(p),torch.tensor([False,True])).backward()
        self.assertEqual(float(p.grad[0].abs().sum()),0.)
        self.assertGreater(float(p.grad[1].abs().sum()),0.)

    def test_teacher_ends_and_prediction_only_matches_stream_inference(self):
        cfg=config();self.assertEqual(teacher_probability(cfg,0),1.)
        self.assertEqual(teacher_probability(cfg,9),0.)
        self.assertEqual(teacher_probability(cfg,20),0.)
        corpus=Corpus();sample=dict(video_id='pp_v',frame_id=6)
        system=EventSystem(cfg,torch.device('cpu'));system.configure('joint',True);system.training_epoch=9
        with patch('ttnet.events.joint.annotations',return_value=({},{})):
            online,_=joint_forward(system,corpus,sample)
        # Evaluation is full chronology, and the training-only entry is forbidden.
        system.configure('joint',False)
        with self.assertRaisesRegex(ValueError,'training-only'):joint_forward(system,corpus,sample)
        with torch.no_grad():
            features,records,_=system.forward_frames(corpus,'pp_v',range(13))
            streamed=system.head(*assemble_window(corpus,'pp_v',6,features,records,system.device,cfg))
        torch.testing.assert_close(online,streamed,atol=2e-4,rtol=2e-4)
        # Label changes cannot alter inference, even in joint checkpoint mode.
        changed=copy.deepcopy(corpus)
        for row in changed.videos['pp_v']['frames'].values():row.update(label_status='unknown',xy=[0,0])
        system.reset_tracking('pp_v')
        with torch.no_grad():
            f,r,_=system.forward_frames(changed,'pp_v',range(13))
            other=system.head(*assemble_window(changed,'pp_v',6,f,r,system.device,cfg))
        torch.testing.assert_close(streamed,other,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
