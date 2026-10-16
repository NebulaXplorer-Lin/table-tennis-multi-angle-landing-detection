import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace

import numpy as np
import torch
import av

from ttnet.common import sha
from ttnet.events.data import assemble_window, balanced_samples
from ttnet.events.engine import EventSystem
from ttnet.events.metrics import landing_match, peaks
from ttnet.events.prepare import target_samples, nearest_index
from ttnet.models.event import EventHead, event_loss
from ttnet.models.localizer import BallLocalizer
from ttnet.events.inference import predict_video,source_mapping
from ttnet.events.evaluate import sparse_evaluation
from ttnet.events.train import train,atomic_checkpoint
from ttnet.events.epoch_public import capture


class EventTargetTests(unittest.TestCase):
    def test_nominal_25fps_preserves_last_frame_and_genuine_upsampling_is_rejected(self):
        count,mapping=source_mapping(178,Fraction(3750250,150001),25)
        self.assertEqual(count,178);self.assertEqual(mapping[177],177)
        self.assertEqual(source_mapping(72,120,25)[0],15)
        with self.assertRaises(ValueError):source_mapping(100,20,25)

    def test_dropped_event_preserves_continuous_time(self):
        events=[dict(frame_id=1,timestamp_sec=.025,kind='bounce')]
        samples=target_samples(events,{i:i*.04 for i in range(5)})
        by_id={s['frame_id']:s for s in samples}
        self.assertAlmostEqual(by_id[0]['target'][0],.375)
        self.assertAlmostEqual(by_id[1]['target'][0],.625)
        self.assertEqual(by_id[1]['hard_target'],[1.,0.])
        self.assertFalse(by_id[0]['supervised'][1])
        self.assertEqual(by_id[2]['target'][0],0.)

    def test_collision_keeps_positive_and_empty_is_not_expanded(self):
        events=[dict(frame_id=4,timestamp_sec=.16,kind='bounce'),
                dict(frame_id=4,timestamp_sec=.17,kind='empty_event'),
                dict(frame_id=12,timestamp_sec=.48,kind='empty_event')]
        samples=target_samples(events)
        anchor=next(s for s in samples if s['anchor']==4 and s['original'])
        self.assertEqual(anchor['hard_target'],[1.,0.])
        self.assertEqual(len([s for s in samples if s['anchor']==12]),1)
        self.assertEqual(nearest_index(2.5),3)

    def test_balancing_is_reproducible_and_anchors_do_not_multiply(self):
        events=[dict(frame_id=f,timestamp_sec=f*.04,kind=k) for f,k in
                [(4,'bounce'),(12,'bounce'),(20,'net'),(28,'empty_event')]]
        samples=[dict(s,video_id='a') for s in target_samples(events)]
        a=balanced_samples(samples,42,1)
        self.assertEqual(a,balanced_samples(samples,42,1))
        self.assertEqual([sum(s['group']==k for s in a) for k in ['bounce','net','empty_event']],[2,2,2])


class EventHeadTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(42)

    def test_nan_padding_and_invalid_local_cannot_affect_output(self):
        model=EventHead().eval();f=torch.randn(2,13,256);m=torch.randn(2,13,9)
        valid=torch.ones(2,13,dtype=torch.bool);valid[:,:3]=False;m[:,:,7]=0
        with torch.no_grad():a=model(f,m,valid)
        f[~valid]=float('nan');m[~valid]=float('nan');f[:,:,128:]=float('nan')
        with torch.no_grad():b=model(f,m,valid)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_masked_loss_does_not_train_unknown_class(self):
        logits=torch.zeros(1,2,requires_grad=True)
        event_loss(logits,torch.ones(1,2),torch.tensor([[True,False]])).backward()
        self.assertLess(float(logits.grad[0,0]),0)
        self.assertEqual(float(logits.grad[0,1]),0)


class TinyCorpus:
    def __init__(self):
        self.videos={'v':dict(meta=dict(n_frames=17,width=96,height=64),
                    frames={i:dict(frame_id=i,timestamp_sec=i*.04,label_status='positive' if i%2==0 else 'unknown',xy=[48.,32.] if i%2==0 else None)
                            for i in range(17)})}
        self.images=[np.random.default_rng(i).integers(0,256,(64,96,3),dtype=np.uint8) for i in range(17)]
    def image(self,vid,fid):return self.images[fid]
    def window_ids(self,vid,center):return [i if 0<=i<17 else None for i in range(center-6,center+7)]


class EventGradientTests(unittest.TestCase):
    def test_frozen_and_joint_feature_paths_and_checkpoint_reload(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g=root/'g.pt';l=root/'l.pt'
            model=BallLocalizer('fixed3')
            torch.save(dict(model=model.state_dict(),config=dict(temporal_mode='fixed3')),g)
            torch.save(dict(model=model.state_dict(),config=dict(temporal_mode='fixed3'),global_checkpoint_sha256=sha(g)),l)
            cfg=dict(global_checkpoint=str(g),local_checkpoint=str(l),global_input=[64,32],local_input=[32,32],
                     local_crop_side_1080=256,localization_threshold=0.,sigma=1.5,amp=False)
            system=EventSystem(cfg,torch.device('cpu'));corpus=TinyCorpus()
            for mode in ['frozen','joint']:
                system.zero_grad(set_to_none=True);system.configure(mode,True)
                d,r,losses=system.forward_frames(corpus,'v',range(1,15))
                logits=system.head(*assemble_window(corpus,'v',8,d,r,system.device))
                e=event_loss(logits,torch.tensor([[1.,0.]]),torch.ones(1,2,dtype=torch.bool))
                # Event loss alone must reach both branches, independent of localization loss.
                e.backward()
                self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
                for locator in [system.global_model,system.local_model]:
                    grad=next(locator.encoder.parameters()).grad
                    if mode=='frozen':self.assertIsNone(grad)
                    else:self.assertGreater(float(grad.abs().sum()),0.)
            system.configure('joint',False)
            portable_cfg=dict(cfg,global_checkpoint='missing/pretrain_g.pt',local_checkpoint='missing/pretrain_l.pt')
            system2=EventSystem(portable_cfg,torch.device('cpu'),initialize=False)
            system2.load_state_dict(copy.deepcopy(system.state_dict()));system2.configure('joint',False)
            with torch.no_grad():
                d,r,_=system.forward_frames(corpus,'v',range(1,15))
                a=system.head(*assemble_window(corpus,'v',8,d,r,system.device))
                d,r,_=system2.forward_frames(corpus,'v',range(1,15))
                b=system2.head(*assemble_window(corpus,'v',8,d,r,system.device))
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_full_video_inference_decodes_120_to_25_and_flushes_tail(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g=root/'g.pt';l=root/'l.pt';video=root/'tiny.mp4'
            model=BallLocalizer('fixed3')
            torch.save(dict(model=model.state_dict(),config=dict(temporal_mode='fixed3')),g)
            torch.save(dict(model=model.state_dict(),config=dict(temporal_mode='fixed3'),global_checkpoint_sha256=sha(g)),l)
            with av.open(str(video),'w') as out:
                stream=out.add_stream('libx264',rate=120);stream.width=96;stream.height=64;stream.pix_fmt='yuv420p'
                for i in range(72):
                    frame=av.VideoFrame.from_ndarray(np.full((64,96,3),i*3,np.uint8),format='rgb24')
                    for packet in stream.encode(frame):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
            cfg=dict(global_checkpoint=str(g),local_checkpoint=str(l),global_input=[64,32],local_input=[32,32],
                     local_crop_side_1080=256,localization_threshold=0.,sigma=1.5,amp=False)
            system=EventSystem(cfg,torch.device('cpu'));system.configure('frozen',False)
            rows,records,speed=predict_video(system,video,'v',progress=False)
            self.assertEqual(len(rows),15)
            self.assertEqual(rows[-1]['frame_id'],14)
            self.assertTrue(all(np.isfinite(r['probabilities']).all() for r in rows))
            self.assertEqual(len(records),15)
            self.assertGreater(speed['e2e_fps'],0)

    def test_sparse_evaluation_keeps_hard_and_soft_targets_separate(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g=root/'g.pt';l=root/'l.pt'
            model=BallLocalizer('fixed3')
            torch.save(dict(model=model.state_dict(),config=dict(temporal_mode='fixed3')),g)
            torch.save(dict(model=model.state_dict(),config=dict(temporal_mode='fixed3'),global_checkpoint_sha256=sha(g)),l)
            cfg=dict(global_checkpoint=str(g),local_checkpoint=str(l),global_input=[64,32],local_input=[32,32],
                     local_crop_side_1080=256,localization_threshold=0.,sigma=1.5,amp=False)
            system=EventSystem(cfg,torch.device('cpu'));system.configure('frozen',False)
            corpus=TinyCorpus();corpus.split={'val':['v']}
            events=[dict(frame_id=8,timestamp_sec=.30,kind='bounce'),dict(frame_id=12,timestamp_sec=.48,kind='net')]
            corpus.videos['v']['events']=events
            corpus.samples={'val':[dict(s,video_id='v') for s in target_samples(events)]}
            features={i:torch.randn(256) for i in range(17)}
            records={i:dict(frame_id=i,prediction=dict(x=48.,y=32.,confidence=.9,valid=True),local_valid=True) for i in range(17)}
            metrics=sparse_evaluation(system,corpus,'val',root/'eval',[.5,.5],{'v':(features,records)})
            self.assertEqual(metrics['samples'],2)
            self.assertEqual(metrics['localization']['10']['f1'],1.)
            self.assertTrue((root/'eval/errors.html').exists())
            anchors=[json.loads(s) for s in (root/'eval/anchors.jsonl').read_text().splitlines()]
            self.assertEqual(anchors[0]['target'],[1.,0.])
            self.assertLess(anchors[0]['soft_target'][0],1.)


class EventMatchingTests(unittest.TestCase):
    def test_maximum_matching_and_ignored_gt(self):
        p=[dict(frame_id=1,x=0,y=0),dict(frame_id=0,x=0,y=0),dict(frame_id=8,x=0,y=0)]
        gt=[dict(frame_id=0,x=0,y=0),dict(frame_id=2,x=0,y=0),dict(frame_id=8,x=0,y=0,valid=False)]
        result=landing_match(p,gt)
        self.assertEqual((result['tp'],result['fp'],result['fn'],result['ignored']),(2,0,0,1))

    def test_peaks_and_chunk_equivalence_requires_full_context(self):
        rows=[dict(frame_id=i,timestamp_sec=i*.04,probabilities=[p,0]) for i,p in enumerate([.1,.8,.8,.4,.1,.9,.1])]
        selected=peaks(rows,[.5,.5])
        self.assertEqual([r['frame_id'] for r in selected],[1,5])


class CheckpointResumeTests(unittest.TestCase):
    def test_public_snapshot_uses_exact_epoch_not_next_epoch_partial_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'checkpoints').mkdir();dest=root/'eval';dest.mkdir()
            torch.save(dict(epoch=2,step=200,system={'weight':torch.tensor([3.])}),root/'checkpoints/last.pt')
            correct=dict(epoch=2,step=0,thresholds=[.1,.3],config={'dataset_id':'fixture'},system={'weight':torch.tensor([2.])})
            torch.save(correct,root/'checkpoints/best.pt')
            state=capture(root,dest,2,1)
            torch.testing.assert_close(state['system']['weight'],torch.tensor([2.]))
            self.assertEqual(sha(dest/'checkpoint.pt'),sha(root/'checkpoints/best.pt'))

    def test_mid_epoch_resume_matches_uninterrupted_training(self):
        torch.set_num_threads(2)
        class Corpus:
            def __init__(self,root):
                self.identity='fixture';self.metadata={'complete':True,'pilot':False}
                self.split={'split_id':'fixture','train':['a'],'val':['b'],'test':['c']}
                self.videos={v:{'meta':{'fps':25,'source_identity':{'soft_radius_sec':.04}}} for v in 'abc'}
                self.samples={'val':[],'train':[]}
                for group,target in [('bounce',[1.,0.]),('net',[0.,1.]),('empty_event',[0.,0.])]:
                    for i in range(100):
                        self.samples['train'].append(dict(video_id='a',anchor=i+100*len(self.samples['train']),frame_id=i,
                            group=group,target=target,supervised=[True,True]))
        class System(torch.nn.Module):
            def __init__(self,cfg,device):
                super().__init__();self.config=cfg;self.mode='frozen'
                self.global_model=torch.nn.Linear(2,2);self.local_model=torch.nn.Linear(2,2)
                self.head=torch.nn.Sequential(torch.nn.Linear(9,8),torch.nn.Dropout(.1),torch.nn.Linear(8,2))
            def configure(self,mode,training):
                self.mode=mode;self.train(training)
                for model in [self.global_model,self.local_model]:model.requires_grad_(False);model.eval()
        def forward(system,corpus,cache,sample):return system.head(torch.full((1,9),sample['frame_id']/100))
        anchors=[dict(target=t,probabilities=p) for t,p in [([1.,0.],[.9,.1]),([0.,1.],[.1,.9]),([0.,0.],[.1,.1])]]
        real_atomic=atomic_checkpoint
        class Interrupted(Exception):pass
        def interrupt(path,state):
            real_atomic(path,state)
            if state.get('epoch')==0 and state.get('step')==200:raise Interrupted()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=json.loads(Path('ttnet/src/configs/event.json').read_text())
            for name in ['g','l']:(root/name).write_bytes(name.encode())
            config.update(global_checkpoint=str(root/'g'),local_checkpoint=str(root/'l'),epochs=1,amp=False,torch_threads=2)
            cfg=root/'config.json';cfg.write_text(json.dumps(config))
            def args(run,resume=False):return Namespace(config=cfg,dataset=root/'data',cache=root/'cache',run=root/run,
                device='cpu',mode='frozen',smoke=False,resume=resume,restart_incomplete=False)
            with patch('ttnet.events.train.EventCorpus',Corpus),patch('ttnet.events.train.EventSystem',System),\
                 patch('ttnet.events.train.build_cache'),patch('ttnet.events.train.load_cache',return_value={}),\
                 patch('ttnet.events.train.cached_forward',side_effect=forward),\
                 patch('ttnet.events.train.validate',return_value=(.5,anchors)):
                train(args('full'))
                with patch('ttnet.events.train.atomic_checkpoint',side_effect=interrupt):
                    with self.assertRaises(Interrupted):train(args('resume'))
                state=torch.load(root/'resume/checkpoints/last.pt',weights_only=False)
                self.assertEqual(state['step'],200)
                train(args('resume',True))
            full=torch.load(root/'full/checkpoints/last.pt',weights_only=False)
            resumed=torch.load(root/'resume/checkpoints/last.pt',weights_only=False)
            for key,value in full['system'].items():torch.testing.assert_close(value,resumed['system'][key],rtol=0,atol=0)
            self.assertEqual(resumed['epoch'],1);self.assertEqual(resumed['step'],0)


if __name__=='__main__':unittest.main()
