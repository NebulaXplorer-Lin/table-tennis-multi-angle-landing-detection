import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ttnet.configuration import load_config
from ttnet.models.build import build_event_head
from ttnet.models.event import event_loss
from ttnet.tracking.association import BeamTracker
from ttnet.tracking.candidates import candidates
from ttnet.events.engine import EventSystem, cache_identity
from ttnet.events.data import assemble_window
from ttnet.events.continuous import mixed_samples, score_continuous, select_continuous_threshold, known
from ttnet.events.metrics import classification
from ttnet.evaluation.candidate_audit import classify_error
from ttnet.tests.test_events import TinyCorpus
from ttnet.data_tools.geometry import letterbox_transform


class ComparisonTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_presets_share_training_protocol_and_feature_pairs(self):
        configs={k:load_config(f'configs/experiments/event_comparison_{k}.json')[0] for k in 'ABCD'}
        for cfg in configs.values():
            self.assertEqual((cfg['seed'],cfg['window'],cfg['lr'],cfg['effective_batch'],cfg['epochs'],cfg['patience']), (42,13,.0003,8,30,5))
            self.assertEqual(cfg['best_metric'],'pp_landing_macro_f1')
        class Corpus:identity='data'
        class System:
            def __init__(self,c):self.config=c
        with patch('ttnet.events.engine.sha',return_value='hash'):
            ids={k:cache_identity(System(c),Corpus()) for k,c in configs.items()}
        self.assertEqual(ids['A'],ids['C']);self.assertEqual(ids['B'],ids['D']);self.assertNotEqual(ids['A'],ids['B'])
        cfg=copy.deepcopy(configs['B']);cfg['association']['beam_size']=4
        with patch('ttnet.events.engine.sha',return_value='hash'):self.assertNotEqual(ids['B'],cache_identity(System(cfg),Corpus()))

    def test_transformer_mask_time_backward_and_reload(self):
        torch.manual_seed(42)
        cfg={'event_head_type':'transformer'};model=build_event_head(cfg).eval()
        f=torch.randn(3,13,256);m=torch.randn(3,13,9);m[:,:,7]=0
        v=torch.ones(3,13,dtype=torch.bool);v[0,:3]=False;v[1,:]=False
        original=model(f,m,v)
        f[~v]=float('nan');m[~v]=float('nan');f[:,:,128:]=float('nan')
        actual=model(f,m,v);torch.testing.assert_close(original,actual,atol=0,rtol=0)
        self.assertTrue(torch.isfinite(actual).all())
        shifted=m.clone();shifted[2,:,8]=torch.arange(13)*.1
        self.assertFalse(torch.equal(actual[2],model(f,shifted,v)[2]))
        target=torch.ones(3,2);target[:,1]=float('nan')
        event_loss(actual,target,torch.tensor([[1,0]]*3)).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        restored=build_event_head(cfg).eval();restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(actual,restored(f,m,v),atol=0,rtol=0)

    def test_causal_tracker_time_missing_turn_and_reacquire(self):
        tracker=BeamTracker()
        point=lambda x,y=0:dict(x=x,y=y,confidence=.95)
        tracker.update([point(0)],0,0.)
        tracker.update([point(10)],1,.1)
        self.assertAlmostEqual(tracker.predict(.3)['x'],30.)
        tracker.update([point(10,10)],2,.2) # a bounce/turn must not be rejected
        self.assertIsNotNone(tracker.predict(.3))
        for f in range(3,7):self.assertIsNone(tracker.update([],f,f*.1))
        self.assertIsNone(tracker.predict(.7))
        self.assertEqual(tracker.update([point(1000)],7,.7)['x'],1000)
        with self.assertRaises(ValueError):tracker.update([],7,.7)

    def test_distractor_and_short_missing_remain_observations(self):
        tracker=BeamTracker(dict(motion_weight=1.))
        for fid in range(4):
            p=tracker.update([dict(x=fid*5.,y=20.,confidence=.8)],fid,fid*.04)
        selected=tracker.update([dict(x=20.,y=20.,confidence=.8),dict(x=800.,y=700.,confidence=.9)],4,.16)
        self.assertEqual(selected['x'],20.)
        self.assertIsNone(tracker.update([],5,.20))
        self.assertFalse(tracker.predict(.24)['valid'])
        self.assertTrue(tracker.predict(.24)['predicted'])
        self.assertEqual(tracker.update([dict(x=30.,y=20.,confidence=.8)],6,.24)['x'],30.)

    def test_runtime_state_is_continuous_across_chunks(self):
        cfg=dict(tracker_type='association',temporal_mode='fixed3',global_input=[64,32],local_input=[32,32],
                 local_crop_side_1080=256,localization_threshold=0.,sigma=1.5,amp=False,association={'min_confidence':.01})
        torch.manual_seed(42);system=EventSystem(cfg,torch.device('cpu'),initialize=False);system.configure('frozen',False)
        corpus=TinyCorpus()
        with torch.inference_mode():
            a,ar,_=system.forward_frames(corpus,'v',range(17))
            system.reset_tracking('v');b={};br={}
            for ids in (range(4),range(4,11),range(11,17)):
                d,r,_=system.forward_frames(corpus,'v',ids);b.update(d);br.update(r)
        self.assertEqual(ar,br)
        for fid in a:
            torch.testing.assert_close(a[fid],b[fid],atol=0,rtol=0)
            self.assertLessEqual(len(ar[fid]['roi_transforms']),2)
        with self.assertRaises(ValueError):system.forward_frames(corpus,'v',[3])
        # Independent A remains callable in arbitrary training-window order.
        cfg['tracker_type']='independent';baseline=EventSystem(cfg,torch.device('cpu'),initialize=False)
        baseline.configure('frozen',False)
        with torch.inference_mode():baseline.forward_frames(corpus,'v',[7,8]);baseline.forward_frames(corpus,'v',[3])

    def test_declared_missing_frames_are_masked_without_resetting_video(self):
        corpus=TinyCorpus();corpus.videos['v']['frames'][4]['frame_valid']=False
        original=corpus.image
        def image(vid,fid):
            if fid==4:raise AssertionError('missing frame was decoded')
            return original(vid,fid)
        corpus.image=image
        for tracker in ('independent','association'):
            cfg=dict(tracker_type=tracker,temporal_mode='fixed3',global_input=[64,32],local_input=[32,32],
                     local_crop_side_1080=256,localization_threshold=0.,sigma=1.5,amp=False)
            system=EventSystem(cfg,torch.device('cpu'),initialize=False);system.configure('frozen',False)
            with torch.no_grad():features,records,_=system.forward_frames(corpus,'v',range(17))
            self.assertIsNone(records[4]['prediction'])
            f,m,v=assemble_window(corpus,'v',6,features,records,system.device)
            self.assertFalse(v[0,4]);self.assertEqual(float(f[0,4].abs().sum()),0.)
            if tracker=='association':self.assertEqual(records[4]['tracking_state']['missing'],1)

    def test_batched_frozen_features_match_independent_decode(self):
        from ttnet.events.batched_features import independent_features
        torch.manual_seed(42);corpus=TinyCorpus();corpus.videos['v']['frames'][4]['frame_valid']=False
        cfg=dict(temporal_mode='fixed3',global_input=[64,32],local_input=[32,32],local_crop_side_1080=256,localization_threshold=0.,sigma=1.5,amp=False)
        system=EventSystem(cfg,torch.device('cpu'),initialize=False);system.configure('frozen',False)
        with torch.inference_mode():
            expected,records,_=system.forward_frames(corpus,'v',range(17))
            for size in (2,8):
                for start in range(0,17,size):
                    d,r,_=independent_features(system,corpus,'v',range(start,min(start+size,17)))
                    for fid in d:
                        torch.testing.assert_close(d[fid],expected[fid],atol=1e-5,rtol=1e-4)
                        if r[fid]['prediction'] is not None:
                            for key in ('x','y','confidence'):self.assertAlmostEqual(r[fid]['prediction'][key],records[fid]['prediction'][key],places=4)
                        else:self.assertIsNone(records[fid]['prediction'])

    def test_microbatch_keeps_equal_per_sample_weight_for_partial_labels(self):
        torch.manual_seed(42);logits=torch.randn(8,2,requires_grad=True);target=torch.rand(8,2)
        mask=torch.tensor([[True,False],[True,True]]*4)
        old=torch.stack([event_loss(logits[i:i+1],target[i:i+1],mask[i:i+1]) for i in range(8)]).mean()
        expected=torch.autograd.grad(old,logits,retain_graph=True)[0]
        new=event_loss(logits,target,mask,per_sample=True).mean()
        actual=torch.autograd.grad(new,logits)[0]
        torch.testing.assert_close(old,new,atol=0,rtol=0);torch.testing.assert_close(expected,actual,atol=0,rtol=0)

    def test_candidates_and_diagnostic_categories(self):
        heat=np.zeros((16,32),np.float32);heat[4,4]=.9;heat[10,20]=.8
        c=candidates(heat,letterbox_transform(64,32,64,32),dict(width=64,height=32))
        self.assertEqual((c[0]['x'],c[0]['y']),(8.,8.))
        row=dict(label_status='positive',ball_centers=[[40.,20.]])
        record=dict(prediction=dict(x=8.,y=8.,valid=True),global_candidates=c,roi_transforms=[])
        self.assertEqual(classify_error(row,record),['ranking_or_selection','search_region_wrong'])
        record['roi_transforms']=[dict(left=32,top=12,side=16)]
        self.assertEqual(classify_error(row,record),['ranking_or_selection','local_error_in_correct_roi'])

    def test_continuous_matching_unknown_and_threshold_tie(self):
        ann={'v':dict(view='side',reviewed_intervals=[[0,8]],unknown_intervals=[[6,8]],events=[dict(kind='bounce',frame_id=2,x=20.,y=30.)])}
        rows={'v':[dict(frame_id=i,timestamp_sec=i*.04,probabilities=[.8 if i in (2,7) else .01,0.],
                       prediction=dict(x=20.,y=30.,valid=True)) for i in range(9)]}
        result=score_continuous(rows,ann,.5)
        self.assertEqual((result['landing']['tp'],result['landing']['fp']),(1,0))
        self.assertEqual(result['macro_f1'],1.)
        rows['v'][2]['prediction']['x']=100.
        result=score_continuous(rows,ann,.5)
        self.assertEqual(result['continuous_bounce']['f1'],1.)
        self.assertEqual(result['landing']['f1'],0.)
        chosen,_=select_continuous_threshold(rows,ann,[.5,.9])
        self.assertEqual(chosen['threshold'],.9) # tied F1, fewer false positives
        self.assertFalse(known(ann['v'],7))

    def test_missing_spatial_gt_keeps_time_supervision_without_fake_locations(self):
        ann={'v':dict(view='side',reviewed_intervals=[[0,9]],unknown_intervals=[],events=[
            dict(kind='bounce',frame_id=2,x=20.,y=30.),
            dict(kind='bounce',frame_id=7,x=None,y=None,spatial_valid=False,coordinate_source='missing_ball_position_gt')])}
        rows={'v':[dict(frame_id=i,timestamp_sec=i*.04,probabilities=[.9 if i in (2,7) else .01,0.],
                       prediction=dict(x=20.,y=30.,valid=True)) for i in range(10)]}
        result=score_continuous(rows,ann,.5)
        self.assertEqual(result['continuous_bounce']['tp'],2)
        self.assertEqual((result['landing']['tp'],result['landing']['fp']),(1,0))
        self.assertEqual(result['per_video']['v']['spatial_gt_missing'],1)

    def test_masked_metrics_and_half_source_sampling(self):
        rows=[dict(target=[1.,0.],probabilities=[.9,.99],supervised=[True,False])]
        self.assertEqual(classification(rows,[.5,.5])['net']['fp'],0)
        samples=[]
        for k in ('bounce','net','empty_event'):
            samples.append(dict(video_id='g',anchor=1,frame_id=1,group=k,target=[0,0],supervised=[True,True]))
        samples += [dict(video_id='pp_v',anchor=i,frame_id=i,group='pp_continuous',source='pp',target=[0,0],supervised=[True,False]) for i in range(5)]
        result=mixed_samples(samples,42,0)
        self.assertEqual(sum(s.get('source')=='pp' for s in result)*2,len(result))
        self.assertEqual(result,mixed_samples(samples,42,0))


class ContextAndProtocolTests(unittest.TestCase):
    def test_sparse_context_uses_original_video_and_shared_arrays(self):
        import av
        from functools import lru_cache
        from ttnet.events.context import ensure_continuous_context
        class Corpus:
            identity='fixture'
            def __init__(self,source):
                self.context_stores={}
                self.videos={'v':dict(meta=dict(source=str(source),fps=25,n_frames=18),frames={
                    i:dict(frame_id=i,source_frame_id=i,timestamp_sec=i*.04,label_status='unknown',xy=None,path='') for i in (0,1,12,17)})}
            @lru_cache(maxsize=32)
            def image(self,vid,fid):
                if vid in self.context_stores:return self.context_stores[vid].get_rgb(vid,fid)
                return np.full((32,64,3),250,np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'source.mp4'
            with av.open(str(video),'w') as out:
                stream=out.add_stream('libx264',rate=25);stream.width=64;stream.height=32;stream.pix_fmt='yuv420p'
                for i in range(18):
                    frame=av.VideoFrame.from_ndarray(np.full((32,64,3),i*10,np.uint8),format='rgb24')
                    for packet in stream.encode(frame):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
            with patch('ttnet.events.context.CACHE',root/'cache'):
                corpus=Corpus(video);ensure_continuous_context(corpus,'v')
                self.assertEqual(sorted(corpus.videos['v']['frames']),list(range(18)))
                self.assertTrue(np.all(corpus.image('v',12)==250)) # preserve existing cached pixels
                self.assertLess(corpus.image('v',4).mean(),80)
                resumed=Corpus(video);ensure_continuous_context(resumed,'v')
                np.testing.assert_array_equal(corpus.image('v',4),resumed.image('v',4))
                self.assertFalse(list(root.rglob('*.jpg')))

    def test_native_video_preserves_variable_pts_and_original_frame_ids(self):
        import av
        from fractions import Fraction
        from ttnet.events.inference import predict_video
        cfg=dict(temporal_mode='fixed3',global_input=[64,32],local_input=[32,32],
                 local_crop_side_1080=256,localization_threshold=0.,sigma=1.5,amp=False)
        system=EventSystem(cfg,torch.device('cpu'),initialize=False);system.configure('frozen',False)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'vfr.mp4'
            with av.open(str(path),'w') as out:
                stream=out.add_stream('libx264',rate=25);stream.width=64;stream.height=32;stream.pix_fmt='yuv420p'
                for i in range(18):
                    frame=av.VideoFrame.from_ndarray(np.full((32,64,3),i,np.uint8),format='rgb24')
                    frame.pts=i*40+(i//4)*40;frame.time_base=Fraction(1,1000)
                    for packet in stream.encode(frame):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
            rows,records,performance=predict_video(system,path,'v',target_fps=None,progress=False)
            self.assertEqual([r['source_frame_id'] for r in rows],list(range(18)))
            deltas=np.diff([r['timestamp_sec'] for r in rows])
            self.assertGreater(deltas.max(),deltas.min()*1.5)
            self.assertEqual(len(records),18);self.assertTrue(performance['native_frame_ids'])

    def test_streaming_association_matches_full_video_without_second_raw_cache(self):
        import av
        from ttnet.events.context import stream_context_features
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.mp4'
            with av.open(str(source),'w') as out:
                stream=out.add_stream('libx264',rate=25);stream.width=96;stream.height=64;stream.pix_fmt='yuv420p'
                for i in range(17):
                    frame=av.VideoFrame.from_ndarray(np.full((64,96,3),i*12,np.uint8),format='rgb24')
                    for packet in stream.encode(frame):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
            with av.open(str(source)) as container:pixels=[f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
            corpus=TinyCorpus();corpus.images=pixels
            corpus.videos['v']['meta'].update(source=str(source),fps=25)
            for i,r in corpus.videos['v']['frames'].items():r['source_frame_id']=i
            wanted=[0,1,5,6,7,10,11,12,15,16]
            for i in wanted:corpus.images[i]=np.full((64,96,3),230,np.uint8)
            cfg=dict(tracker_type='association',temporal_mode='fixed3',global_input=[64,32],local_input=[32,32],
                     local_crop_side_1080=256,localization_threshold=0.,sigma=1.5,amp=False)
            system=EventSystem(cfg,torch.device('cpu'),initialize=False);system.configure('frozen',False)
            with torch.inference_mode():full,records,_=system.forward_frames(corpus,'v',range(17))
            corpus.videos['v']['frames']={i:r for i,r in corpus.videos['v']['frames'].items() if i in wanted}
            with torch.inference_mode():actual=list(stream_context_features(system,corpus,'v',wanted))
            for fid,f,r in actual:
                torch.testing.assert_close(f,full[fid],atol=0,rtol=0);self.assertEqual(r,records[fid])
            self.assertFalse(list(root.rglob('*.npy')))

    def test_annotation_review_split_and_unknown_guards(self):
        from ttnet.common import save,save_jsonl,sha
        from ttnet.events.continuous import validate_annotations
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'v.mp4';source.write_bytes(b'fixture')
            save(root/'metadata/dataset.json',dict(dataset_id='fixture'))
            for part,vid in [('train','a'),('val','b')]:
                save_jsonl(root/f'splits/{part}.jsonl',[dict(video_id=vid,group=vid,video=str(source),video_sha256=sha(source),split=part)])
                save_jsonl(root/f'labels/{part}.jsonl',[dict(video_id=vid,frame_id=i,resolution=dict(width=64,height=32)) for i in range(3)])
            doc=dict(schema=1,ball_dataset_id='fixture',ball_split_hashes={p:sha(root/f'splits/{p}.jsonl') for p in ('train','val')},videos=[])
            for part,vid in [('train','a'),('val','b')]:
                doc['videos'].append(dict(video_id=vid,match_group=vid,split=part,source_sha256=sha(source),view='v1',status='reviewed',reviewed_by='human',
                    reviewed_intervals=[[0,2]],unknown_intervals=[[2,2]],events=[dict(frame_id=1,kind='bounce',x=10,y=10)],net_reviewed=False))
            self.assertEqual(validate_annotations(doc,root)['train'],1)
            bad=copy.deepcopy(doc);bad['videos'][0]['status']='pending_review'
            with self.assertRaises(ValueError):validate_annotations(bad,root)
            bad=copy.deepcopy(doc);bad['videos'][0]['events'][0]['frame_id']=2
            with self.assertRaises(ValueError):validate_annotations(bad,root)
            bad=copy.deepcopy(doc);bad['videos'][0]['split']='val'
            with self.assertRaises(ValueError):validate_annotations(bad,root)

    def test_new_training_selects_landing_f1_not_sparse_loss(self):
        from argparse import Namespace
        from ttnet.events.train import train
        class Corpus:
            def __init__(self,root):
                self.identity='fixture';self.metadata=dict(complete=True,pilot=False)
                self.split=dict(split_id='fixture',train=['a'],val=['b'],test=[])
                self.videos={v:dict(meta=dict(fps=25,source_identity=dict(soft_radius_sec=.04))) for v in 'ab'}
                self.samples=dict(val=[],train=[dict(video_id='a',anchor=i,frame_id=i,group=k,target=[1.,0.],supervised=[True,True]) for i,k in enumerate(('bounce','net','empty_event'))])
        def attach(corpus,path):
            corpus.samples['train'].append(dict(video_id='pp_a',anchor=0,frame_id=0,group='pp_continuous',source='pp',target=[0.,0.],supervised=[True,False]))
        class System(torch.nn.Module):
            def __init__(self,cfg,device):
                super().__init__();self.config=cfg;self.mode='frozen';self.global_model=torch.nn.Linear(2,2);self.local_model=torch.nn.Linear(2,2);self.head=torch.nn.Linear(9,2)
            def configure(self,mode,training):
                self.train(training)
                for m in (self.global_model,self.local_model):m.requires_grad_(False)
        anchors=[dict(target=[1.,0.],probabilities=[.9,.1],supervised=[True,True])]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('g','l'):(root/name).write_bytes(b'fixture')
            cfg=json.loads(Path('ttnet/src/configs/event.json').read_text())
            cfg.update(global_checkpoint=str(root/'g'),local_checkpoint=str(root/'l'),amp=False,epochs=2,
                       best_metric='pp_landing_macro_f1',sampling_protocol='openttgames_pp_50_50',pp_annotations='fixture')
            path=root/'config.json';path.write_text(json.dumps(cfg))
            args=Namespace(config=path,dataset=root,cache=root/'cache',run=root/'run',device='cpu',mode='frozen',smoke=False,resume=False,restart_incomplete=False)
            continuous=[(dict(macro_f1=f,landing=dict(fp=2),threshold=.6),[],{}) for f in (.2,.8)]
            with patch('ttnet.events.train.EventCorpus',Corpus),patch('ttnet.events.train.EventSystem',System),\
                 patch('ttnet.events.continuous.attach_pp',side_effect=attach),patch('ttnet.events.train.build_cache'),\
                 patch('ttnet.events.train.load_cache',return_value={}),patch('ttnet.events.train.cached_forward',side_effect=lambda s,c,k,x:s.head(torch.ones(1,9))),\
                 patch('ttnet.events.train.validate',side_effect=[(.1,anchors),(.9,anchors)]),\
                 patch('ttnet.events.continuous.validate_continuous',side_effect=continuous):
                train(args)
            state=torch.load(root/'run/checkpoints/best.pt',weights_only=False)
            self.assertEqual(state['epoch'],2)
            self.assertEqual(state['thresholds'],[.6,.5])


if __name__=='__main__':unittest.main()
