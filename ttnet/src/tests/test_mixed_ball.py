import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import io
from contextlib import redirect_stdout

import numpy as np
import torch

from ttnet.common import save, sha
from ttnet.configuration import load_config
from ttnet.data_tools.prepare_mixed_ball import ball_label
from ttnet.models.build import build_localizer
from ttnet.models.event import EventHead
from ttnet.models.localizer import BallLocalizer, LocalizationLoss
from ttnet.training.dataset import native_window
from ttnet.training.mixed_sampling import MixedSourceSampler
from ttnet.evaluation.inference import load_model
from ttnet.events.engine import EventSystem


class MixedDataTests(unittest.TestCase):
    def test_mixed_local_caches_share_gt_and_preserve_native_pixels(self):
        import cv2
        from ttnet.common import save_jsonl
        from ttnet.training.cache import ensure_local_cache
        from ttnet.training.dataset import BallDataset
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/'data';maps={}
            for split in ('train','val'):
                labels=[];maps[split]={}
                for native in (False,True):
                    vid=('native_' if native else 'pp_')+split
                    context=[]
                    for fid in ((9,10,11) if native else (0,1,2)):
                        path=dataset/'frames'/vid/f'{fid}.png';path.parent.mkdir(parents=True,exist_ok=True)
                        image=np.zeros((32,32,3),np.uint8);image[8:22,8+fid%3:20]=180;cv2.imwrite(str(path),image)
                        r=dict(video_id=vid,frame_id=fid,timestamp_sec=fid/(120 if native else 25),frame_path=str(path.relative_to(dataset)),
                               native_sparse=native,video_frame_count=20 if native else 3,source_domain='opentt' if native else 'pp',
                               resolution=dict(width=32,height=32),label_status='positive',ball_centers=[[16.,16.]])
                        context.append(r)
                    if native:save_jsonl(dataset/f'context/{vid}.jsonl',context)
                    selected=[context[1]] if native else context
                    labels.extend(selected)
                    for row in selected:maps[split][vid,row['frame_id']]=dict(x=16.,y=16.,valid=True)
                save_jsonl(dataset/f'labels/{split}.jsonl',labels)
            save(dataset/'metadata/dataset.json',dict(dataset_id='fixture'))
            cfg=dict(local_input=[32,32],local_crop_side_1080=1080,temporal_mode='fixed3',workers=1,source_sampling='pp_opentt_50_50')
            with patch('ttnet.training.cache.CACHE',root/'cache'):
                a=ensure_local_cache(dataset,maps,'global_A',cfg)
                b=ensure_local_cache(dataset,maps,'global_B',cfg)
            self.assertEqual((a/'gt_context').resolve(),(b/'gt_context').resolve());self.assertNotEqual(a,b)
            for gt in (False,True):
                ds=BallDataset(dataset,'train','local',predictions=maps['train'],temporal_mode='fixed3',input_size=(32,32),crop_side_at_1080=1080,local_cache=a)
                fresh=BallDataset(dataset,'train','local',predictions=maps['train'],temporal_mode='fixed3',input_size=(32,32),crop_side_at_1080=1080)
                for obj in (ds,fresh):obj.gt_crops=gt;obj.augmentation_seed=42
                for i in range(len(ds)):
                    cached=ds[i];original=fresh[i]
                    torch.testing.assert_close(cached[0]['images'],original[0]['images'],rtol=0,atol=0)
                    torch.testing.assert_close(cached[1],original[1],rtol=0,atol=0)

    def test_result_index_ignores_dataset_review_manifests(self):
        from ttnet.publication import write_index
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            save(root/'output/table_review/manifest.json',dict(dataset_id='table',schema=1))
            write_index(root)
            self.assertEqual(json.loads((root/'output/index.json').read_text()),[])

    def test_epoch_resume_restores_weights_optimizer_and_sampling(self):
        import cv2
        from ttnet.common import save_jsonl
        from ttnet.training.train import train
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/'data'
            for split in ('train','val'):
                rows=[]
                for source in ('pp','opentt'):
                    vid=f'{source}_{split}'
                    for fid in range(4):
                        path=dataset/'frames'/vid/f'{fid}.png';path.parent.mkdir(parents=True,exist_ok=True)
                        image=np.zeros((32,32,3),np.uint8);cv2.circle(image,(12+fid,16),2,(255,255,255),-1);cv2.imwrite(str(path),image)
                        rows.append(dict(video_id=vid,frame_id=fid,timestamp_sec=fid*.04,frame_path=str(path.relative_to(dataset)),
                                         source_domain=source,resolution=dict(width=32,height=32),label_status='positive',ball_centers=[[12.+fid,16.]]))
                save_jsonl(dataset/f'labels/{split}.jsonl',rows)
            save(dataset/'metadata/dataset.json',dict(status='completed',dataset_id='fixture'))
            cfg,_=load_config('configs/experiments/ball_mixed_baseline.json')
            cfg.update(global_input=[32,32],local_input=[32,32],micro_batch=2,effective_batch=4,workers=0,torch_threads=1,
                       resize_cache=False,amp=False,epochs=2)
            def attempt(name,epochs,resume=False):
                run=root/name;log=run/'logs'/('attempt_002' if resume else 'attempt_001')
                (run/'checkpoints').mkdir(parents=True,exist_ok=True);log.mkdir(parents=True)
                args=SimpleNamespace(stage='global',dataset=dataset,device='cpu',global_checkpoint=None,resume=resume,smoke=False,started='fixture')
                with patch('ttnet.training.train.write_results'),redirect_stdout(io.StringIO()):
                    train(args,run,dict(cfg,epochs=epochs),log)
                return torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False)
            full=attempt('full',2)
            attempt('resumed',1)
            resumed=attempt('resumed',2,True)
            for key in full['model']:torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
            self.assertEqual(full['optimizer']['param_groups'],resumed['optimizer']['param_groups'])
            self.assertEqual(json.loads((root/'full/sampling/epoch_002.json').read_text()),json.loads((root/'resumed/sampling/epoch_002.json').read_text()))

    def test_native_stream_covers_unknown_and_tail_without_gt_selection(self):
        import av
        from ttnet.evaluation.native_ball import evaluate_native
        from ttnet.common import save_jsonl
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video=root/'source.mp4';times=[]
            with av.open(str(video),'w') as container:
                stream=container.add_stream('mpeg4',rate=120);stream.width=64;stream.height=48;stream.pix_fmt='yuv420p'
                for fid in range(20):
                    frame=av.VideoFrame.from_ndarray(np.full((48,64,3),fid,np.uint8),format='rgb24')
                    for packet in stream.encode(frame):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
            with av.open(str(video)) as container:
                times=[float(f.pts*f.time_base) for f in container.decode(video=0)]
            annotation=root/'balls.json';save(annotation,{'0':dict(x=1,y=1),'10':dict(x=-1,y=-1),'19':dict(x=1,y=1)})
            save(root/'metadata/videos/v.json',dict(frames=20,width=64,height=48,fps=120,video=str(video),annotation=str(annotation)))
            save_jsonl(root/'splits/val.jsonl',[dict(video_id='v',source_domain='opentt')])
            (root/'native_store/v').mkdir(parents=True);np.save(root/'native_store/v/all_timestamps.npy',times)
            g=root/'g';g.write_bytes(b'g');l=root/'l';l.write_bytes(b'l')
            class FakeSystem:
                def __init__(self,cfg,device):self.config=cfg;self.seen=[]
                def configure(self,*args):pass
                def reset_tracking(self,*args):pass
                def forward_frames(self,corpus,vid,ids):
                    for fid in ids:
                        assert corpus.videos[vid]['frames'][fid]['label_status']=='unknown'
                        self.seen.append(fid)
                    pred=dict(x=1.,y=1.,confidence=.9,valid=True,source='global')
                    return {},{ids[0]:dict(frame_id=ids[0],prediction=pred,global_candidates=[pred],roi_transforms=[],crop_transform=None)},{}
            with patch('ttnet.evaluation.native_ball.system_config',return_value={}),patch('ttnet.evaluation.native_ball.EventSystem',FakeSystem):
                result=evaluate_native(root,g,l,root/'result')
            self.assertEqual(result['full_frames'],20);self.assertEqual(result['scored_frames'],3)
            self.assertEqual(result['metric']['localization']['10']['tp'],2)
            self.assertEqual(result['metric']['localization']['10']['fp'],1)
            records=[json.loads(s) for s in (root/'result/v_predictions.jsonl').read_text().splitlines()]
            self.assertEqual([r['frame_id'] for r in records],list(range(20)))

    def test_sparse_native_neighbors_are_not_annotation_neighbors(self):
        frames={i:dict(frame_id=i,timestamp_sec=i/120) for i in (9,10,11,99,100,101)}
        row=dict(video_id='v',frame_id=10,video_frame_count=200,timestamp_sec=10/120)
        w=native_window(row,frames)
        self.assertEqual(w['frame_ids'],[9,10,11])
        np.testing.assert_allclose(w['time_offsets_sec'],[-1/120,0,1/120])
        with self.assertRaises(ValueError):native_window(dict(row,frame_id=11),frames)
        edge={0:dict(timestamp_sec=0),1:dict(timestamp_sec=1/120)}
        self.assertEqual(native_window(dict(row,frame_id=0,timestamp_sec=0),edge)['frame_valid'],[False,True,True])

    def test_missing_and_partial_labels_are_unknown(self):
        self.assertEqual(ball_label(None,100,100),('unknown',[]))
        self.assertEqual(ball_label(dict(x=-1,y=5),100,100),('unknown',[]))
        self.assertEqual(ball_label(dict(x=-1,y=-1),100,100),('negative',[]))
        with self.assertRaises(ValueError):ball_label(dict(x=101,y=1),100,100)

    def test_sources_balanced_rotating_and_paired(self):
        rows=[dict(video_id=s,frame_id=i,source_domain=s) for s,n in [('pp',16),('opentt',37)] for i in range(n)]
        a=MixedSourceSampler(rows,42,16);b=MixedSourceSampler(rows,42,16)
        covered=set()
        for epoch in range(4):
            a.set_epoch(epoch);b.set_epoch(epoch)
            self.assertEqual(a.indices,b.indices)
            self.assertEqual(a.manifest['sources'],dict(pp=16,opentt=16))
            for start in range(0,len(a),16):
                self.assertEqual(sum(rows[i]['source_domain']=='pp' for i in a.indices[start:start+16]),8)
            covered.update(i for i in a if rows[i]['source_domain']=='opentt')
        self.assertEqual(len(covered),37)


class NewLocatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_shapes_mask_attention_and_event_gradients(self):
        torch.manual_seed(42)
        cfg=dict(temporal_mode='fixed3',global_localizer_type='high_resolution_se',local_localizer_type='residual_fpn')
        model=build_localizer(cfg,'global')
        x=torch.randn(1,3,3,48,64,requires_grad=True)
        valid=torch.tensor([[False,True,True]]);times=torch.tensor([[0.,0.,1/120]])
        heat,desc,spatial=model(x,valid,times,return_spatial=True)
        self.assertEqual(heat.shape,(1,1,24,32));self.assertEqual(desc.shape,(1,128));self.assertEqual(spatial.shape,(1,32,24,32))
        with torch.no_grad():
            changed=x.detach().clone();changed[:,0]*=100
            actual=model(changed,valid,times)
            torch.testing.assert_close(actual[0],heat.detach(),rtol=0,atol=0)
        f=torch.cat([desc,desc],-1)[:,None,:].expand(-1,13,-1)
        head=EventHead()
        logits=head(f,torch.zeros(1,13,9),torch.ones(1,13,dtype=torch.bool))
        logits.square().mean().backward()
        self.assertGreater(float(x.grad[:,1:].abs().sum()),0)
        attention=[p for n,p in model.named_parameters() if 'attention' in n]
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in attention))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in attention),0)
        self.assertIsInstance(build_localizer(cfg,'local'),BallLocalizer)

    def test_old_checkpoint_and_mixed_event_checkpoint_reload(self):
        cfg=dict(temporal_mode='fixed3',global_localizer_type='high_resolution_se',local_localizer_type='residual_fpn',
                 global_input=[64,48],local_input=[32,32],local_crop_side_1080=256,
                 tracker_type='independent',event_head_type='tcn',localization_threshold=.7,sigma=1.5,amp=False)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);g=root/'g.pt';l=root/'l.pt'
            torch.save(dict(model=build_localizer(cfg,'global').state_dict(),config=dict(cfg,stage='global'),stage='global',dataset_id='mixed'),g)
            torch.save(dict(model=build_localizer(cfg,'local').state_dict(),config=dict(cfg,stage='local'),stage='local',
                            global_checkpoint_sha256=sha(g),dataset_id='mixed'),l)
            system=EventSystem(dict(cfg,global_checkpoint=str(g),local_checkpoint=str(l)),torch.device('cpu'))
            reloaded=EventSystem(copy.deepcopy(system.config),torch.device('cpu'),initialize=False)
            reloaded.load_state_dict(system.state_dict())
            self.assertEqual(type(system.global_model),type(reloaded.global_model))
            old=root/'old.pt';m=BallLocalizer('fixed3')
            torch.save(dict(model=m.state_dict(),config=dict(temporal_mode='fixed3')),old)
            previous,_=load_model(old,torch.device('cpu'))
            for a,b in zip(m.parameters(),previous.parameters()):torch.testing.assert_close(a,b)

    def test_presets_use_identical_training_recipe(self):
        a,_=load_config('configs/experiments/ball_mixed_baseline.json')
        b,_=load_config('configs/experiments/ball_mixed_hr_se.json')
        for k in ('source_sampling','effective_batch','lr','epochs','augmentation','seed','data'):
            self.assertEqual(a[k],b[k])


if __name__=='__main__':unittest.main()
