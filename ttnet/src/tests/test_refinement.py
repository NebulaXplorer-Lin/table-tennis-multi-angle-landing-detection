import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ttnet.events.continuous import mixed_samples
from ttnet.common import sha
from ttnet.events.refinement import load_hard_negative_keys, mine_hard_negatives, smooth_window_tables, stratified_pp_samples
from ttnet.events.data import assemble_window, sample_identity
from ttnet.events.inference import predict_windows
from ttnet.events.train import cached_forward_batch
from ttnet.models.event import ContextEventHead
from ttnet.tracking.autoregressive import CandidateGRU, INPUT_DIM
from ttnet.tracking.train_ar import compatible_resume_config, drop_correct_candidates, run_epoch


class RefinementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_source_balance_strata_shortage_and_reproducibility(self):
        public=[dict(video_id='tt',frame_id=i,anchor=i,group=g,target=[0,0],supervised=[True,True])
                for g in ('bounce','net','empty_event') for i in range(10)]
        pp=[dict(video_id='pp_train',frame_id=i,anchor=i,source='pp',group='pp_continuous',
                 target=[int(i<10),0],supervised=[True,False]) for i in range(100)]
        hard={('pp_train',i) for i in range(10,20)}
        cfg=dict(pp_positive_fraction=.2,pp_hard_negative_fraction=.2)
        a=mixed_samples(public+pp,42,0,cfg,hard)
        self.assertEqual(a,mixed_samples(public+pp,42,0,cfg,hard))
        chosen=[s for s in a if s.get('source')=='pp']
        self.assertEqual((len(a),len(chosen)),(60,30))
        self.assertEqual(sum(s['target'][0] for s in chosen),6)
        self.assertEqual(sum((s['video_id'],s['frame_id']) in hard for s in chosen),6)
        shortage=stratified_pp_samples(pp,100,np.random.default_rng(1),.2,.2,{('pp_train',10)})
        self.assertEqual(sum(s['frame_id']==10 for s in shortage),1)
        self.assertEqual(sum(s['target'][0] for s in shortage),20)
        self.assertEqual(mixed_samples(public+pp,42,0),mixed_samples(public+pp,42,0,{}))

    def test_miner_excludes_gt_unknown_and_nonpeaks(self):
        ann=dict(reviewed_intervals=[[0,50]],unknown_intervals=[[35,40]],
                 events=[dict(kind='bounce',frame_id=10)])
        scores={i:.1 for i in range(51)}
        scores.update({10:.99,13:.8,23:.6,25:.9,37:.95,47:.7})
        rows=[dict(frame_id=i,probabilities=[v,0]) for i,v in scores.items()]
        self.assertEqual([r['frame_id'] for r in mine_hard_negatives(rows,ann)],[25,47])

    def test_manifest_rejects_validation_samples_and_changed_identity(self):
        train=[dict(video_id='pp_train',frame_id=20,anchor=20,source='pp',
                    target=[0,0],supervised=[True,False])]
        corpus=SimpleNamespace(identity='dataset',metadata=dict(pp_annotation_id='labels'),
                               samples={'train':train},pp_annotations={'pp_train':{'events':[]}})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'splits').mkdir();(root/'splits/train.jsonl').write_text('train\n')
            manifest=dict(identity=dict(dataset_id='dataset',annotation_id='labels',
                          train_samples_sha256=sample_identity(train),train_split_sha256=sha(root/'splits/train.jsonl')),
                          samples=[dict(video_id='pp_train',frame_id=20)])
            path=root/'hard.json'
            with patch('ttnet.events.refinement.DATA',root):
                path.write_text(json.dumps(manifest))
                self.assertEqual(load_hard_negative_keys(path,corpus),{('pp_train',20)})
                manifest['samples'][0]['video_id']='pp_val';path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'training backgrounds'):load_hard_negative_keys(path,corpus)
                manifest['identity']['annotation_id']='other';path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'identity mismatch'):load_hard_negative_keys(path,corpus)

    def test_ema_masks_resets_and_shared_window_features(self):
        frames={i:dict(timestamp_sec=i*.04,frame_valid=True) for i in range(13)}
        rec={i:dict(table=dict(valid=True,bbox_xyxy=[10+i,10,50+i,50]),prediction=None,local_valid=False) for i in frames}
        original=copy.deepcopy(rec)
        cfg=dict(context_head=True,scene_mode='box',event_window_mode='frames',
                 table_smoothing='window_ema',table_smoothing_tau_sec=.15)
        tables=smooth_window_tables(list(frames),frames,rec,cfg)
        self.assertGreater(tables[1]['bbox_xyxy'][0],10)
        self.assertLess(tables[1]['bbox_xyxy'][0],11)
        rec[2]['table']=None
        rec[5]['table']['bbox_xyxy']=[70,70,95,95]
        tables=smooth_window_tables(list(frames),frames,rec,cfg)
        self.assertIsNone(tables[2]);self.assertEqual(tables[3]['bbox_xyxy'],rec[3]['table']['bbox_xyxy'])
        self.assertEqual(tables[5]['bbox_xyxy'],rec[5]['table']['bbox_xyxy'])
        gap=smooth_window_tables([0,3],frames,rec,cfg)
        self.assertEqual(gap[3]['bbox_xyxy'],rec[3]['table']['bbox_xyxy'])
        class Corpus:
            videos={'v':dict(meta=dict(width=100,height=100,n_frames=13),frames=frames)}
            def window_ids(self,vid,center):return list(range(13))
        features={i:torch.zeros(256) for i in frames}
        a=assemble_window(Corpus(),'v',6,features,rec,torch.device('cpu'),cfg)
        b=assemble_window(Corpus(),'v',6,features,copy.deepcopy(rec),torch.device('cpu'),cfg)
        for x,y in zip(a,b):torch.testing.assert_close(x,y)
        self.assertEqual(a[1][0,2,15].item(),0)
        corpus=Corpus();cache={'v':(features,rec)};sample=dict(video_id='v',frame_id=6)
        system=SimpleNamespace(config=dict(cfg,amp=False),device=torch.device('cpu'),
                               head=ContextEventHead('tcn').eval())
        with torch.no_grad():
            for mode in ('none','window_ema'):
                system.config['table_smoothing']=mode
                logits=cached_forward_batch(system,corpus,cache,[sample])
                rows=predict_windows(system,corpus,'v',features,rec,centers=[6])
                np.testing.assert_allclose(rows[0]['probabilities'],logits.sigmoid()[0].numpy(),atol=1e-7)
        self.assertEqual(original[1],rec[1])

    def test_dropout_unknown_masks_and_weighted_gradients(self):
        mask=torch.zeros(1,6,8,dtype=torch.bool);mask[:,:,[0,1,7]]=True
        targets=torch.zeros_like(mask);targets[0,[0,1,3,4],0]=True
        targets[0,5,7]=True
        nearest=torch.tensor([[0,0,7,0,0,7]])
        unknown_mask=mask[0,2].clone()
        self.assertEqual(drop_correct_candidates(mask,targets,nearest,torch.ones(1,6,dtype=torch.bool),1.,3),4)
        self.assertFalse(mask[0,0,0]);self.assertTrue(mask[0,0,1]);self.assertTrue(targets[0,0,7])
        self.assertFalse(targets[0,2].any());torch.testing.assert_close(mask[0,2],unknown_mask)
        model=CandidateGRU()
        seq=dict(video='v',x=torch.randn(6,8,INPUT_DIM),mask=mask[0],target=targets[0],nearest=nearest[0])
        counts=[]
        hook=model.recurrent.register_forward_hook(lambda *args:counts.append(1))
        with torch.no_grad():metrics=run_epoch(model,[seq])
        self.assertEqual(len(counts),6);self.assertEqual(metrics['supervised_frames'],5)
        hook.remove()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        metrics=run_epoch(model,[seq],opt,null_loss_weight=5.,candidate_dropout_prob=.05)
        self.assertTrue(np.isfinite(metrics['weighted_loss']))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_legacy_ar_resume_defaults_do_not_enable_refinements(self):
        old=dict(seed=42,use_table=True)
        current=dict(old,null_loss_weight=1.,candidate_dropout_prob=0.,candidate_dropout_span=3,
                     selection_metric='val_loss',reference_checkpoint_sha256=None,minimum_positive_accuracy=0.)
        self.assertTrue(compatible_resume_config(old,current))
        self.assertFalse(compatible_resume_config(old,dict(current,null_loss_weight=5.)))
        self.assertEqual(old,dict(seed=42,use_table=True))


if __name__=='__main__':unittest.main()
