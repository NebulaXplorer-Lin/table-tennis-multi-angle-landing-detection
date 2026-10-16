import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from ttnet.scene.table import TableHead,table_loss,box_iou,geometry,decode_table
from ttnet.data_tools.geometry import letterbox_transform,map_point
from ttnet.models.event import ContextEventHead
from ttnet.tracking.autoregressive import CandidateGRU,LearnedBeamTracker,INPUT_DIM
from ttnet.events.data import event_window_ids


class SceneTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_coordinate_roundtrip_and_missing(self):
        t=letterbox_transform(1266,720);a=map_point(10,20,t);b=map_point(900,600,t)
        r=decode_table(np.array([*a,*b])/[512,288,512,288],t,dict(width=1266,height=720))
        np.testing.assert_allclose(r['bbox_xyxy'],[10,20,900,600])
        self.assertEqual(geometry(None,None,1280,720).sum(),0)
        g=geometry(r,dict(x=1200,y=700,valid=True),1266,720);self.assertGreater(g[4],.5);self.assertEqual(g[7],1)
        g=geometry(r,None,1266,720);self.assertEqual(g[6],1);self.assertEqual(g[7],0)

    def test_box_gradient(self):
        m=TableHead();p=m(torch.randn(4,32,6,10));target=torch.tensor([[.1,.1,.9,.9]]*4)
        table_loss(p,target).backward();self.assertTrue(all(torch.isfinite(x.grad).all() for x in m.parameters()))
        torch.testing.assert_close(box_iou(target,target),torch.ones(4))

    def test_context_mask_and_reload(self):
        for kind in ('tcn','transformer'):
            m=ContextEventHead(kind).eval();f=torch.randn(2,13,256);motion=torch.zeros(2,13,18);v=torch.ones(2,13,dtype=torch.bool);v[0,:]=False
            a=m(f,motion,v);f[~v]=float('nan');motion[~v]=float('nan');b=m(f,motion,v)
            torch.testing.assert_close(a,b);b.sum().backward();self.assertTrue(torch.isfinite(b).all())
            n=ContextEventHead(kind).eval();n.load_state_dict(m.state_dict());torch.testing.assert_close(n(f,motion,v),b)

    def test_candidate_masks_and_unknown_gradient(self):
        m=CandidateGRU();x=torch.randn(3,8,INPUT_DIM);mask=torch.zeros(3,8,dtype=torch.bool);mask[:,-1]=True;mask[0,:2]=True
        p,e=m.distribution(x,mask,torch.zeros(3,128));self.assertEqual(p[1,-1].item(),0.);self.assertTrue(torch.isneginf(p[1,:-1]).all())
        loss=-p[0,:2].logsumexp(-1);loss.backward();self.assertTrue(all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None))

    def test_beam_recurrent_state_reset_and_order(self):
        m=CandidateGRU().eval();tr=LearnedBeamTracker(m,4);tr.context(np.zeros(128),None,dict(width=100,height=100))
        p=dict(x=30.,y=40.,confidence=.8,source='global')
        tr.update([p],0,0.);tr.finalize([]);self.assertGreater(len(tr.beam),1)
        with self.assertRaises(ValueError):tr.update([p],0,0.)
        tr.update([p],1,.04);tr.finalize([]);self.assertTrue(all(torch.isfinite(h.hidden).all() for h in tr.beam))
        tr.reset();self.assertIsNone(tr.last_frame);self.assertEqual(len(tr.beam),1);self.assertEqual(tr.beam[0].hidden.abs().sum(),0)

    def test_physical_window(self):
        class C:
            videos={'v':{'frames':{i:{'timestamp_sec':i/60} for i in range(80)}}}
        ids=event_window_ids(C(),'v',40,{'event_window_mode':'seconds'})
        self.assertEqual(ids[6],40);self.assertEqual(len(ids),13);self.assertLess(ids[0],34);self.assertGreater(ids[-1],46)
        self.assertEqual(len(set(ids)),13)

if __name__=='__main__':unittest.main()
