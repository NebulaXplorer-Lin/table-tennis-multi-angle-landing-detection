"""Real-frame compatibility and gradient checks; never trains on fabricated labels."""
import argparse
import copy
import json
from pathlib import Path

import torch
from ttnet.common import DATA, save, sha
from ttnet.configuration import load_config
from ttnet.evaluation.ball_corpus import BallCorpus
from ttnet.events.data import assemble_window
from ttnet.events.engine import EventSystem, cache_identity
from ttnet.models.event import EventHead


def verify(output, historical=None, device="cpu"):
    torch.set_num_threads(2);torch.manual_seed(42)
    corpus=BallCorpus(DATA);vid=corpus.split['val'][0];ids=list(range(min(15,len(corpus.rows[vid]))))
    results={};identities={}
    for name in 'ABCD':
        cfg,_=load_config(f'configs/experiments/event_comparison_{name}.json');cfg['amp']=False
        system=EventSystem(cfg,torch.device(device));system.configure('frozen',False)
        before=[p.detach().clone() for p in system.global_model.parameters()]
        with torch.no_grad():features,records,_=system.forward_frames(corpus,vid,ids)
        logits=system.head(*assemble_window(corpus,vid,6,features,records,system.device))
        # A probe objective, not an event label and not an optimizer update.
        logits.square().mean().backward()
        if any(p.grad is not None for p in system.global_model.parameters()):raise AssertionError('frozen gradients')
        if not all(torch.equal(a,p) for a,p in zip(before,system.global_model.parameters())):raise AssertionError('frozen weights changed')
        if not all(torch.isfinite(p.grad).all() for p in system.head.parameters() if p.grad is not None):raise AssertionError('invalid head gradient')
        restored=EventSystem(cfg,torch.device(device),initialize=False);restored.load_state_dict(system.state_dict());restored.configure('frozen',False)
        actual=restored.head(*assemble_window(corpus,vid,6,features,records,system.device))
        torch.testing.assert_close(logits,actual,atol=0,rtol=0)
        identities[name]=cache_identity(system,corpus)
        results[name]=dict(passed=True,frames=len(ids),head_gradient_l1=sum(float(p.grad.abs().sum()) for p in system.head.parameters() if p.grad is not None),
            max_local_rois=max(len(r.get('roi_transforms',[r['crop_transform']] if r.get('crop_transform') else [])) for r in records.values()),checkpoint_restored_exactly=True)
    if identities['A']!=identities['C'] or identities['B']!=identities['D'] or identities['A']==identities['B']:raise AssertionError('feature pairing')
    compatibility=None
    if historical:
        state=torch.load(historical,map_location='cpu',weights_only=False)
        old=EventSystem(state['config'],torch.device(device),initialize=False);old.load_state_dict(state['system']);old.configure('frozen',False)
        cfg,_=load_config('configs/experiments/event_comparison_A.json')
        current=EventSystem(cfg,torch.device(device),initialize=False);current.load_state_dict(state['system']);current.configure('frozen',False)
        with torch.no_grad():
            f,r,_=old.forward_frames(corpus,vid,ids);g,s,_=current.forward_frames(corpus,vid,ids)
            if r!=s:raise AssertionError('legacy observations differ')
            for i in ids:torch.testing.assert_close(f[i],g[i],atol=0,rtol=0)
            a=old.head(*assemble_window(corpus,vid,6,f,r,old.device));b=current.head(*assemble_window(corpus,vid,6,g,s,current.device))
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        compatibility=dict(checkpoint_sha256=sha(historical),exact_features=True,exact_observations=True,exact_logits=True)
    result=dict(passed=True,video=vid,dataset_id=corpus.identity,backend=f'PyTorch {device} FP32',
        probe='real video frames; squared logits only for gradient check; no synthetic event labels or optimizer updates',
        schemes=results,feature_pairs=['A=C','B=D','A!=B'],historical_compatibility=compatibility)
    save(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--historical',type=Path);p.add_argument('--device',default='cpu')
    a=p.parse_args();print(json.dumps(verify(a.output,a.historical,a.device),indent=2))


if __name__=='__main__':main()
