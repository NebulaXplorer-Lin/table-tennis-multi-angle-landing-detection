"""Learned candidate likelihoods with per-path GRU states and bounded beam search."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from ttnet.scene.table import geometry
from ttnet.tracking.association import DEFAULTS

INPUT_DIM=145


def candidate_inputs(candidates,descriptor,table,resolution,dt,use_table=True):
    values=[]
    for p in candidates:
        scene=geometry(table,p,resolution['width'],resolution['height']) if use_table else np.zeros(9,np.float32)
        values.append(np.concatenate(([p['x']/resolution['width'],p['y']/resolution['height'],p['confidence'],float(p.get('source')=='local')],((p.get('appearance') or [0.,0.,0.]) if p.get('source')!='local' else [0.,0.,0.]),scene,[dt],np.asarray(descriptor,dtype=np.float32).reshape(128))))
    return np.asarray(values,dtype=np.float32).reshape(-1,INPUT_DIM)


class CandidateGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode=nn.Sequential(nn.Linear(INPUT_DIM,128),nn.LayerNorm(128),nn.SiLU())
        self.null=nn.Parameter(torch.zeros(128))
        self.history=nn.Linear(128,128,bias=False)
        self.score=nn.Sequential(nn.Tanh(),nn.Linear(128,1))
        self.recurrent=nn.GRUCell(128,128)

    def distribution(self,inputs,mask,hidden):
        encoded=self.encode(inputs)
        # Last candidate is always the explicitly modeled missing observation.
        encoded=torch.cat([encoded[:,:-1],self.null.expand(inputs.shape[0],1,-1)],1)
        logits=self.score(encoded+self.history(hidden)[:,None]).squeeze(-1)
        return logits.masked_fill(~mask,-torch.inf).log_softmax(-1),encoded

    def advance(self,encoded,hidden,index):
        return self.recurrent(encoded[torch.arange(len(index),device=index.device),index],hidden)


@dataclass
class PathState:
    hidden: torch.Tensor
    score: float=0.
    history: tuple=()
    missing: int=0


class LearnedBeamTracker:
    def __init__(self,model,beam_size=8,config=None,use_table=True):
        self.model=model;self.beam_size=beam_size;self.config=dict(DEFAULTS,**(config or {}));self.use_table=use_table
        self.reset()

    def reset(self):
        self.beam=[PathState(next(self.model.parameters()).new_zeros(128))]
        self.last_frame=self.last_time=None;self.pending=None

    def predict(self,timestamp):
        h=self.beam[0]
        if not h.history or h.missing>self.config['reset_after']:return None
        t,p=h.history[-1];xy=np.array([p['x'],p['y']],float)
        if len(h.history)>1:
            pt,q=h.history[-2];xy+=(xy-[q['x'],q['y']])*((timestamp-t)/(t-pt))
        return dict(x=float(xy[0]),y=float(xy[1]),valid=False,predicted=True,missing=h.missing)

    def context(self,descriptor,table,resolution):
        self.descriptor=np.asarray(descriptor,dtype=np.float32);self.table=table;self.resolution=resolution

    def _expand(self,candidates,frame_id,timestamp,base,previous_time):
        dt=timestamp-previous_time if previous_time is not None else 0.
        inputs=candidate_inputs(candidates,self.descriptor,self.table,self.resolution,dt,self.use_table)
        inputs=np.concatenate([inputs,np.zeros((1,INPUT_DIM),np.float32)])
        device=next(self.model.parameters()).device
        hidden=torch.stack([p.hidden for p in base]);x=torch.from_numpy(inputs).to(device).unsqueeze(0).expand(len(base),-1,-1)
        mask=torch.ones(x.shape[:2],dtype=torch.bool,device=device)
        with torch.no_grad():
            logp,emb=self.model.distribution(x,mask,hidden)
            totals=logp+logp.new_tensor([p.score for p in base])[:,None]
            scores,flat=totals.flatten().topk(min(self.beam_size,totals.numel()))
            parent=flat//x.shape[1];choice=flat%x.shape[1]
            next_hidden=self.model.advance(emb[parent],hidden[parent],choice)
        paths=[];selected=[]
        for j,(pi,ci,score) in enumerate(zip(parent.tolist(),choice.tolist(),scores.tolist())):
            old=base[pi];p=candidates[ci] if ci<len(candidates) else None
            missing=0 if p else old.missing+1
            history=(old.history+((timestamp,p),))[-6:] if p else old.history
            if missing>self.config['reset_after']:history=()
            paths.append(PathState(next_hidden[j],score,history,missing));selected.append(p)
        # Subtract a common constant for numerical stability; likelihood ranking is unchanged.
        maximum=paths[0].score
        for p in paths:p.score-=maximum
        return paths,selected[0]

    def update(self,candidates,frame_id,timestamp):
        if not math.isfinite(timestamp) or (self.last_frame is not None and (frame_id!=self.last_frame+1 or timestamp<=self.last_time)):
            raise ValueError('learned association requires complete ordered context')
        valid=[dict(p) for p in candidates if p.get('valid',True) and np.isfinite([p['x'],p['y'],p['confidence']]).all() and p['confidence']>=self.config['min_confidence']]
        self.pending=(self.beam,valid,frame_id,timestamp,self.last_time)
        self.beam,selected=self._expand(valid,frame_id,timestamp,self.beam,self.last_time)
        self.last_frame,self.last_time=frame_id,timestamp
        return selected

    def finalize(self,locals):
        base,raw,fid,ts,previous=self.pending
        valid=[dict(p) for p in locals if p.get('valid')]
        self.beam,selected=self._expand(raw+valid,fid,ts,base,previous)
        self.pending=None
        return selected

    def state(self,timestamp):
        return dict(prediction=self.predict(timestamp),missing=self.beam[0].missing,hypotheses=len(self.beam),observation_valid=self.beam[0].missing==0)
