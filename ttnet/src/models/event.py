"""Masked, non-causal visual/trajectory event head (13 frame receptive field)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.conv = nn.Conv1d(128, 128, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(.1)

    def forward(self, x, mask):
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return (x + self.dropout(F.silu(self.norm(y)))) * mask


class EventHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_norm = nn.LayerNorm(128)
        self.local_norm = nn.LayerNorm(128)
        self.visual = nn.Sequential(nn.Linear(256, 96), nn.LayerNorm(96), nn.SiLU())
        self.motion = nn.Sequential(nn.Linear(9, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.temporal = nn.ModuleList([TemporalBlock(d) for d in (1, 2, 3)])
        self.classifier = nn.Linear(128, 2)

    def forward(self, features, motion, frame_valid):
        if features.ndim != 3 or features.shape[1:] != (13, 256):
            raise ValueError("features must be [B,13,256]")
        if motion.shape != (*features.shape[:2], 9) or frame_valid.shape != features.shape[:2]:
            raise ValueError("motion/mask shape mismatch")
        valid = frame_valid.bool().unsqueeze(-1)
        # where rather than multiply: even NaN padding must not contaminate a window.
        features = torch.where(valid, features, 0)
        motion = torch.where(valid, motion, 0)
        g = self.global_norm(features[..., :128])
        local_valid = (motion[..., 7:8] > .5) & valid
        l = self.local_norm(torch.where(local_valid, features[..., 128:], 0))
        l = torch.where(local_valid, l, 0)
        x = torch.cat([self.visual(torch.cat([g, l], -1)), self.motion(motion)], -1)
        x = x * valid
        for block in self.temporal:
            x = block(x, valid)
        return self.classifier(x[:, 6])


def event_loss(logits, target, supervised, per_sample=False):
    target = torch.where(supervised.bool(), target, 0)
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    if per_sample:return (loss*supervised).sum(-1)/supervised.sum(-1).clamp_min(1)
    return (loss * supervised).sum() / supervised.sum().clamp_min(1)


class TransformerEventHead(nn.Module):
    """Center-frame classifier with continuous time encoding and padding masks."""
    def __init__(self, layers=2, dimension=128, heads=4, ffn=256, dropout=.1):
        super().__init__()
        if dimension % heads or dimension < 2 or layers < 1 or ffn < 1 or not 0 <= dropout < 1:
            raise ValueError('invalid Transformer dimensions/dropout')
        self.global_norm = nn.LayerNorm(128)
        self.local_norm = nn.LayerNorm(128)
        self.input = nn.Linear(265, dimension)
        layer = nn.TransformerEncoderLayer(dimension, heads, ffn, dropout,
                                            activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(dimension, 2)
        self.register_buffer('time_frequencies', torch.exp(torch.linspace(0, 6, (dimension+1)//2)), persistent=False)

    def forward(self, features, motion, frame_valid):
        if features.ndim != 3 or features.shape[1:] != (13,256):
            raise ValueError('features must be [B,13,256]')
        if motion.shape != (*features.shape[:2],9) or frame_valid.shape != features.shape[:2]:
            raise ValueError('motion/mask shape mismatch')
        valid = frame_valid.bool()
        f = torch.where(valid[...,None], features, 0)
        m = torch.where(valid[...,None], motion, 0)
        local_valid = valid[...,None] & (m[...,7:8] > .5)
        local = torch.where(local_valid, self.local_norm(torch.where(local_valid,f[...,128:],0)), 0)
        x = self.input(torch.cat([self.global_norm(f[...,:128]),local,m], -1))
        angles = m[...,8:9] * self.time_frequencies
        encoding = torch.stack([angles.sin(),angles.cos()], -1).flatten(-2)[...,:x.shape[-1]]
        x = torch.where(valid[...,None], x+encoding, 0)
        # Avoid all-masked attention NaNs; dummy slot is removed after attention.
        safe = valid.clone()
        safe[~safe.any(dim=1),6] = True
        x = self.encoder(x, src_key_padding_mask=~safe)
        center = torch.where(valid[:,6,None], x[:,6], 0)
        return self.classifier(center)


class ContextEventHead(nn.Module):
    """Identical front end and time encoding for controlled TCN/attention trials."""
    def __init__(self, kind='tcn', scene_mode='relative', dropout=.1, time_encoding=False):
        super().__init__()
        self.kind=kind;self.scene_mode=scene_mode;self.time_encoding=time_encoding
        self.global_norm=nn.LayerNorm(128);self.local_norm=nn.LayerNorm(128)
        self.visual=nn.Sequential(nn.Linear(256,96),nn.LayerNorm(96),nn.SiLU())
        self.motion=nn.Sequential(nn.Linear(9,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.scene=nn.Sequential(nn.Linear(9,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU()) if scene_mode!='none' else None
        self.fusion=nn.Sequential(nn.Linear(160 if self.scene else 128,128),nn.LayerNorm(128),nn.SiLU())
        if kind=='tcn':
            self.temporal=nn.ModuleList([TemporalBlock(d) for d in (1,2,3)])
            for block in self.temporal:block.dropout.p=dropout
        elif kind=='transformer':
            layer=nn.TransformerEncoderLayer(128,4,256,dropout,activation='gelu',batch_first=True)
            self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        else:raise ValueError(kind)
        self.classifier=nn.Linear(128,2)
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,6,64)),persistent=False)

    def forward(self,features,motion,frame_valid):
        if features.shape[1:]!=(13,256) or motion.shape!=(*features.shape[:2],18):raise ValueError('context head requires visual256 + motion9/scene9')
        valid=frame_valid.bool();m=torch.where(valid[...,None],motion,0);f=torch.where(valid[...,None],features,0)
        lv=valid[...,None] & (m[...,7:8]>.5)
        local=torch.where(lv,self.local_norm(torch.where(lv,f[...,128:],0)),0)
        branches=[self.visual(torch.cat([self.global_norm(f[...,:128]),local],-1)),self.motion(m[...,:9])]
        if self.scene:
            scene=m[...,9:].clone()
            if self.scene_mode=='box':scene[...,4:6]=0;scene[...,7]=0
            branches.append(self.scene(scene)*(scene[...,6:7]>.5))
        x=self.fusion(torch.cat(branches,-1))
        if self.time_encoding:
            angles=m[...,8:9]*self.frequencies;x=x+torch.stack([angles.sin(),angles.cos()],-1).flatten(-2)
        x=torch.where(valid[...,None],x,0)
        if self.kind=='tcn':
            for block in self.temporal:x=block(x,valid[...,None])
        else:
            safe=valid.clone();safe[~safe.any(1),6]=True
            x=self.temporal(x,src_key_padding_mask=~safe)
        return self.classifier(torch.where(valid[:,6,None],x[:,6],0))
