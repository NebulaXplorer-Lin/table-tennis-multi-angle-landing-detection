from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from ttnet.temporal import slots_for_mode


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False),
                                  nn.GroupNorm(4, outgoing), nn.SiLU(),
                                  nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False),
                                  nn.GroupNorm(4, outgoing))
        self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Conv2d(incoming, outgoing, 1, stride, bias=False)

    def forward(self, x):
        return F.silu(self.main(x) + self.skip(x))


class TemporalFusion(nn.Module):
    """Keep ordered spatial maps, explicit masks and real elapsed time."""
    def __init__(self, channels, slots):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(slots*(channels+2), channels, 1, bias=False),
                                     nn.GroupNorm(4,channels), nn.SiLU())
        self.context = ResidualBlock(channels,channels)

    def forward(self, features, valid, offsets):
        b,t,c,h,w = features.shape
        mask = valid.to(features.dtype).reshape(b,t,1,1,1)
        masked = features*mask
        mask_map = mask.expand(b,t,1,h,w)
        # Express time in units of one 25fps interval; padding time is explicitly zero.
        time_map = (offsets.to(features.dtype).reshape(b,t,1,1,1)/.04*mask).expand(b,t,1,h,w)
        stacked = torch.cat([masked,mask_map,time_map],dim=2).reshape(b,t*(c+2),h,w)
        return self.context(F.silu(masked[:,t//2]+self.project(stacked)))


class BallLocalizer(nn.Module):
    """Independent global/local 2D model; feature branch is shared with localization.

    Event descriptor uses spatially pooled, localization-trained FPN features;
    there are no untrained projection parameters or Event Head in this stage.
    """
    def __init__(self, temporal_mode="single"):
        super().__init__()
        self.temporal_mode=temporal_mode
        self.slots=slots_for_mode(temporal_mode)
        channels = [16, 32, 64, 96]
        self.encoder = nn.ModuleList([ResidualBlock(a,b,2) for a,b in zip([3]+channels[:-1],channels)])
        if temporal_mode!="single":
            self.temporal = nn.ModuleList([TemporalFusion(c,self.slots) for c in channels])
        self.lateral = nn.ModuleList([nn.Conv2d(c,32,1) for c in channels])
        self.fuse = nn.Sequential(ResidualBlock(32,32), nn.Conv2d(32,1,1))
        nn.init.constant_(self.fuse[-1].bias, -2.19)

    def forward(self, x, frame_valid=None, time_offsets_sec=None, return_spatial=False):
        if self.temporal_mode!="single":
            if x.ndim!=5 or x.shape[1]!=self.slots:
                raise ValueError(f"expected [B,{self.slots},3,H,W]")
            b,t,c,h,w=x.shape
            if frame_valid is None or time_offsets_sec is None:
                raise ValueError("multi-frame localization requires masks and elapsed time")
            # Zero before encoding and after encoding: invalid data cannot influence output.
            mask=frame_valid.to(x.dtype).reshape(b,t,1,1,1)
            x=(x*mask).reshape(b*t,c,h,w)
        levels = []
        for i,block in enumerate(self.encoder):
            x = block(x)
            if self.temporal_mode=="single":levels.append(x)
            else:
                maps=x.reshape(b,t,x.shape[1],x.shape[2],x.shape[3])
                levels.append(self.temporal[i](maps,frame_valid,time_offsets_sec))
        size = levels[0].shape[-2:]
        f = self.lateral[0](levels[0])
        for lateral, level in zip(self.lateral[1:], levels[1:]):
            f = f + F.interpolate(lateral(level), size=size, mode="bilinear", align_corners=False)
        f = F.silu(f)
        heatmap = torch.sigmoid(self.fuse(f))
        # 32 channels x 2x2 positions retains coarse spatial layout = 128 dimensions.
        # Static input H/W on deployment permit an equivalent fixed pooling kernel.
        # This avoids legacy ONNX export losing spatial sizes after dynamic Resize.
        pool_kernel = getattr(self, "export_pool_kernel", None)
        descriptor = (F.avg_pool2d(f, pool_kernel) if pool_kernel is not None
                      else F.adaptive_avg_pool2d(f, (2,2))).flatten(1)
        return (heatmap, descriptor, f) if return_spatial else (heatmap, descriptor)


class LocalizationLoss(nn.Module):
    def forward(self, prediction, target, valid):
        foreground = target > .01
        squared = (prediction-target).square()
        axes = (1,2,3)
        fg = (squared*foreground).sum(axes)/foreground.sum(axes).clamp_min(1)
        bg = (squared*~foreground).sum(axes)/(~foreground).sum(axes).clamp_min(1)
        return ((fg+bg)*valid).sum()/valid.sum().clamp_min(1)
