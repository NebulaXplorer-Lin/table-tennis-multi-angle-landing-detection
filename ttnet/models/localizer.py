from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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


class BallLocalizer(nn.Module):
    """Independent global/local 2D model; feature branch is shared with localization.

    Event descriptor uses spatially pooled, localization-trained FPN features;
    there are no untrained projection parameters or Event Head in this stage.
    """
    def __init__(self):
        super().__init__()
        channels = [16, 32, 64, 96]
        self.encoder = nn.ModuleList([ResidualBlock(a,b,2) for a,b in zip([3]+channels[:-1],channels)])
        self.lateral = nn.ModuleList([nn.Conv2d(c,32,1) for c in channels])
        self.fuse = nn.Sequential(ResidualBlock(32,32), nn.Conv2d(32,1,1))
        nn.init.constant_(self.fuse[-1].bias, -2.19)

    def forward(self, x):
        levels = []
        for block in self.encoder:
            x = block(x); levels.append(x)
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
        return heatmap, descriptor


class LocalizationLoss(nn.Module):
    def forward(self, prediction, target, valid):
        foreground = target > .01
        squared = (prediction-target).square()
        axes = (1,2,3)
        fg = (squared*foreground).sum(axes)/foreground.sum(axes).clamp_min(1)
        bg = (squared*~foreground).sum(axes)/(~foreground).sum(axes).clamp_min(1)
        return ((fg+bg)*valid).sum()/valid.sum().clamp_min(1)
