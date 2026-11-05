"""Three-frame high-resolution Global locator with localization-trained features."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .localizer import BallLocalizer, ResidualBlock, TemporalFusion


class StableSpatialMean(nn.Module):
    """Large constant maps need accurate accumulation in PyTorch and ORT."""
    def forward(self,x):
        return x.float().mean(dim=-1,keepdim=True).mean(dim=-2,keepdim=True).to(x.dtype)


class ExportStableGroupNorm(nn.GroupNorm):
    def forward(self,x):
        if not torch.onnx.is_in_onnx_export():
            return F.group_norm(x,self.num_groups,self.weight,self.bias,self.eps)
        # Avoid reducing a flattened full-resolution group in one long FP32 sum.
        grouped=x.reshape(x.shape[0],self.num_groups,self.num_channels//self.num_groups,x.shape[2],x.shape[3])
        mean=grouped.mean(4,keepdim=True).mean(3,keepdim=True).mean(2,keepdim=True)
        centered=grouped-mean
        variance=centered.square().mean(4,keepdim=True).mean(3,keepdim=True).mean(2,keepdim=True)
        normalized=(centered*torch.rsqrt(variance+self.eps)).reshape_as(x)
        return normalized*self.weight.reshape(1,-1,1,1)+self.bias.reshape(1,-1,1,1)


def export_stable_norms(module):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.GroupNorm):
            replacement=ExportStableGroupNorm(child.num_groups,child.num_channels,eps=child.eps,affine=child.affine)
            replacement.load_state_dict(child.state_dict());setattr(module,name,replacement)
        else:export_stable_norms(child)


class SEResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels))
        self.attention = nn.Sequential(
            StableSpatialMean(),
            nn.Conv2d(channels, max(1, channels // 16), 1, bias=False), nn.ReLU(),
            nn.Conv2d(max(1, channels // 16), channels, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        y = self.main(x)
        return F.silu(x + y * self.attention(y))


class ResolutionExchange(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList(SEResidualBlock(c) for c in channels)
        self.paths = nn.ModuleDict()
        for target, outgoing in enumerate(channels):
            for source, incoming in enumerate(channels):
                if target == source:
                    continue
                if source > target:
                    path = nn.Sequential(nn.Conv2d(incoming, outgoing, 1, bias=False),
                                         nn.GroupNorm(4, outgoing))
                else:
                    layers = []
                    for step in range(target - source):
                        following = outgoing if step == target - source - 1 else incoming
                        layers.extend([nn.Conv2d(incoming, following, 3, 2, 1, bias=False),
                                       nn.GroupNorm(4, following)])
                        if step != target - source - 1:
                            layers.append(nn.SiLU())
                        incoming = following
                    path = nn.Sequential(*layers)
                self.paths[f'{source}_{target}'] = path

    def forward(self, levels):
        levels = [block(x) for block, x in zip(self.blocks, levels)]
        outputs = []
        for target, base in enumerate(levels):
            result = base
            for source, x in enumerate(levels):
                if source == target:
                    continue
                y = self.paths[f'{source}_{target}'](x)
                if y.shape[-2:] != base.shape[-2:]:
                    y = F.interpolate(y, size=base.shape[-2:], mode='nearest')
                result = result + y
            outputs.append(F.silu(result))
        return outputs


class HighResolutionSELocalizer(BallLocalizer):
    """Full-resolution branch, two exchanges; legacy heatmap/128-D contract."""
    def __init__(self, temporal_mode='fixed3'):
        # Reuse forward-independent conventions, without keeping unused FPN weights.
        nn.Module.__init__(self)
        from ttnet.temporal import slots_for_mode
        self.temporal_mode = temporal_mode
        self.slots = slots_for_mode(temporal_mode)
        channels = [16, 32, 64, 96]
        self.encoder = nn.ModuleList(ResidualBlock(a, b, 1 if i == 0 else 2)
                                     for i, (a, b) in enumerate(zip([3] + channels[:-1], channels)))
        if temporal_mode != 'single':
            self.temporal = nn.ModuleList(TemporalFusion(c, self.slots) for c in channels)
        self.exchanges = nn.ModuleList(ResolutionExchange(channels) for _ in range(2))
        self.high_projections = nn.ModuleList(nn.Conv2d(c, 16, 1, bias=False) for c in channels[1:])
        self.output_projection = nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1, bias=False),
                                               nn.GroupNorm(4, 32), nn.SiLU())
        self.fuse = nn.Sequential(ResidualBlock(32, 32), nn.Conv2d(32, 1, 1))
        nn.init.constant_(self.fuse[-1].bias, -2.19)
        export_stable_norms(self)

    def forward(self, x, frame_valid=None, time_offsets_sec=None, return_spatial=False):
        multi = self.temporal_mode != 'single'
        if multi:
            if x.ndim != 5 or x.shape[1] != self.slots:
                raise ValueError(f'expected [B,{self.slots},3,H,W]')
            if frame_valid is None or time_offsets_sec is None:
                raise ValueError('multi-frame localization requires masks and elapsed time')
            b, t, c, h, w = x.shape
            x = (x * frame_valid.to(x.dtype).reshape(b, t, 1, 1, 1)).reshape(b*t, c, h, w)
        levels = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            levels.append(self.temporal[i](x.reshape(b, t, *x.shape[1:]), frame_valid, time_offsets_sec)
                          if multi else x)
        for exchange in self.exchanges:
            levels = exchange(levels)
        high = levels[0]
        for projection, level in zip(self.high_projections, levels[1:]):
            high = high + F.interpolate(projection(level), size=high.shape[-2:], mode='nearest')
        spatial = self.output_projection(F.silu(high))
        heatmap = torch.sigmoid(self.fuse(spatial))
        b,c,h,w=spatial.shape
        descriptor=spatial.float().reshape(b,c,2,h//2,2,w//2).mean(dim=5).mean(dim=3).to(spatial.dtype).flatten(1)
        return (heatmap, descriptor, spatial) if return_spatial else (heatmap, descriptor)
