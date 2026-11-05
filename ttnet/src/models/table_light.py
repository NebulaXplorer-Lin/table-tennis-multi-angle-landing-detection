"""Native single-frame table encoder and lightweight dense CNN head."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ttnet.models.table_encoder import SharedGlobalTableHead, load_table_checkpoint


class DepthwiseBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming,incoming,3,2,1,groups=incoming,bias=False),
            nn.Conv2d(incoming,outgoing,1,bias=False),nn.GroupNorm(4,outgoing),nn.SiLU(),
            nn.Conv2d(outgoing,outgoing,3,1,1,groups=outgoing,bias=False),
            nn.Conv2d(outgoing,outgoing,1,bias=False),nn.GroupNorm(4,outgoing),nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class NativeTableEncoder(nn.Module):
    """RGB -> 16/32/64/96-channel pyramid -> stride4 map32 -> center/bbox.

    Trained from scratch using table supervision. It has no detector dependency,
    pretrained weight download or random projection frozen outside supervision.
    """
    source = 'independent_cnn'
    channels = 32
    stride = 4

    def __init__(self, input_size=(512,288)):
        super().__init__()
        self.input_size = tuple(input_size)
        self.encoder = nn.ModuleList(DepthwiseBlock(a,b) for a,b in zip((3,16,32,64),(16,32,64,96)))
        self.lateral = nn.ModuleList(nn.Conv2d(c,32,1,bias=False) for c in (32,64,96))
        self.spatial = nn.Sequential(nn.Conv2d(32,32,3,padding=1,bias=False),nn.GroupNorm(4,32),nn.SiLU())
        self.head = SharedGlobalTableHead(32)
        self.architecture = dict(kind='native_depthwise_fpn_v1',encoder_channels=[16,32,64,96],
                                 spatial_channels=32,stride=4,input=list(input_size))

    def spatial_features(self, images):
        x = images*2-1
        levels = []
        for i,block in enumerate(self.encoder):
            x = block(x)
            if i:
                levels.append(x)
        size = levels[0].shape[-2:]
        features = self.lateral[0](levels[0])
        for lateral,level in zip(self.lateral[1:],levels[1:]):
            features = features+F.interpolate(lateral(level),size=size,mode='bilinear',align_corners=False)
        return self.spatial(features)

    def forward(self, images):
        return self.head(self.spatial_features(images))

    def loss(self, prediction, targets):
        return self.head.loss(prediction,targets)

    def candidates(self, prediction, confidence=.1, nms_iou=.7):
        return self.head.candidates(prediction,confidence,nms_iou,self.input_size)


class NativeTableMLP(NativeTableEncoder):
    """Independent trainable spatial encoder followed by a full-image table MLP."""
    def __init__(self, input_size=(512,288), pool=(16,4)):
        super().__init__(input_size)
        from ttnet.scene.table import TableHead
        self.head=TableHead(pool)
        self.architecture.update(kind='native_depthwise_fpn_mlp_v1',table_pool=list(pool),hidden=128)

    def forward(self, images):
        spatial=self.spatial_features(images)
        return self.head(spatial),spatial

    def loss(self, prediction, targets):
        from ttnet.scene.table import box_iou
        l1=F.l1_loss(prediction,targets);giou=(1-box_iou(prediction,targets,True)).mean()
        return l1+giou,dict(l1=l1,giou=giou)

    def candidates(self, prediction, confidence=.1, nms_iou=.7):
        # One regressed table per image; score=1 is a presence convention, not quality.
        boxes=prediction.detach().float()*prediction.new_tensor([*self.input_size,*self.input_size]).float()
        return [torch.cat([box,box.new_tensor([1.,0.])])[None] for box in boxes]


def load_spatial_table(path, device='cpu', expected_global_sha=None):
    state = torch.load(path,map_location='cpu',weights_only=False)
    if state.get('config',{}).get('source') != 'independent_cnn':
        return load_table_checkpoint(path,device,expected_global_sha)
    kind=state.get('architecture',{}).get('kind')
    if state.get('schema')!='table_visual_v1' or kind not in ('native_depthwise_fpn_v1','native_depthwise_fpn_mlp_v1'):
        raise ValueError('unsupported native table checkpoint')
    model = (NativeTableMLP(state['config']['input'],state['architecture']['table_pool'])
             if kind=='native_depthwise_fpn_mlp_v1' else NativeTableEncoder(state['config']['input']))
    model.load_state_dict(state['model'])
    return model.to(device),state
