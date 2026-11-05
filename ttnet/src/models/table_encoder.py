"""Independent YOLO11s and shared-Global table front ends with spatial outputs."""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ttnet.common import PROJECT, sha
from ttnet.data_tools.geometry import map_point
from ttnet.scene.table import box_iou


def ultralytics_api():
    """Use the repository's isolated pinned dependency; never install/download."""
    dependency = PROJECT/'out/yolo-deps'
    if str(dependency) not in sys.path:
        sys.path.insert(0,str(dependency))
    module = importlib.import_module('ultralytics')
    if module.__version__ != '8.3.225':
        raise RuntimeError(f'table adapter requires ultralytics 8.3.225, found {module.__version__}')
    return module


class IndependentTableEncoder(nn.Module):
    """Explicit graph traversal returns the native P3/8 map without hooks."""
    source = 'independent_yolo'

    def __init__(self, pretrained=None, architecture=None):
        super().__init__()
        api = ultralytics_api()
        from ultralytics.nn.tasks import DetectionModel, yaml_model_load
        if architecture is None:
            architecture = yaml_model_load('yolo11s.yaml')
        architecture = copy.deepcopy(architecture)
        if architecture.get('scale') != 's':
            raise ValueError('independent table recipe requires YOLO11s')
        self.detector = DetectionModel(architecture,nc=1,verbose=False)
        self.detector.names = {0:'main_table'}
        self.detector.args = SimpleNamespace(box=7.5,cls=.5,dfl=1.5)
        if pretrained is not None:
            pretrained = Path(pretrained)
            if not pretrained.is_file():
                raise FileNotFoundError(f'pretrained table weights must be prepared explicitly: {pretrained}')
            pretrained_model = api.YOLO(str(pretrained)).model
            if pretrained_model.yaml.get('scale') != 's':
                raise ValueError('pretrained weights must be YOLO11s')
            self.detector.load(pretrained_model,verbose=False)
        detect = self.detector.model[-1]
        self.feature_layer = int(detect.f[0])
        self.channels = detect.cv2[0][0].conv.in_channels
        self.stride = int(detect.stride[0])
        if self.stride != 8:
            raise ValueError('expected highest-resolution detection input at stride8')
        self.architecture = copy.deepcopy(self.detector.yaml)

    def forward(self, images):
        outputs, spatial, x = [], None, images
        for module in self.detector.model:
            if module.f != -1:
                x = outputs[module.f] if isinstance(module.f,int) else [x if i == -1 else outputs[i] for i in module.f]
            x = module(x)
            if module.i == self.feature_layer:
                spatial = x
            outputs.append(x if module.i in self.detector.save else None)
        if spatial is None:
            raise RuntimeError('missing explicit table spatial output')
        return x, spatial

    def loss(self, prediction, targets):
        n = len(targets)
        xywh = torch.cat([(targets[:,:2]+targets[:,2:])/2, targets[:,2:]-targets[:,:2]],-1)
        batch = dict(batch_idx=torch.arange(n,device=targets.device),cls=targets.new_zeros((n,1)),bboxes=xywh)
        total, parts = self.detector.loss(batch,prediction)
        return total.sum()/n, dict(zip(('box','classification','dfl'),parts))

    def candidates(self, prediction, confidence=.1, nms_iou=.7):
        from ultralytics.utils.nms import non_max_suppression
        decoded = prediction[0] if isinstance(prediction,tuple) else prediction
        return non_max_suppression(decoded.detach().float().clone(),conf_thres=confidence,iou_thres=nms_iou,
                                   nc=1,max_det=100,max_time_img=10.)


class SharedGlobalTableHead(nn.Module):
    """Center heatmap plus dense bbox regression on B's unpooled spatial map."""
    source = 'shared_global'

    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels
        self.features = nn.Sequential(nn.Conv2d(channels,32,3,padding=1,bias=False),
                                      nn.GroupNorm(4,32),nn.SiLU())
        self.center = nn.Conv2d(32,1,1)
        self.box = nn.Conv2d(32,4,1)
        nn.init.constant_(self.center.bias,-2.19)

    def forward(self, spatial):
        h = self.features(spatial)
        z = self.box(h).sigmoid()
        low = z[:,:2]
        boxes = torch.cat([low,low+(1-low)*z[:,2:]],1)
        return dict(center_logits=self.center(h),boxes=boxes), spatial

    def loss(self, prediction, targets):
        logits, boxes = prediction['center_logits'], prediction['boxes']
        b, _, h, w = logits.shape
        center = (targets[:,:2]+targets[:,2:])/2
        ix = (center[:,0]*w).long().clamp(0,w-1)
        iy = (center[:,1]*h).long().clamp(0,h-1)
        yy, xx = torch.meshgrid(torch.arange(h,device=logits.device),torch.arange(w,device=logits.device),indexing='ij')
        sigma = ((targets[:,2:]-targets[:,:2])*targets.new_tensor([w,h])).amin(-1).mul(.1).clamp_min(1.)
        target = torch.exp(-((xx[None]-ix[:,None,None])**2+(yy[None]-iy[:,None,None])**2)/(2*sigma[:,None,None]**2))[:,None]
        probability = logits.sigmoid()
        positive = target == 1
        heat = -(positive*(1-probability).pow(2)*F.logsigmoid(logits)
                 + (~positive)*(1-target).pow(4)*probability.pow(2)*F.logsigmoid(-logits)).sum()/b
        selected = boxes[torch.arange(b,device=boxes.device),:,iy,ix]
        l1 = F.l1_loss(selected,targets)
        giou = (1-box_iou(selected,targets,True)).mean()
        return heat+l1+giou,dict(center=heat,l1=l1,giou=giou)

    def candidates(self, prediction, confidence=.1, nms_iou=.7, input_size=(512,288)):
        logits, boxes = prediction['center_logits'],prediction['boxes']
        result = []
        for heat, field in zip(logits,boxes):
            score, index = heat.sigmoid().flatten().max(0)
            if score < confidence:
                result.append(field.new_empty((0,6)))
                continue
            box = field.flatten(1)[:,index]*field.new_tensor([*input_size,*input_size])
            result.append(torch.cat([box,score[None],score.new_zeros(1)])[None].detach())
        return result


def select_table(candidates, transform):
    """Clip in original pixels, then deterministic score/area/coordinate ordering."""
    w,h = transform['original_width'],transform['original_height']
    legal = []
    for row in candidates.detach().float().cpu().tolist():
        if not np.isfinite(row).all():
            continue
        x1,y1 = map_point(*row[:2],transform,inverse=True)
        x2,y2 = map_point(*row[2:4],transform,inverse=True)
        box = [max(0.,min(w,x1)),max(0.,min(h,y1)),max(0.,min(w,x2)),max(0.,min(h,y2))]
        if box[2] > box[0] and box[3] > box[1]:
            legal.append(dict(bbox_xyxy=box,score=row[4]))
    legal.sort(key=lambda r:(-r['score'],-(r['bbox_xyxy'][2]-r['bbox_xyxy'][0])*(r['bbox_xyxy'][3]-r['bbox_xyxy'][1]),*r['bbox_xyxy']))
    return dict(legal[0],valid=True,age_sec=0.,source='predicted') if legal else dict(valid=False,bbox_xyxy=None,score=0.,age_sec=0.,source='predicted')


def spatial_visual(spatial, table, transform, roi_config=None):
    """4x4 bin-centre bilinear sampling; ROI coordinates stop gradient, maps do not.

    Grid convention: input pixel edge x maps to feature coordinate x/stride-.5;
    align_corners=False therefore uses normalized grid 2*x/input_width-1.
    """
    cfg = dict(width_scale=1.10,height_scale=1.50,min_height=.08)
    cfg.update(roi_config or {})
    if spatial.ndim != 3:
        raise ValueError('single-frame table spatial map must be [C,H,W]')
    context = spatial.mean((-2,-1))
    result = dict(context=context,context_valid=True,roi=spatial.new_zeros(spatial.shape[0]*16),roi_valid=False)
    if not table.get('valid'):
        return result
    x1,y1,x2,y2 = table['bbox_xyxy']
    w,h = transform['original_width'],transform['original_height']
    cx,cy = (x1+x2)/2,(y1+y2)/2
    rw,rh = (x2-x1)*cfg['width_scale'],max((y2-y1)*cfg['height_scale'],cfg['min_height']*h)
    box = [max(0.,cx-rw/2),max(0.,cy-rh/2),min(w,cx+rw/2),min(h,cy+rh/2)]
    a,b = map_point(*box[:2],transform),map_point(*box[2:],transform)
    fractions = (torch.arange(4,device=spatial.device,dtype=torch.float32)+.5)/4
    xs,ys = a[0]+fractions*(b[0]-a[0]),a[1]+fractions*(b[1]-a[1])
    yy,xx = torch.meshgrid(ys,xs,indexing='ij')
    grid = torch.stack([2*xx/transform['width']-1,2*yy/transform['height']-1],-1)[None]
    roi = F.grid_sample(spatial[None].float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0]
    result.update(roi=roi.flatten().to(spatial.dtype),roi_valid=True,roi_box_xyxy=box)
    return result


def load_table_checkpoint(path, device='cpu', expected_global_sha=None):
    state = torch.load(path,map_location='cpu',weights_only=False)
    cfg = state['config']
    if state.get('schema') != 'table_visual_v1':
        raise ValueError('expected independent/shared spatial table checkpoint')
    if cfg['source'] == 'shared_global':
        if expected_global_sha is None or state['global_sha256'] != expected_global_sha:
            raise ValueError('shared table/Global checkpoint mismatch')
        model = SharedGlobalTableHead(cfg['channels'])
    else:
        model = IndependentTableEncoder(architecture=state['architecture'])
    model.load_state_dict(state['model'])
    return model.to(device),state
