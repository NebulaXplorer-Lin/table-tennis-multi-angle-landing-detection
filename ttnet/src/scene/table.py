"""Single table box on frozen spatial features; missing labels are unknown."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ttnet.common import PROJECT, DATA, jsonl
from ttnet.data_tools.geometry import map_point


class TableHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(512,128), nn.SiLU(), nn.Linear(128,4))

    def forward(self, spatial):
        if spatial.ndim == 4:
            spatial = F.adaptive_avg_pool2d(spatial, (4,4))
        z = self.regressor(spatial).sigmoid()
        # Ordered corners without nondifferentiable sorting.
        lo = z[:,:2]
        hi = lo + (1-lo)*z[:,2:]
        return torch.cat([lo,hi],-1)


def box_iou(a,b,giou=False):
    inter=(torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp_min(0).prod(-1)
    union=(a[:,2:]-a[:,:2]).prod(-1)+(b[:,2:]-b[:,:2]).prod(-1)-inter
    iou=inter/union.clamp_min(1e-8)
    if not giou:return iou
    enclosing=(torch.maximum(a[:,2:],b[:,2:])-torch.minimum(a[:,:2],b[:,:2])).prod(-1)
    return iou-(enclosing-union)/enclosing.clamp_min(1e-8)


def table_loss(pred,target):
    return F.l1_loss(pred,target)+(1-box_iou(pred,target,True)).mean()


def annotations(part):
    result={};audit=dict(frames=0,valid=0,missing=0,invalid=0,multiple=0)
    for row in jsonl(DATA/f'splits/{part}.jsonl'):
        doc=json.loads((PROJECT/row['annotation']).read_text())
        w,h=doc['frame_width'],doc['frame_height']
        for frame in doc['frames']:
            audit['frames']+=1
            boxes=[b for a in frame['annotations'] if a['label']=='table' for b in a['bbox_xyxy']]
            if not boxes:audit['missing']+=1;continue
            if len(boxes)!=1:audit['multiple']+=1;continue
            b=np.asarray(boxes[0],dtype=float)
            if b.shape!=(4,) or not np.isfinite(b).all() or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):
                audit['invalid']+=1;continue
            result[(row['video_id'],frame['frame_id'])]=b.tolist();audit['valid']+=1
    return result,audit


def decode_table(box,transform,resolution):
    b=np.asarray(box,dtype=float)*np.array([transform['width'],transform['height']]*2)
    x1,y1=map_point(*b[:2],transform,inverse=True);x2,y2=map_point(*b[2:],transform,inverse=True)
    w,h=resolution['width'],resolution['height']
    b=[max(0.,min(w,x1)),max(0.,min(h,y1)),max(0.,min(w,x2)),max(0.,min(h,y2))]
    return dict(bbox_xyxy=b,valid=bool(np.isfinite(b).all() and b[2]>b[0] and b[3]>b[1]),source='predicted')


def geometry(table,ball,width,height,mode='relative'):
    """9D: normalized cx/cy/w/h, relative ball xy, table/relationship masks, source age."""
    out=np.zeros(9,np.float32)
    if not table or not table.get('valid'):return out
    x1,y1,x2,y2=table['bbox_xyxy'];w=x2-x1;h=y2-y1
    if not np.isfinite([x1,y1,x2,y2]).all() or min(w,h)<=0:return out
    cx,cy=(x1+x2)/2,(y1+y2)/2
    out[:4]=[cx/width,cy/height,w/width,h/height];out[6]=1
    if mode=='relative' and ball and ball.get('valid'):
        out[4:6]=[(ball['x']-cx)/w,(ball['y']-cy)/h];out[7]=1
    out[8]=table.get('age_sec',0.)
    return out
