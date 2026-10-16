"""Image-space peaks, with no GT input or oracle selection."""
import cv2
import numpy as np
from ttnet.data_tools.geometry import map_point

def candidates(heat,transform,resolution,count=5,min_separation=6):
    heat=np.asarray(heat).squeeze()
    maximum=cv2.dilate(heat,np.ones((3,3),np.uint8))
    ys,xs=np.nonzero(heat==maximum)
    order=np.argsort(-heat[ys,xs],kind='stable')
    chosen=[];anchors=[]
    for idx in order:
        iy,ix=int(ys[idx]),int(xs[idx])
        if any((ix-px)**2+(iy-py)**2<min_separation**2 for px,py in anchors):continue
        y1,y2,x1,x2=max(0,iy-3),min(heat.shape[0],iy+4),max(0,ix-3),min(heat.shape[1],ix+4)
        patch=heat[y1:y2,x1:x2].astype(np.float64)
        weights=np.maximum(patch-patch.min(),0);total=weights.sum()
        if total>1e-12:
            xx,yy=np.meshgrid(np.arange(x1,x2),np.arange(y1,y2))
            x,y=float((xx*weights).sum()/total)*2,float((yy*weights).sum()/total)*2
        else:x,y=ix*2.,iy*2.
        x,y=map_point(x,y,transform,inverse=True)
        if not (0<=x<resolution['width'] and 0<=y<resolution['height']):continue
        anchors.append((ix,iy))
        chosen.append(dict(x=x,y=y,confidence=float(heat[iy,ix])))
        if len(chosen)==count:break
    return chosen

