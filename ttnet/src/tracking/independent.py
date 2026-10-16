"""Baseline selector: strongest independent peak with a local 7x7 centroid."""
import numpy as np


def decode_heatmap(heat):
    heat = np.asarray(heat).squeeze()
    iy,ix = np.unravel_index(np.argmax(heat),heat.shape)
    confidence = float(heat[iy,ix])
    y1,y2,x1,x2 = max(0,iy-3),min(heat.shape[0],iy+4),max(0,ix-3),min(heat.shape[1],ix+4)
    patch = heat[y1:y2,x1:x2].astype(np.float64)
    weights = np.maximum(patch-patch.min(),0)
    total = weights.sum()
    if total > 1e-12:
        xx,yy = np.meshgrid(np.arange(x1,x2),np.arange(y1,y2))
        return float((xx*weights).sum()/total)*2,float((yy*weights).sum()/total)*2,confidence
    return float(ix)*2,float(iy)*2,confidence
