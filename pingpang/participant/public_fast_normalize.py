"""Exact float32 uint8 normalization through a 256-entry per-channel table."""
import cv2
import numpy as np
from ttnet.training.dataset import normalize as reference_normalize

_pixels=np.repeat(np.arange(256,dtype=np.uint8)[:,None,None],3,axis=2)
_table=np.ascontiguousarray(reference_normalize(_pixels).transpose(1,2,0))


def normalize(rgb):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        return reference_normalize(rgb)
    return np.ascontiguousarray(cv2.LUT(rgb,_table).transpose(2,0,1))

