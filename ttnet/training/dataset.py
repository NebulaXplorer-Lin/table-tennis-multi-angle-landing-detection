from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ttnet.common import jsonl
from ttnet.data_tools.geometry import crop_image, crop_point, crop_transform, letterbox, letterbox_transform, map_point
from .scales import input_size as configured_input_size, validate_size

MEAN = np.array([.485,.456,.406], np.float32)
STD = np.array([.229,.224,.225], np.float32)


def normalize(rgb):
    return np.ascontiguousarray(((rgb.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1))


def read_rgb(path):
    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def gaussian(width, height, center, sigma=1.5):
    if center is None:
        return np.zeros((1,height,width), np.float32)
    x,y = center
    xx,yy = np.arange(width, dtype=np.float32)[None,:], np.arange(height,dtype=np.float32)[:,None]
    return np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma))[None]


class BallDataset(Dataset):
    def __init__(self, dataset, split, stage="global", augment=False, sigma=1.5, predictions=None, all_frames=False,
                 input_size=None, crop_side_at_1080=256):
        self.dataset, self.stage, self.augment, self.sigma = dataset, stage, augment, sigma
        self.rows = jsonl(dataset/"labels"/f"{split}.jsonl")
        if not all_frames:
            self.rows = [r for r in self.rows if r["label_status"] in ["positive","negative"]]
        self.predictions = predictions
        self.gt_crops = augment
        self.input_size = validate_size(input_size or configured_input_size({}, stage), stage)
        self.crop_side_at_1080 = crop_side_at_1080

    def __len__(self):
        return len(self.rows)

    def transform_for_row(self, row, input_size=None):
        width, height = validate_size(input_size or self.input_size, self.stage)
        r = row["resolution"]
        if self.stage == "global":
            return letterbox_transform(r["width"], r["height"], width, height)
        p = (self.predictions or {}).get((row["video_id"], row["frame_id"]))
        if p is None or not p["valid"]:
            return None
        return crop_transform(p["x"], p["y"], r["width"], r["height"], width, self.crop_side_at_1080)

    def item(self, i, gt_crop=None, input_size=None):
        row = self.rows[i]
        width, height = validate_size(input_size or self.input_size, self.stage)
        valid = row["label_status"] in ["positive","negative"]
        center = row["ball_centers"][0] if row["label_status"]=="positive" else None
        transform = None
        if self.stage == "global":
            rgb, transform = letterbox(read_rgb(self.dataset/row["frame_path"]), width, height)
            point = map_point(*center, transform) if center else None
        else:
            w,h = row["resolution"]["width"],row["resolution"]["height"]
            use_gt = self.gt_crops if gt_crop is None else gt_crop
            if use_gt and center is not None:
                side = self.crop_side_at_1080*h/1080
                cx,cy = np.asarray(center)+np.random.uniform(-.2,.2,2)*side
            else:
                p = (self.predictions or {}).get((row["video_id"], row["frame_id"]))
                if p is None or not p["valid"]:
                    rgb = np.zeros((height,width,3),np.uint8)
                    return torch.from_numpy(normalize(rgb)), torch.zeros((1,height//2,width//2)), torch.tensor(0.,dtype=torch.float32), i
                cx,cy = p["x"],p["y"]
            transform = crop_transform(cx,cy,w,h,width,self.crop_side_at_1080)
            rgb = crop_image(read_rgb(self.dataset/row["frame_path"]),transform)
            point = crop_point(*center, transform) if center else None
            # A known ball outside this ROI establishes a local empty target.
            if point and not (0<=point[0]<width and 0<=point[1]<height):
                point = None
        if self.augment:
            if np.random.random()<.5:
                rgb = rgb[:,::-1].copy()
                if point:
                    point = (rgb.shape[1]-1-point[0], point[1])
            rgb = np.clip(rgb.astype(np.float32)*np.random.uniform(.85,1.15)+np.random.uniform(-8,8),0,255).astype(np.uint8)
        hh,ww = rgb.shape[0]//2,rgb.shape[1]//2
        heat = gaussian(ww,hh,(point[0]/2,point[1]/2) if point else None,self.sigma)
        return torch.from_numpy(normalize(rgb)),torch.from_numpy(heat),torch.tensor(float(valid)),i

    def __getitem__(self,i):
        if isinstance(i, tuple):
            index, size = i
            return self.item(index, input_size=size)
        return self.item(i)
