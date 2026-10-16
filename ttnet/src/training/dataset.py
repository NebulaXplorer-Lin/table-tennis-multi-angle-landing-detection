from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import lru_cache

from ttnet.common import jsonl
from ttnet.data_tools.geometry import crop_image, crop_point, crop_transform, letterbox, letterbox_transform, map_point
from .scales import input_size as configured_input_size, validate_size
from ttnet.temporal import sample_window
from ttnet.data_tools.frame_store import FrameStore

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
                 input_size=None, crop_side_at_1080=256, temporal_mode="single", window_half_span_sec=1/30,
                 input_cache=None,local_cache=None,augmentation=None):
        self.dataset, self.stage, self.augment, self.sigma = dataset, stage, augment, sigma
        self.rows = jsonl(dataset/"labels"/f"{split}.jsonl")
        self.video_rows={}
        for row in self.rows:
            self.video_rows.setdefault(row["video_id"],[]).append(row)
        self.timestamps={}
        for video,rows in self.video_rows.items():
            rows.sort(key=lambda r:r["frame_id"])
            if [r["frame_id"] for r in rows]!=list(range(len(rows))):
                raise ValueError("temporal sampling requires all ordered source frames")
            self.timestamps[video]=[r["timestamp_sec"] for r in rows]
        if not all_frames:
            self.rows = [r for r in self.rows if r["label_status"] in ["positive","negative"]]
        self.predictions = predictions
        self.frame_store = FrameStore(dataset)
        self.path_rows = {r['frame_path']:r for rows in self.video_rows.values() for r in rows}
        self.input_stores = {}
        self.local_stores = {}
        self.local_pairs = {}
        self.augmentation = dict(horizontal_flip=.5,brightness_scale=[.85,1.15],
                                 brightness_offset=[-8,8],gt_crop_jitter_fraction=.2)
        self.augmentation.update(augmentation or {})
        self.gt_crops = augment
        self.input_size = validate_size(input_size or configured_input_size({}, stage), stage)
        self.crop_side_at_1080 = crop_side_at_1080
        self.temporal_mode=temporal_mode
        self.window_half_span_sec=window_half_span_sec
        self.input_cache=input_cache
        self.local_cache=local_cache

    @lru_cache(maxsize=64)
    def original(self,path):
        return self.frame_store.row_rgb(self.path_rows[path])

    def global_image(self,row,width,height):
        if self.input_cache:
            from pathlib import Path
            store_path = Path(self.input_cache)/f'{width}x{height}'
            if (store_path/'frame_store.json').is_file():
                if (width,height) not in self.input_stores:
                    self.input_stores[width,height] = FrameStore(store_path)
                return self.input_stores[width,height].get_rgb(row['video_id'],row['frame_id'])
            cached=Path(self.input_cache)/f"{width}x{height}"/row["video_id"]/f"{row['frame_id']:06d}.png"
            if cached.exists():return read_rgb(cached)
        image,_=letterbox(self.original(row["frame_path"]),width,height)
        return image

    def temporal_index(self,row):
        return sample_window(self.timestamps[row["video_id"]],row["frame_id"],self.temporal_mode,self.window_half_span_sec)

    def local_image(self,center,neighbor,transform,use_gt):
        if self.local_cache:
            from pathlib import Path
            kind='gt_context' if use_gt else 'predicted'
            path=Path(self.local_cache)/kind/center['video_id']/f"{center['frame_id']:06d}_{neighbor['frame_id']:06d}.png"
            reference=Path(self.local_cache)/kind/'frame_store.json'
            image=None
            if reference.is_file():
                if kind not in self.local_stores:self.local_stores[kind]=FrameStore(reference.parent)
                key=(kind,center['video_id'])
                if key not in self.local_pairs:
                    import json
                    pair_file=reference.parent/center['video_id']/'pairs.json'
                    self.local_pairs[key]=json.loads(pair_file.read_text()) if pair_file.exists() else {}
                ordinal=self.local_pairs[key].get(f"{center['frame_id']}:{neighbor['frame_id']}")
                if ordinal is not None:image=self.local_stores[kind].get_rgb(center['video_id'],ordinal)
            elif path.exists():image=read_rgb(path)
            if image is not None:
                if not use_gt and image.shape[:2]==(transform['output_size'],transform['output_size']):return image
                if use_gt:
                    from .cache import gt_carrier_transform
                    carrier=gt_carrier_transform(center,self.crop_side_at_1080,self.augmentation['gt_crop_jitter_fraction'])
                    relative=dict(transform,left=transform['left']-carrier['left'],top=transform['top']-carrier['top'])
                    return crop_image(image,relative)
        return crop_image(self.original(neighbor['frame_path']),transform)

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
            transform=self.transform_for_row(row,(width,height))
            rgb=self.global_image(row,width,height)
            point = map_point(*center, transform) if center else None
        else:
            w,h = row["resolution"]["width"],row["resolution"]["height"]
            use_gt = self.gt_crops if gt_crop is None else gt_crop
            if use_gt and center is not None:
                side = self.crop_side_at_1080*h/1080
                jitter=self.augmentation['gt_crop_jitter_fraction']
                cx,cy = np.asarray(center)+np.random.uniform(-jitter,jitter,2)*side
            else:
                p = (self.predictions or {}).get((row["video_id"], row["frame_id"]))
                if p is None or not p["valid"]:
                    rgb = np.zeros((height,width,3),np.uint8)
                    if self.temporal_mode=="single":payload=torch.from_numpy(normalize(rgb))
                    else:
                        index=self.temporal_index(row);n=len(index["frame_ids"])
                        payload=dict(images=torch.zeros(n,3,height,width),frame_valid=torch.tensor(index["frame_valid"]),
                                     time_offsets_sec=torch.tensor(index["time_offsets_sec"],dtype=torch.float32))
                    return payload, torch.zeros((1,height//2,width//2)), torch.tensor(0.,dtype=torch.float32), i
                cx,cy = p["x"],p["y"]
            transform = crop_transform(cx,cy,w,h,width,self.crop_side_at_1080)
            rgb = self.local_image(row,row,transform,use_gt)
            point = crop_point(*center, transform) if center else None
            # A known ball outside this ROI establishes a local empty target.
            if point and not (0<=point[0]<width and 0<=point[1]<height):
                point = None
        index=self.temporal_index(row)
        images=[]
        for fid in index["frame_ids"]:
            if fid is None:images.append(None);continue
            neighbor=self.video_rows[row["video_id"]][fid]
            if fid==row["frame_id"]:image=rgb
            elif self.stage=="global":image=self.global_image(neighbor,width,height)
            else:image=self.local_image(row,neighbor,transform,use_gt)
            images.append(image)
        if self.augment:
            flip=np.random.random()<self.augmentation['horizontal_flip']
            scale=np.random.uniform(*self.augmentation['brightness_scale']);shift=np.random.uniform(*self.augmentation['brightness_offset'])
            if flip:
                if point:
                    point = (width-1-point[0], point[1])
            images=[None if im is None else np.clip((im[:,::-1] if flip else im).astype(np.float32)*scale+shift,0,255).astype(np.uint8) for im in images]
        hh,ww = rgb.shape[0]//2,rgb.shape[1]//2
        heat = gaussian(ww,hh,(point[0]/2,point[1]/2) if point else None,self.sigma)
        normalized=np.stack([np.zeros((3,height,width),np.float32) if im is None else normalize(im) for im in images])
        if self.temporal_mode=="single":payload=torch.from_numpy(normalized[0])
        else:payload=dict(images=torch.from_numpy(normalized),frame_valid=torch.tensor(index["frame_valid"]),
                          time_offsets_sec=torch.tensor(index["time_offsets_sec"],dtype=torch.float32))
        return payload,torch.from_numpy(heat),torch.tensor(float(valid)),i

    def __getitem__(self,i):
        if isinstance(i, tuple):
            index, size = i
            return self.item(index, input_size=size)
        return self.item(i)
