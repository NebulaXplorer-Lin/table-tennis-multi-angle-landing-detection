"""Read-only table dataset adapter and reproducible, rotating training samples.

The CLI prepares indices only. It never constructs an optimizer or trains a model.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ttnet.common import PROJECT, identity, jsonl, save, save_jsonl
from ttnet.data_tools.frame_store import FrameStore
from ttnet.data_tools.geometry import letterbox, map_point
from ttnet.scene.table import annotation_identity, annotations


class TableCorpus:
    def __init__(self, root):
        self.root = Path(root)
        self.identity = annotation_identity(self.root)
        self.store = FrameStore(self.root)
        self.rows, self.groups = {}, {}
        for part in ('train', 'val'):
            annotations(part, self.root)  # Validate every row, including unknowns.
            self.rows[part] = jsonl(self.root/f'labels/{part}.jsonl')
            self.groups.update({r['video_id']: r['group'] for r in jsonl(self.root/f'splits/{part}.jsonl')})
        self.by_key = {(r['video_id'], r['frame_id']): r for rows in self.rows.values() for r in rows}
        self.frames = defaultdict(list)
        for vid, fid in self.by_key:
            self.frames[vid].append(fid)
        for ids in self.frames.values():
            ids.sort()

    def image(self, vid, fid):
        return self.store.get_rgb(vid, fid)

    def positives(self, part):
        return [r for r in self.rows[part] if r['table_status'] == 'positive']

    def audit(self):
        result = {}
        for part, rows in self.rows.items():
            positives = self.positives(part)
            result[part] = dict(videos=len({r['video_id'] for r in rows}), frames=len(rows),
                                positive=len(positives), unknown=len(rows)-len(positives),
                                thin=sum(is_thin(r) for r in positives))
        return result


def is_thin(row):
    b = row['table_bbox_xyxy']
    return (b[3]-b[1])/row['resolution']['height'] < .06


def sample_epoch(rows, groups, seed, epoch, count=4000, thin_weight=3.):
    """Group/video-balanced weighted sampling without replacement, rotating in time.

    A persistent seeded ordering within each video's thin/ordinary bucket ensures
    later epochs visit different frames. Buckets have at most 3x thin preference.
    """
    if not rows or count < 1 or not 1 <= thin_weight <= 3 or epoch < 0:
        raise ValueError('invalid table sampling parameters')
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        if row['split'] != 'train' or row['table_status'] != 'positive':
            raise ValueError('table sampler accepts positive training rows only')
        buckets[(groups[row['video_id']], row['video_id'])].append(i)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    queues = {}
    budget = min(count, len(rows))
    # Balance groups first, then videos; random cyclic offsets avoid a fixed prefix.
    for key in sorted(buckets):
        items = np.array(sorted(buckets[key], key=lambda i: rows[i]['frame_id']))
        fixed = np.random.default_rng(np.random.SeedSequence([seed, int(identity(key)[:8], 16)]))
        weights = np.array([thin_weight if is_thin(rows[i]) else 1. for i in items])
        priority = -np.log(fixed.uniform(1e-10, 1., len(items)))/weights
        order = items[np.argsort(priority)]
        offset = epoch*max(1, int(np.ceil(budget/len(buckets)))) % len(order)
        queues[key] = list(np.roll(order, -offset))
    by_group = defaultdict(list)
    for key in queues:
        by_group[key[0]].append(key)
    selected = []
    while len(selected) < budget:
        for group in rng.permutation(sorted(by_group)):
            active = [key for key in by_group[group] if queues[key]]
            if not active:
                continue
            key = active[int(rng.integers(len(active)))]
            selected.append(int(queues[key].pop(0)))
            if len(selected) == budget:
                break
    return [rows[i] for i in rng.permutation(selected)]


def color_jitter(rgb, rng, config):
    """RGB photometric changes only; hue uses the uint8 OpenCV 180-bin circle."""
    if rng.random()>=config['probability']:
        return rgb
    factors={name:rng.uniform(*config[name]) for name in ('brightness','contrast','saturation','hue')}
    image=rgb.astype(np.float32)/255.
    for name in rng.permutation(list(factors)):
        factor=factors[name]
        if name=='brightness':
            image*=factor
        elif name=='contrast':
            mean=(image*np.array([.299,.587,.114],np.float32)).sum(-1).mean()
            image=(image-mean)*factor+mean
        elif name=='saturation':
            gray=(image*np.array([.299,.587,.114],np.float32)).sum(-1,keepdims=True)
            image=gray+(image-gray)*factor
        else:
            hsv=cv2.cvtColor(np.clip(image*255,0,255).astype(np.uint8),cv2.COLOR_RGB2HSV)
            hsv[:,:,0]=((hsv[:,:,0].astype(np.float32)+factor*180)%180).astype(np.uint8)
            image=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB).astype(np.float32)/255.
        image=np.clip(image,0.,1.)
    return np.round(image*255).astype(np.uint8)


def transform_frame(rgb, bbox, size, rng=None, augmentation=None, padding_value=0):
    """One explicit affine transformation shared by image and all bbox corners."""
    h, w = rgb.shape[:2]
    matrix = np.eye(3, dtype=np.float64)
    geometry_fallback=False
    if rng is not None and augmentation:
        scale = rng.uniform(*augmentation['scale'])
        tx, ty = rng.uniform(-augmentation['translate'], augmentation['translate'], 2)*[w, h]
        matrix[:2] = [[scale, 0, (1-scale)*w/2+tx], [0, scale, (1-scale)*h/2+ty]]
        if rng.random() < augmentation['horizontal_flip']:
            matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]]) @ matrix
        if bbox is not None and augmentation.get('preserve_positive',False):
            x1,y1,x2,y2=bbox
            corners=np.array([[x1,y1,1],[x2,y1,1],[x1,y2,1],[x2,y2,1]])@matrix.T
            low=np.maximum(corners[:,:2].min(0),0);high=np.minimum(corners[:,:2].max(0),[w,h])
            if np.any(high-low<1):
                matrix=np.eye(3,dtype=np.float64);geometry_fallback=True
    # bbox coordinates describe pixel edges; OpenCV warps pixel centres.
    pixel_matrix = matrix.copy()
    pixel_matrix[:2,2] += matrix[:2,:2] @ np.array([.5,.5])-.5
    warped = cv2.warpAffine(rgb, pixel_matrix[:2], (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    boxed, transform = letterbox(warped, size[0], size[1],padding_value=padding_value)
    if augmentation and augmentation.get('preserve_positive',False):
        transform['augmentation_geometry_fallback']=geometry_fallback
    if rng is not None and augmentation and augmentation.get('color'):
        left,top=transform['pad_left'],transform['pad_top'];rw,rh=transform['resized_width'],transform['resized_height']
        boxed[top:top+rh,left:left+rw]=color_jitter(boxed[top:top+rh,left:left+rw],rng,augmentation['color'])
    target = None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        corners = np.array([[x1,y1,1], [x2,y1,1], [x1,y2,1], [x2,y2,1]]) @ matrix.T
        low, high = corners[:,:2].min(0), corners[:,:2].max(0)
        low, high = np.maximum(low, 0), np.minimum(high, [w,h])
        if np.any(high-low < 1):
            raise ValueError('augmentation produced a degenerate table')
        target = np.array([*map_point(*low, transform), *map_point(*high, transform)], np.float32)
        target /= [size[0],size[1],size[0],size[1]]
    return boxed, target, transform, matrix


class TableDataset(Dataset):
    def __init__(self, corpus, rows, config, epoch=0, training=False):
        self.corpus, self.rows, self.config = corpus, rows, config
        self.epoch, self.training = epoch, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        vid, fid = r['video_id'], r['frame_id']
        shared = self.config['source'] == 'shared_global'
        ids = [fid-1, fid, fid+1] if shared else [fid]
        size = self.config['input']
        rng_seed = [self.config['seed'], self.epoch, int(identity([vid,fid])[:8],16)]
        if self.config.get('augment_per_draw',False):
            rng_seed.append(index)
        images, masks, offsets = [], [], []
        center = self.corpus.by_key[(vid,fid)]
        target, transform = None, None
        for i in ids:
            row = self.corpus.by_key.get((vid,i))
            masks.append(row is not None)
            offsets.append(0. if row is None else row['timestamp_sec']-center['timestamp_sec'])
            if row is None:
                images.append(np.zeros((3,size[1],size[0]),np.float32))
                continue
            rng = np.random.default_rng(np.random.SeedSequence(rng_seed)) if self.training else None
            image, b, t, _ = transform_frame(self.corpus.image(vid,i), r['table_bbox_xyxy'], size,
                                             rng, self.config['augmentation'],padding_value=114 if self.config['source']=='independent_yolo' else 0)
            # Ball localizers and YOLO have different normalization contracts.
            if shared:
                from ttnet.training.dataset import normalize
                images.append(normalize(image))
            else:
                images.append(image.transpose(2,0,1).astype(np.float32)/255.)
            if i == fid:
                target, transform = b, t
        return dict(images=torch.from_numpy(np.stack(images) if shared else images[0]),
                    target=torch.tensor(target if target is not None else [0.]*4),
                    valid=r['table_status']=='positive', frame_valid=torch.tensor(masks),
                    offsets=torch.tensor(offsets,dtype=torch.float32), row=r, transform=transform)


def collate_table(items):
    return {k: torch.stack([i[k] for i in items]) if k in ('images','target','frame_valid','offsets')
            else [i[k] for i in items] for k in items[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=PROJECT/'data/datasets/pp_table_reannotated_v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=80)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    corpus = TableCorpus(args.dataset)
    args.output.mkdir(parents=True)
    manifests, seen = [], set()
    for epoch in range(args.epochs):
        rows = sample_epoch(corpus.positives('train'),corpus.groups,42,epoch)
        keys = [dict(video_id=r['video_id'],frame_id=r['frame_id']) for r in rows]
        seen.update((r['video_id'],r['frame_id']) for r in rows)
        save_jsonl(args.output/f'epoch_{epoch+1:03d}.jsonl',keys)
        manifests.append(dict(epoch=epoch+1,count=len(rows),thin=sum(is_thin(r) for r in rows),
                              sample_sha256=identity(keys),cumulative_unique=len(seen)))
    save(args.output/'manifest.json',dict(status='prepared_no_training',table_annotations=corpus.identity,
                                         stats=corpus.audit(),epochs=manifests))
    print(json.dumps(corpus.audit(),ensure_ascii=False))


if __name__ == '__main__':
    main()
