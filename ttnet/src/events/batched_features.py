"""Batch the frozen independent locator while keeping its decoding contract."""
import numpy as np
import torch

from ttnet.data_tools.geometry import letterbox,crop_image,crop_transform,map_point,crop_point
from ttnet.evaluation.metrics import accept,decode_heatmap
from ttnet.training.dataset import normalize


def independent_features(system,corpus,vid,ids):
    if system.mode!='frozen' or system.config.get('tracker_type','independent')!='independent':
        raise ValueError('batched feature extraction requires frozen independent locators')
    cfg=system.config;video=corpus.videos[vid];frames=video['frames'];meta=video['meta']
    resolution=dict(width=meta['width'],height=meta['height']);gw,gh=cfg['global_input'];lw,lh=cfg['local_input']
    ids=sorted(set(ids));selected=[];descriptors={};records={};images=[];valids=[];offsets=[];triples=[];transforms=[]
    zero=next(system.parameters()).new_zeros(())
    for fid in ids:
        row=frames[fid]
        if not row.get('frame_valid',True):
            descriptors[fid]=zero.new_zeros(256);records[fid]=dict(frame_id=fid,prediction=None,global_prediction=None,local_prediction=None,
                local_valid=False,crop_transform=None,frame_valid=False,global_candidates=[],roi_transforms=[])
            continue
        neighbors=[i if i in frames and frames[i].get('frame_valid',True) and 0<=i<meta['n_frames'] else None for i in (fid-1,fid,fid+1)]
        rgb=[corpus.image(vid,i) if i is not None else None for i in neighbors]
        boxed=[letterbox(im,gw,gh) if im is not None else (np.zeros((gh,gw,3),np.uint8),None) for im in rgb]
        selected.append(fid);triples.append(rgb);transforms.append(boxed[1][1]);valids.append([i is not None for i in neighbors])
        offsets.append([frames[i]['timestamp_sec']-row['timestamp_sec'] if i is not None else 0. for i in neighbors])
        images.append(np.stack([normalize(im) for im,t in boxed]))
    if not selected:return descriptors,records,{'global':zero,'local':zero}
    gx=torch.from_numpy(np.stack(images)).to(system.device)
    valid=torch.tensor(valids,device=system.device);times=torch.tensor(offsets,device=system.device,dtype=torch.float32)
    heat,gdesc=system._run(system.global_model,gx,valid,times)
    heat=heat.float().cpu().numpy();local_indices=[];local_images=[];crops={};globals_={}
    for i,fid in enumerate(selected):
        x,y,conf=decode_heatmap(heat[i]);x,y=map_point(x,y,transforms[i],inverse=True)
        g=accept(dict(x=x,y=y,confidence=conf,source='global'),cfg['localization_threshold'],resolution);globals_[i]=g
        if g['valid']:
            crop=crop_transform(x,y,meta['width'],meta['height'],lw,cfg['local_crop_side_1080']);crops[i]=crop
            local_indices.append(i);local_images.append(np.stack([normalize(crop_image(im,crop)) if im is not None else np.zeros((3,lh,lw),np.float32) for im in triples[i]]))
    local={};local_descriptors={}
    if local_indices:
        ix=torch.tensor(local_indices,device=system.device)
        lx=torch.from_numpy(np.stack(local_images)).to(system.device)
        lheat,ldesc=system.local_model(lx,valid[ix],times[ix]);lheat=lheat.float().cpu().numpy()
        for j,i in enumerate(local_indices):
            x,y,conf=decode_heatmap(lheat[j]);x,y=crop_point(x,y,crops[i],inverse=True)
            local[i]=accept(dict(x=x,y=y,confidence=conf,source='local'),cfg['localization_threshold'],resolution)
            if local[i]['valid']:local_descriptors[i]=ldesc[j]
    for i,fid in enumerate(selected):
        g=globals_[i];l=local.get(i);ld=local_descriptors.get(i,gdesc[i].new_zeros(128))
        descriptors[fid]=torch.cat([gdesc[i],ld],-1)
        records[fid]=dict(frame_id=fid,prediction=l if l and l['valid'] else g,global_prediction=g,local_prediction=l,
                          local_valid=bool(l and l['valid']),crop_transform=crops.get(i),table=system.table_record(transforms[i],resolution,i))
    return descriptors,records,{'global':zero,'local':zero}
