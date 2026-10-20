"""Online joint training. Discrete tracking/ROI selection deliberately stops gradients.

Each shuffled training window starts a fresh rule tracker. The centre has seven
preceding frames of context; validation uses the full chronological video. Only
training may use jittered GT crops, which anneal to zero before early stopping.
"""
import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from ttnet.data_tools.geometry import letterbox, map_point, crop_transform, crop_image, crop_point
from ttnet.events.data import assemble_window, event_window_ids
from ttnet.evaluation.metrics import accept, decode_heatmap
from ttnet.scene.table import annotations, decode_table, table_loss
from ttnet.tracking.association import BeamTracker, appearance
from ttnet.tracking.candidates import candidates
from ttnet.training.dataset import normalize, gaussian


def teacher_probability(config, epoch):
    full=config.get('gt_crop_full_epochs',3)
    end=config.get('gt_crop_end_epoch',10)
    if not 0 <= full < end:raise ValueError('invalid GT crop annealing schedule')
    return 1. if epoch < full else max(0., (end-1-epoch)/(end-full))


def heat_target(row, transform, width, height, sigma, local=False):
    """Unknown is masked; known balls outside a crop mean empty *crop*."""
    known=row['label_status'] in ('positive','negative')
    point=None
    if known and row['label_status']=='positive' and row.get('xy') is not None:
        point=(crop_point if local else map_point)(*row['xy'],transform)
        if local and not (0 <= point[0] < width and 0 <= point[1] < height):point=None
    return gaussian(width//2,height//2,(point[0]/2,point[1]/2) if point else None,sigma),known


def _batched(model, images, masks, offsets, chunk, spatial=False):
    values=[]
    for start in range(0,len(images),chunk):
        args=(images[start:start+chunk],masks[start:start+chunk],offsets[start:start+chunk])
        def forward(x,m,t):return model(x,m,t,return_spatial=True) if spatial else model(x,m,t)
        values.append(checkpoint(forward,*args,use_reentrant=False))
    return tuple(torch.cat([v[k] for v in values]) for k in range(len(values[0])))


def joint_forward(system, corpus, sample):
    if not system.training or system.mode!='joint':raise ValueError('joint_forward is training-only')
    cfg=system.config
    if cfg.get('tracker_type')!='association':raise ValueError('scratch baseline uses the rule association tracker')
    vid=sample['video_id'];center=sample['frame_id']
    frames=corpus.videos[vid]['frames'];meta=corpus.videos[vid]['meta']
    ids=sorted({i for i in event_window_ids(corpus,vid,center,cfg) if i is not None})
    if ids[0]-1 in frames:ids=[ids[0]-1]+ids
    valid_ids=[i for i in ids if frames[i].get('frame_valid',True)]
    if not valid_ids:raise ValueError('empty training window')
    gw,gh=cfg['global_input'];lw,lh=cfg['local_input'];device=system.device
    resolution=dict(width=meta['width'],height=meta['height'])
    # Read/resize each overlapping source frame only once per window.
    needed={n for fid in valid_ids for n in (fid-1,fid,fid+1)
            if n in frames and frames[n].get('frame_valid',True)}
    rgb={i:corpus.image(vid,i) for i in needed}
    boxed={i:letterbox(im,gw,gh) for i,im in rgb.items()}
    normalized={i:normalize(v[0]) for i,v in boxed.items()}
    triples={fid:[i if i in needed else None for i in (fid-1,fid,fid+1)] for fid in valid_ids}
    x=torch.from_numpy(np.stack([np.stack([normalized[i] if i is not None else np.zeros((3,gh,gw),np.float32)
                                         for i in triples[fid]]) for fid in valid_ids])).to(device)
    masks=torch.tensor([[i is not None for i in triples[fid]] for fid in valid_ids],device=device)
    offsets=torch.tensor([[frames[i]['timestamp_sec']-frames[fid]['timestamp_sec'] if i is not None else 0.
                           for i in triples[fid]] for fid in valid_ids],device=device,dtype=torch.float32)
    outputs=_batched(system.global_model,x,masks,offsets,cfg.get('joint_frame_batch',2),system.table_head is not None)
    heat,gdesc=outputs[:2]
    boxes=system.table_head(outputs[2]) if system.table_head is not None else None
    heat_np=heat.detach().float().cpu().numpy()
    boxes_np=boxes.detach().float().cpu().numpy() if boxes is not None else None
    gtargets=[];gknown=[];table_targets=[];table_indexes=[]
    if system.table_head is not None and not hasattr(system,'_joint_table_annotations'):
        system._joint_table_annotations=annotations('train')[0]
    tracker=BeamTracker(cfg.get('association'))
    # A shuffled clip is a distinct tracking sequence, never previous sample state.
    records={};local_inputs=[];local_masks=[];local_offsets=[];local_jobs=[]
    p_teacher=teacher_probability(cfg,getattr(system,'training_epoch',0))
    descriptor_indexes={fid:j for j,fid in enumerate(valid_ids)}
    for fid in ids:
        row=frames[fid];ts=row['timestamp_sec']
        if fid not in descriptor_indexes:
            tracker.update([],fid,ts)
            records[fid]=dict(prediction=None,local_valid=False,frame_valid=False,table=None)
            continue
        j=descriptor_indexes[fid];transform=boxed[fid][1]
        target,known=heat_target(row,transform,gw,gh,cfg['sigma'])
        gtargets.append(target);gknown.append(known)
        if boxes is not None:
            b=system._joint_table_annotations.get((vid.removeprefix('pp_'),fid)) if vid.startswith('pp_') else None
            if b is not None:
                table_indexes.append(j)
                table_targets.append(np.array([*map_point(*b[:2],transform),*map_point(*b[2:],transform)],np.float32)/[gw,gh,gw,gh])
        raw=candidates(heat_np[j:j+1],transform,resolution,count=tracker.config['top_k'])
        for point in raw:point.update(source='global',appearance=appearance(rgb[fid],point))
        xx,yy,cc=decode_heatmap(heat_np[j:j+1]);xx,yy=map_point(xx,yy,transform,inverse=True)
        g=accept(dict(x=xx,y=yy,confidence=cc,source='global'),cfg['localization_threshold'],resolution)
        search=tracker.predict(ts);selected=tracker.update(raw,fid,ts)
        centers=[]
        if search and 0<=search['x']<meta['width'] and 0<=search['y']<meta['height']:centers.append(search)
        proposal=next(iter(raw),None);side=cfg['local_crop_side_1080']*meta['height']/1080
        if proposal and proposal['confidence']>=tracker.config['min_confidence']:
            if not centers or math.hypot(proposal['x']-centers[0]['x'],proposal['y']-centers[0]['y'])>side*tracker.config['roi_merge_fraction']:
                centers.append(proposal)
        teacher=bool(row['label_status']=='positive' and row.get('xy') is not None and np.random.random()<p_teacher)
        if teacher:
            jitter=cfg.get('gt_crop_jitter_fraction',.2)*side
            tx,ty=np.asarray(row['xy'])+np.random.uniform(-jitter,jitter,2)
            centers=[dict(x=float(tx),y=float(ty))]
        jobs=[]
        for roi_center in centers[:2]:
            crop=crop_transform(roi_center['x'],roi_center['y'],meta['width'],meta['height'],lw,cfg['local_crop_side_1080'])
            local_inputs.append(np.stack([normalize(crop_image(rgb[i],crop)) if i is not None else np.zeros((3,lh,lw),np.float32)
                                          for i in triples[fid]]))
            local_masks.append(masks[j]);local_offsets.append(offsets[j])
            jobs.append(len(local_jobs));local_jobs.append((fid,crop,teacher))
        records[fid]=dict(frame_id=fid,prediction=accept(selected,cfg['localization_threshold'],resolution) if selected else None,
                          global_prediction=g,local_prediction=None,local_valid=False,crop_transform=None,
                          table=decode_table(boxes_np[j],transform,resolution) if boxes_np is not None else None,
                          associated_selection=selected,search_prediction=search,roi_transforms=[],_jobs=jobs)
    losses={}
    losses['global']=system.criterion(heat,torch.from_numpy(np.stack(gtargets)).to(device),torch.tensor(gknown,device=device))
    zero=gdesc.sum()*0
    losses['local']=zero
    losses['table']=table_loss(boxes[table_indexes].float(),torch.tensor(np.array(table_targets),device=device,dtype=torch.float32)) if table_indexes else zero
    local_results={};ldesc=None
    if local_inputs:
        lheat,ldesc=_batched(system.local_model,torch.from_numpy(np.stack(local_inputs)).to(device),
                            torch.stack(local_masks),torch.stack(local_offsets),cfg.get('joint_frame_batch',2))
        local_np=lheat.detach().float().cpu().numpy();targets=[];known=[]
        for k,(fid,crop,teacher) in enumerate(local_jobs):
            xx,yy,cc=decode_heatmap(local_np[k:k+1]);xx,yy=crop_point(xx,yy,crop,inverse=True)
            local_results[k]=accept(dict(x=xx,y=yy,confidence=cc,source='local'),cfg['localization_threshold'],resolution)
            t,m=heat_target(frames[fid],crop,lw,lh,cfg['sigma'],True);targets.append(t);known.append(m)
        losses['local']=system.criterion(lheat,torch.from_numpy(np.stack(targets)).to(device),torch.tensor(known,device=device))
    features={}
    for fid in ids:
        if fid not in descriptor_indexes:features[fid]=gdesc.new_zeros(256);continue
        rec=records[fid];reference=rec['associated_selection'] or rec['search_prediction']
        def cost(k):
            point=local_results[k]
            motion=min(tracker.config['motion_cap'],math.hypot(point['x']-reference['x'],point['y']-reference['y'])/tracker.config['motion_scale_px']) if reference else 0.
            return -math.log(max(1e-8,point['confidence']))+tracker.config['motion_weight']*motion
        valid=[k for k in rec['_jobs'] if local_results[k]['valid']]
        selected=min(valid,key=cost) if valid else None
        # GT assistance exposes the *visual descriptor*, never GT ball geometry.
        descriptor_index=selected if selected is not None else next((k for k in rec['_jobs'] if local_jobs[k][2]),None)
        rec['local_valid']=descriptor_index is not None
        if selected is not None:
            rec['local_prediction']=rec['prediction']=local_results[selected]
        rec['roi_transforms']=[local_jobs[k][1] for k in rec.pop('_jobs')]
        rec['crop_transform']=local_jobs[descriptor_index][1] if descriptor_index is not None else None
        features[fid]=torch.cat([gdesc[descriptor_indexes[fid]],ldesc[descriptor_index] if descriptor_index is not None else gdesc.new_zeros(128)])
    logits=system.head(*assemble_window(corpus,vid,center,features,records,device,cfg))
    return logits,losses
