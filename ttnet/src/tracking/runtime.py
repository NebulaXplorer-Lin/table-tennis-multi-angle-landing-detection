"""Frozen three-frame locator with causal association and at most two local crops."""
import math
import numpy as np
import torch

from ttnet.tracking.association import BeamTracker, appearance
from ttnet.tracking.candidates import candidates
from ttnet.data_tools.geometry import letterbox, map_point, crop_transform, crop_image, crop_point
from ttnet.training.dataset import normalize
from ttnet.evaluation.metrics import accept, decode_heatmap


def associated_frames(system, corpus, vid, ids):
    if system.mode != 'frozen' and system.training:
        raise ValueError('joint association training uses events.joint.joint_forward')
    cfg = system.config
    if system._tracking_video != vid:
        system.reset_tracking(vid)
    tracker = system._tracker
    frames = corpus.videos[vid]['frames']; meta = corpus.videos[vid]['meta']
    resolution = dict(width=meta['width'],height=meta['height'])
    gw,gh = cfg['global_input'];lw,lh = cfg['local_input']
    descriptors, records = {},{}
    for fid in sorted(set(ids)):
        if (tracker.last_frame is None and fid != 0) or (tracker.last_frame is not None and fid != tracker.last_frame+1):
            raise ValueError('association requires complete chronological video context; build the continuous cache first')
        row = frames[fid];ts = row['timestamp_sec']
        if not row.get('frame_valid',True):
            search=tracker.predict(ts)
            if hasattr(tracker,'context'):tracker.context(np.zeros(128),None,resolution)
            tracker.update([],fid,ts)
            descriptors[fid]=next(system.parameters()).new_zeros(256)
            records[fid]=dict(frame_id=fid,prediction=None,global_prediction=None,local_prediction=None,
                raw_observations=dict(global_candidates=[],local=[]),associated_selection=None,
                search_prediction=search,tracking_state=tracker.state(ts),local_valid=False,
                crop_transform=None,roi_transforms=[],global_candidates=[],frame_valid=False)
            continue
        neighbors = [i if i in frames and frames[i].get('frame_valid',True) and 0 <= i < meta['n_frames'] else None for i in (fid-1,fid,fid+1)]
        valid = torch.tensor([[i is not None for i in neighbors]],device=system.device)
        offsets = torch.tensor([[frames[i]['timestamp_sec']-ts if i is not None else 0. for i in neighbors]],device=system.device,dtype=torch.float32)
        rgb = [corpus.image(vid,i) if i is not None else None for i in neighbors]
        boxed = [letterbox(im,gw,gh) if im is not None else (np.zeros((gh,gw,3),np.uint8),None) for im in rgb]
        transform = boxed[1][1]
        gx = torch.from_numpy(np.stack([normalize(im) for im,_ in boxed])).unsqueeze(0).to(system.device)
        heat,gdesc = system._run(system.global_model,gx,valid,offsets)
        table = system.table_record(transform,resolution)
        hm = heat.detach().float().cpu().numpy()
        raw = candidates(hm,transform,resolution,count=tracker.config['top_k'])
        for p in raw:
            p.update(source='global',appearance=appearance(rgb[1],p))
        x,y,conf = decode_heatmap(hm);x,y = map_point(x,y,transform,inverse=True)
        g = accept(dict(x=x,y=y,confidence=conf,source='global'),cfg['localization_threshold'],resolution)
        search = tracker.predict(ts)
        if hasattr(tracker,'context'):tracker.context(gdesc[0].detach().float().cpu().numpy(),table,resolution)
        selected = tracker.update(raw,fid,ts)
        centers = []
        if search and 0 <= search['x'] < meta['width'] and 0 <= search['y'] < meta['height']:
            centers.append(search)
        # Global re-acquisition is never disabled, including an empty/missing beam.
        proposal = next(iter(raw),None)
        side = cfg['local_crop_side_1080']*meta['height']/1080
        if proposal and proposal['confidence'] >= tracker.config['min_confidence']:
            if not centers or math.hypot(proposal['x']-centers[0]['x'],proposal['y']-centers[0]['y']) > side*tracker.config['roi_merge_fraction']:
                centers.append(proposal)
        observations = [];crops = [];local_descriptors = []
        for center in centers[:2]:
            crop = crop_transform(center['x'],center['y'],meta['width'],meta['height'],lw,cfg['local_crop_side_1080'])
            lx = torch.from_numpy(np.stack([normalize(crop_image(im,crop)) if im is not None else np.zeros((3,lh,lw),np.float32) for im in rgb])).unsqueeze(0).to(system.device)
            lheat,ldesc = system._run(system.local_model,lx,valid,offsets)
            xx,yy,cc = decode_heatmap(lheat.detach().float().cpu().numpy());xx,yy = crop_point(xx,yy,crop,inverse=True)
            observations.append(accept(dict(x=xx,y=yy,confidence=cc,source='local',crop_transform=crop),cfg['localization_threshold'],resolution))
            observations[-1]['appearance']=appearance(rgb[1],observations[-1])
            observations[-1]['local_index']=len(crops)
            crops.append(crop);local_descriptors.append(ldesc)
        reference = selected or search
        def cost(i):
            p = observations[i]
            motion = min(tracker.config['motion_cap'],math.hypot(p['x']-reference['x'],p['y']-reference['y'])/tracker.config['motion_scale_px']) if reference else 0.
            return -math.log(max(1e-8,p['confidence']))+tracker.config['motion_weight']*motion
        valid_local = [i for i,p in enumerate(observations) if p['valid']]
        index = min(valid_local,key=cost) if valid_local else None
        local = observations[index] if index is not None else None
        observed = accept(selected,cfg['localization_threshold'],resolution) if selected else None
        prediction = local or observed
        if hasattr(tracker,'finalize'):
            chosen=tracker.finalize(observations)
            index=chosen.get('local_index') if chosen and chosen.get('source')=='local' else None
            local=observations[index] if index is not None else None
            prediction=accept(chosen,cfg['localization_threshold'],resolution) if chosen else None
        ldesc = local_descriptors[index] if index is not None else gdesc.new_zeros((1,128))
        descriptors[fid] = torch.cat([gdesc,ldesc],-1)[0]
        records[fid] = dict(frame_id=fid,prediction=prediction,global_prediction=g,local_prediction=local,
                            raw_observations=dict(global_candidates=raw,local=observations),
                            associated_selection=selected,search_prediction=search,tracking_state=tracker.state(ts),
                            local_valid=local is not None,crop_transform=crops[index] if index is not None else (crops[0] if crops else None),
                            roi_transforms=crops,global_candidates=raw,table=table)
    zero = next(system.parameters()).new_zeros(())
    return descriptors,records,{'global':zero,'local':zero}
