"""Differentiable visual descriptors with explicit detached hard crop/trajectory."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ttnet.common import CACHE, SRC, identity, jsonl, save, save_jsonl, sha
from ttnet.data_tools.geometry import crop_image, crop_point, crop_transform, letterbox, map_point
from ttnet.training.dataset import gaussian, normalize
from ttnet.evaluation.inference import load_model
from ttnet.evaluation.metrics import accept, decode_heatmap
from ttnet.models.event import EventHead
from ttnet.models.localizer import BallLocalizer, LocalizationLoss
from ttnet.models.build import build_localizer, build_event_head


class EventSystem(nn.Module):
    def __init__(self, config, device, initialize=True):
        super().__init__()
        self.config=config
        if initialize:
            self.global_model,ginfo=load_model(Path(config['global_checkpoint']),device)
            self.local_model,linfo=load_model(Path(config['local_checkpoint']),device)
            if linfo['global_checkpoint_sha256']!=sha(config['global_checkpoint']):
                raise ValueError('global/local dependency mismatch')
            for info in (ginfo,linfo):
                if info['config']['temporal_mode']!='fixed3':raise ValueError('event25 requires fixed3 locators')
        else:
            # Full event checkpoints contain both locators, including joint updates.
            # Inference must not depend on the original pretraining files remaining.
            self.global_model=build_localizer(dict(config,temporal_mode='fixed3')).to(device)
            self.local_model=build_localizer(dict(config,temporal_mode='fixed3')).to(device)
        self.head=build_event_head(config).to(device)
        self.criterion=LocalizationLoss()
        self.mode='frozen'
        self.reset_tracking()

    def reset_tracking(self, video_id=None):
        from ttnet.tracking.association import BeamTracker
        self._tracking_video = video_id
        self._tracker = BeamTracker(self.config.get('association')) if self.config.get('tracker_type') == 'association' else None

    @property
    def device(self):return next(self.parameters()).device

    def configure(self, mode, training):
        if mode not in ('frozen','joint'):raise ValueError(mode)
        self.mode=mode
        self.train(training)
        for model in (self.global_model,self.local_model):
            model.requires_grad_(mode=='joint')
            model.train(training and mode=='joint')

    def _run(self, model, images, valid, offsets):
        if self.training and self.mode=='joint':
            return checkpoint(model,images,valid,offsets,use_reentrant=False)
        return model(images,valid,offsets)

    def forward_frames(self, corpus, vid, ids):
        """Return differentiable per-frame features; decoded geometry is deliberately detached."""
        if self.config.get('tracker_type','independent') == 'association':
            from ttnet.tracking.runtime import associated_frames
            return associated_frames(self,corpus,vid,ids)
        frames=corpus.videos[vid]['frames']; meta=corpus.videos[vid]['meta'];cfg=self.config
        descriptors={}; records={}; losses={'global':[], 'local':[]}
        resolution=dict(width=meta['width'],height=meta['height'])
        zero=next(self.parameters()).new_zeros(())
        for fid in sorted(set(ids)):
            row=frames[fid]
            if not row.get('frame_valid',True):
                descriptors[fid]=zero.new_zeros(256)
                records[fid]=dict(frame_id=fid,prediction=None,global_prediction=None,local_prediction=None,
                                  local_valid=False,crop_transform=None,frame_valid=False,global_candidates=[],roi_transforms=[])
                continue
            neighbors=[i if i in frames and frames[i].get('frame_valid',True) and 0<=i<meta['n_frames'] else None for i in (fid-1,fid,fid+1)]
            valid=torch.tensor([[i is not None for i in neighbors]],device=self.device)
            offsets=torch.tensor([[frames[i]['timestamp_sec']-row['timestamp_sec'] if i is not None else 0. for i in neighbors]],
                                 device=self.device,dtype=torch.float32)
            rgb=[corpus.image(vid,i) if i is not None else None for i in neighbors]
            gw,gh=cfg['global_input'];lw,lh=cfg['local_input']
            global_images=[];transform=None
            for im in rgb:
                image, t=letterbox(im,gw,gh) if im is not None else (np.zeros((gh,gw,3),np.uint8),None)
                if t is not None:transform=t
                global_images.append(normalize(image))
            gx=torch.from_numpy(np.stack(global_images)).unsqueeze(0).to(self.device)
            heat,gdesc=self._run(self.global_model,gx,valid,offsets)
            x,y,conf=decode_heatmap(heat.detach().float().cpu().numpy())
            x,y=map_point(x,y,transform,inverse=True)
            g=accept(dict(x=x,y=y,confidence=conf,source='global'),cfg['localization_threshold'],resolution)
            crop=None; l=None; ldesc=gdesc.new_zeros((1,128))
            if g['valid']:
                crop=crop_transform(x,y,meta['width'],meta['height'],lw,cfg['local_crop_side_1080'])
                lx=torch.from_numpy(np.stack([normalize(crop_image(im,crop)) if im is not None else np.zeros((3,lh,lw),np.float32)
                                               for im in rgb])).unsqueeze(0).to(self.device)
                lheat,raw_ldesc=self._run(self.local_model,lx,valid,offsets)
                xx,yy,cc=decode_heatmap(lheat.detach().float().cpu().numpy())
                xx,yy=crop_point(xx,yy,crop,inverse=True)
                l=accept(dict(x=xx,y=yy,confidence=cc,source='local'),cfg['localization_threshold'],resolution)
                if l['valid']:ldesc=raw_ldesc
                if row['label_status'] in ('positive','negative'):
                    point=crop_point(*row['xy'],crop) if row['xy'] else None
                    if point and not (0<=point[0]<lw and 0<=point[1]<lh):point=None
                    target=gaussian(lw//2,lh//2,(point[0]/2,point[1]/2) if point else None,cfg['sigma'])
                    losses['local'].append(self.criterion(lheat,torch.from_numpy(target).unsqueeze(0).to(self.device),zero.new_ones(1)))
            if row['label_status'] in ('positive','negative'):
                point=map_point(*row['xy'],transform) if row['xy'] else None
                target=gaussian(gw//2,gh//2,(point[0]/2,point[1]/2) if point else None,cfg['sigma'])
                losses['global'].append(self.criterion(heat,torch.from_numpy(target).unsqueeze(0).to(self.device),zero.new_ones(1)))
            prediction=l if l and l['valid'] else g
            descriptors[fid]=torch.cat([gdesc,ldesc],-1)[0]
            records[fid]=dict(frame_id=fid,prediction=prediction,global_prediction=g,local_prediction=l,
                              local_valid=bool(l and l['valid']),crop_transform=crop)
            if cfg.get('candidate_diagnostics'):
                from ttnet.tracking.candidates import candidates
                records[fid]['global_candidates']=candidates(heat.detach().float().cpu().numpy(),transform,resolution)
                records[fid]['roi_transforms']=[crop] if crop else []
        loss={k:torch.stack(values).mean() if values else zero for k,values in losses.items()}
        return descriptors,records,loss


def cache_identity(system,corpus):
    return dict(schema=2,dataset_id=corpus.identity,
                tracker=dict(type=system.config.get('tracker_type','independent'),
                             parameters=system.config.get('association',{}) if system.config.get('tracker_type')=='association' else {}),global_sha256=sha(system.config['global_checkpoint']),
                local_sha256=sha(system.config['local_checkpoint']),
                config={k:system.config[k] for k in ('global_input','local_input','local_crop_side_1080','localization_threshold','amp')},
                code_sha256=sha(__file__),
                implementation={name:sha(SRC/name) for name in ('models/localizer.py','models/build.py','tracking/independent.py','tracking/association.py','tracking/candidates.py','tracking/runtime.py',
                    'training/dataset.py','data_tools/geometry.py','data_tools/frame_store.py','events/context.py','events/batched_features.py')},
                extraction_batch=system.config.get('feature_batch',16),
                sampling=dict(target_fps=system.config.get('target_fps',25),temporal_mode='fixed3',association_context='stream_source_with_existing_npy_precedence'))


def feature_cache_path(system,corpus):
    return CACHE/'features'/identity(cache_identity(system,corpus))[:20]


def build_cache(system,corpus,dest,parts=('train','val')):
    dest=Path(dest); ident=cache_identity(system,corpus)
    if (dest/'identity.json').exists() and json.loads((dest/'identity.json').read_text())!=ident:
        raise ValueError('stale feature cache; use another cache directory')
    save(dest/'identity.json',ident)
    system.configure('frozen',False)
    for part in parts:
        for vid in corpus.split[part]:
            path=dest/f'{vid}.npz';records_path=dest/f'{vid}.jsonl'
            system.reset_tracking(vid)
            if path.exists() and records_path.exists():continue
            arrays=[];records=[];ids=sorted(corpus.videos[vid]['frames'])
            with torch.inference_mode(),torch.autocast(device_type=system.device.type,enabled=system.config['amp'] and system.device.type=='cuda'):
                if system.config.get('tracker_type')=='association' and ids!=list(range(len(ids))):
                    from ttnet.events.context import stream_context_features
                    for fid,feature,record in stream_context_features(system,corpus,vid,ids):
                        arrays.append(feature.float().cpu().numpy());records.append(record)
                else:
                    size=system.config.get('feature_batch',16)
                    for start in range(0,len(ids),size):
                        batch=ids[start:start+size]
                        if system.config.get('tracker_type','independent')=='independent':
                            from ttnet.events.batched_features import independent_features
                            d,r,_=independent_features(system,corpus,vid,batch)
                        else:d,r,_=system.forward_frames(corpus,vid,batch)
                        arrays.extend(d[i].float().cpu().numpy() for i in batch)
                        records.extend(r[i] for i in batch)
                        if start%512==0:print(f'cache {vid} {start}/{len(ids)}',flush=True)
            # NPZ is committed before its record marker; interrupted writes are rebuilt.
            temporary=path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,frame_ids=np.array(ids),features=np.stack(arrays))
            temporary.replace(path)
            save_jsonl(records_path,records)
            print(f'cached {vid}: {len(ids)}',flush=True)


def load_cache(corpus,dest,parts=('train','val'),expected=None):
    if expected is not None and json.loads((Path(dest)/'identity.json').read_text())!=expected:
        raise ValueError('feature cache identity mismatch')
    result={}
    for part in parts:
        for vid in corpus.split[part]:
            with np.load(Path(dest)/f'{vid}.npz') as a:
                features={int(i):torch.from_numpy(f.copy()) for i,f in zip(a['frame_ids'],a['features'])}
            result[vid]=(features,{r['frame_id']:r for r in jsonl(Path(dest)/f'{vid}.jsonl')})
    return result
