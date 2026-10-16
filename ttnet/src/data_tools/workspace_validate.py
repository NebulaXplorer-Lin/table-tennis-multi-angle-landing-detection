"""Real-image migration and checkpoint parity checks. Never train or rewrite a run."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch

from ttnet.common import PROJECT,DATA,EVENT_DATA,save,sha
from ttnet.data_tools.frame_store import FrameStore
from ttnet.events.data import EventCorpus,assemble_window
from ttnet.events.engine import EventSystem
from ttnet.evaluation.inference import load_model
from ttnet.evaluation.metrics import decode_heatmap
from ttnet.temporal import move_inputs
from ttnet.training.dataset import BallDataset
from ttnet.workspace import verify_protected


def original_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def close(left,right):
    np.testing.assert_allclose(left,right,atol=1e-6,rtol=1e-5)
    return float(np.max(np.abs(np.asarray(left)-np.asarray(right))))


def benchmark_reader(dataset,kind,count=64):
    if kind=='ball':
        old=BallDataset(PROJECT/'ttnet/data/datasets/ball_v2','val',all_frames=True)
        rows=old.rows
        legacy=FrameStore(old.dataset)
        read_old=lambda row:legacy.row_rgb(row)
    else:
        old=EventCorpus(PROJECT/'ttnet/data/datasets/event25_v3')
        rows=[dict(r,video_id=vid) for vid,video in old.videos.items() for r in video['frames'].values()]
        read_old=lambda row:old.frame_store.get_rgb(row['video_id'],row['frame_id'],path=str(Path(row['video_id'])/row['path']))
    old_store=legacy if kind=='ball' else old.frame_store
    legacy_label='legacy_npy_alias' if old_store.root is not None else 'legacy_image'
    selected=random.Random(42).sample(rows,min(count,len(rows)))
    store=FrameStore(dataset)
    times={}
    for name,reader in [(legacy_label,read_old),('npy_mmap',lambda row:store.get_rgb(row['video_id'],row['frame_id']))]:
        # Touch the entire image, not merely the array header or one pixel.
        start=time.perf_counter();checksum=0
        for row in selected:checksum+=int(reader(row).sum())
        elapsed=time.perf_counter()-start
        times[name]=dict(frames=len(selected),elapsed_sec=elapsed,frames_per_sec=len(selected)/elapsed,checksum=checksum)
    if times[legacy_label]['checksum']!=times['npy_mmap']['checksum']:raise ValueError('reader benchmark pixels differ')
    return dict(protocol='Seeded random frames; sum all RGB bytes; single process; OS cache uncontrolled; not model FPS',
                legacy_reader=legacy_label,old_images_removed=old_store.root is not None,**times)


def validate():
    torch.set_num_threads(4)
    batch=json.loads((PROJECT/'out/workspace/current.json').read_text())
    before=PROJECT/batch['archive']/'before/ttnet_src'
    old_localizer=original_module('workspace_before_localizer',before/'models/localizer.py')
    old_event=original_module('workspace_before_event',before/'models/event.py')
    global_path=PROJECT/'ttnet/output/20261014-103011_local_fixed3_ball/weights/global_best.pt'
    local_path=PROJECT/'ttnet/output/20261014-103011_local_fixed3_ball/weights/best.pt'
    result=dict(schema=1,backend='CPU FP32',atol=1e-6,rtol=1e-5,training_started=False,
                checkpoint_sha256={p.name:sha(p) for p in (global_path,local_path)})
    old=BallDataset(PROJECT/'ttnet/data/datasets/ball_v2','val',all_frames=True,temporal_mode='fixed3')
    new=BallDataset(DATA,'val',all_frames=True,temporal_mode='fixed3')
    if old.rows!=new.rows:raise ValueError('ball labels or ordering changed')
    choices=[];seen=set()
    for index,row in enumerate(new.rows):
        size=tuple(row['resolution'].values())
        if size not in seen:
            seen.add(size)
            video=[i for i,r in enumerate(new.rows) if r['video_id']==row['video_id']]
            choices.extend([video[0],video[len(video)//2],video[-1]])
    differences=[]
    for stage,path in [('global',global_path),('local',local_path)]:
        model,state=load_model(path,torch.device('cpu'))
        previous=old_localizer.BallLocalizer('fixed3').eval();previous.load_state_dict(state['model'])
        predictions={(r['video_id'],r['frame_id']):dict(x=r['ball_centers'][0][0],y=r['ball_centers'][0][1],valid=True)
                     for r in new.rows if r['label_status']=='positive'}
        ds_old=old if stage=='global' else BallDataset(old.dataset,'val',stage,predictions=predictions,all_frames=True,temporal_mode='fixed3')
        ds_new=new if stage=='global' else BallDataset(DATA,'val',stage,predictions=predictions,all_frames=True,temporal_mode='fixed3')
        for index in choices:
            x0,y0,v0,_=ds_old[index];x1,y1,v1,_=ds_new[index]
            for key in x0:np.testing.assert_array_equal(x0[key],x1[key])
            np.testing.assert_array_equal(y0,y1);np.testing.assert_array_equal(v0,v1)
            inputs={k:v.unsqueeze(0) for k,v in x1.items()}
            with torch.inference_mode():
                h0,f0=previous(*move_inputs(inputs,torch.device('cpu')))
                h1,f1=model(*move_inputs(inputs,torch.device('cpu')))
            differences.append(dict(stage=stage,video_id=new.rows[index]['video_id'],frame_id=new.rows[index]['frame_id'],
                         heatmap_max_error=close(h0.numpy(),h1.numpy()),feature_max_error=close(f0.numpy(),f1.numpy()),
                         decoded_max_error=close(decode_heatmap(h0.numpy()),decode_heatmap(h1.numpy()))))
    result['ball_parity']=differences
    event_path=PROJECT/'ttnet/runs/20261014-181231_event25_frozen/checkpoints/best.pt'
    state=torch.load(event_path,map_location='cpu',weights_only=False);cfg=state['config']
    system=EventSystem(cfg,torch.device('cpu'),initialize=False);system.load_state_dict(state['system']);system.configure('frozen',False)
    previous_head=old_event.EventHead().eval();previous_head.load_state_dict(system.head.state_dict())
    old_corpus=EventCorpus(PROJECT/'ttnet/data/datasets/event25_v3');new_corpus=EventCorpus(EVENT_DATA)
    if old_corpus.samples!=new_corpus.samples or old_corpus.split!=new_corpus.split:
        raise ValueError('event labels or split changed')
    sample=next(s for s in new_corpus.samples['val'] if s['original'] and s['group']=='bounce')
    vid=sample['video_id'];center=sample['frame_id']
    ids=[i for i in new_corpus.window_ids(vid,center) if i is not None]
    if ids[0]-1 in new_corpus.videos[vid]['frames']:ids=[ids[0]-1]+ids
    for fid in ids:
        for neighbor in (fid-1,fid,fid+1):
            if neighbor in new_corpus.videos[vid]['frames']:
                np.testing.assert_array_equal(old_corpus.image(vid,neighbor),new_corpus.image(vid,neighbor))
    with torch.inference_mode():
        d0,r0,_=system.forward_frames(old_corpus,vid,ids)
        d1,r1,_=system.forward_frames(new_corpus,vid,ids)
        w0=assemble_window(old_corpus,vid,center,d0,r0,torch.device('cpu'))
        w1=assemble_window(new_corpus,vid,center,d1,r1,torch.device('cpu'))
        for a,b in zip(w0,w1):np.testing.assert_array_equal(a,b)
        logits0=previous_head(*w0);logits1=system.head(*w1)
    if r0!=r1:raise ValueError('decoded geometry or ROI changed')
    result['event_parity']=dict(video_id=vid,center_frame_id=center,frames_checked=len(ids),
        descriptor_max_error=max(close(d0[i].numpy(),d1[i].numpy()) for i in ids),
        logits_max_error=close(logits0.numpy(),logits1.numpy()),checkpoint_sha256=sha(event_path))
    # Load paused optimization state on CPU, without taking a step or writing it.
    resumes=[]
    for run in ('20261014-001042_local_ball','20261014-181231_event25_joint'):
        path=PROJECT/'ttnet/runs'/run/'checkpoints/last.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if 'system' in checkpoint:
            module=EventSystem(checkpoint['config'],torch.device('cpu'),initialize=False)
            module.load_state_dict(checkpoint['system'])
            optimizer=torch.optim.AdamW([dict(params=list(module.head.parameters()),lr=cfg['lr']),
                dict(params=list(module.global_model.parameters())+list(module.local_model.parameters()),lr=cfg['locator_lr'])])
        else:
            module=old_localizer.BallLocalizer(checkpoint['config'].get('temporal_mode','single'))
            module.load_state_dict(checkpoint['model']);optimizer=torch.optim.AdamW(module.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        resumes.append(dict(run=run,epoch=checkpoint.get('epoch'),step=checkpoint.get('step'),
               optimizer_loaded=True,checkpoint_sha256=sha(path),run_status=json.loads((path.parent.parent/'run.json').read_text())['status']))
    result['paused_checkpoint_loads']=resumes
    result['reader_benchmarks']={kind:benchmark_reader(dataset,kind) for dataset,kind in [(DATA,'ball'),(EVENT_DATA,'event')]}
    result['peak_process_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    result['protected_artifacts_verified']=verify_protected(batch)
    result['passed']=True
    save(PROJECT/'out/workspace/validation.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':validate()
