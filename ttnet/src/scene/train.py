"""Extract frozen Global spatial features, train a table head, preserve all evidence."""
import argparse
import json
import random
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ttnet.common import DATA, PROJECT, save, sha, now, identity
from ttnet.evaluation.ball_corpus import BallCorpus
from ttnet.evaluation.inference import load_model
from ttnet.data_tools.geometry import letterbox,map_point
from ttnet.training.dataset import normalize
from ttnet.scene.table import TableHead,table_loss,box_iou,annotations,decode_table
from ttnet.events.train import atomic_checkpoint,rng_state,set_rng


def extract(model,checkpoint,root,device,batch=16):
    ident=dict(global_sha256=sha(checkpoint),splits={p:sha(DATA/f'splits/{p}.jsonl') for p in ('train','val')},
        labels={p:sha(DATA/f'labels/{p}.jsonl') for p in ('train','val')},
        code={p:sha(PROJECT/p) for p in ('ttnet/src/models/localizer.py','ttnet/src/scene/table.py','ttnet/src/scene/train.py')},input=[512,288],pool=[4,4],amp=True)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'identity.json').exists() and json.loads((root/'identity.json').read_text())!=ident:raise ValueError('table cache identity mismatch')
    save(root/'identity.json',ident)
    for part in ('train','val'):
        labels,audit=annotations(part);save(root/f'{part}_audit.json',audit)
        corpus=BallCorpus(DATA,part)
        for vid in corpus.split[part]:
            dest=root/f'{vid}.npz'
            if dest.exists():continue
            frames=corpus.videos[vid]['frames'];meta=corpus.videos[vid]['meta']
            ids=sorted(frames);features=[];targets=[];valid=[]
            with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                for start in range(0,len(ids),batch):
                    images=[];masks=[];offsets=[]
                    for fid in ids[start:start+batch]:
                        ns=[i if i in frames else None for i in (fid-1,fid,fid+1)]
                        images.append(np.stack([normalize(letterbox(corpus.image(vid,i),512,288)[0]) if i is not None else np.zeros((3,288,512),np.float32) for i in ns]))
                        masks.append([i is not None for i in ns]);offsets.append([frames[i]['timestamp_sec']-frames[fid]['timestamp_sec'] if i is not None else 0. for i in ns])
                        _,t=letterbox(corpus.image(vid,fid),512,288)
                        b=labels.get((vid,fid));valid.append(b is not None)
                        if b:
                            a=map_point(*b[:2],t);c=map_point(*b[2:],t)
                            targets.append(np.array([*a,*c],np.float32)/[512,288,512,288])
                        else:targets.append(np.zeros(4,np.float32))
                    _,_,spatial=model(torch.from_numpy(np.stack(images)).to(device),torch.tensor(masks,device=device),torch.tensor(offsets,device=device),return_spatial=True)
                    features.append(F.adaptive_avg_pool2d(spatial,(4,4)).float().cpu().numpy())
            tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,features=np.concatenate(features),targets=np.array(targets,dtype=np.float32),valid=np.array(valid),frame_ids=ids);tmp.replace(dest)
            print(f'table features {part} {vid}: {len(ids)}',flush=True)
    return ident


def load_part(root,part,device):
    corpus=BallCorpus(DATA,part);xs=[];ys=[];keys=[]
    for vid in corpus.split[part]:
        with np.load(root/f'{vid}.npz') as z:
            mask=z['valid'];xs.append(z['features'][mask]);ys.append(z['targets'][mask]);keys.extend((vid,int(i)) for i in z['frame_ids'][mask])
    return torch.tensor(np.concatenate(xs),device=device),torch.tensor(np.concatenate(ys),device=device),keys


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=60);p.add_argument('--resume',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1);device=torch.device(args.device)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    run=args.run
    if run.exists() and not args.resume:raise FileExistsError(run)
    run.mkdir(parents=True,exist_ok=True);save(run/'run.json',dict(status='extracting',started=now()))
    cfg=dict(seed=42,epochs=args.epochs,lr=3e-4,batch=128,patience=10,loss='L1+GIoU',global_checkpoint=str(args.checkpoint),global_sha256=sha(args.checkpoint),selection='val_mean_iou',missing='ignore',amp_features=True)
    save(run/'config.yaml',cfg)
    model,_=load_model(args.checkpoint,device);model.eval().requires_grad_(False)
    ident=extract(model,args.checkpoint,run/'features',device);del model
    x,y,_=load_part(run/'features','train',device);vx,vy,keys=load_part(run/'features','val',device)
    head=TableHead().to(device);opt=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    start=0;best=-1.;patience=0;history=[]
    if args.resume and (run/'last.pt').exists():
        s=torch.load(run/'last.pt',map_location='cpu',weights_only=False);head.load_state_dict(s['head']);opt.load_state_dict(s['optimizer']);start=s['epoch'];best=s['best'];patience=s['patience'];history=s['history'];set_rng(s['rng'])
    for epoch in range(start,args.epochs):
        head.train();losses=[];t=time.perf_counter()
        for ix in torch.randperm(len(x),device=device).split(cfg['batch']):
            opt.zero_grad();loss=table_loss(head(x[ix]),y[ix]);loss.backward();opt.step();losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode():pred=torch.cat([head(a) for a in vx.split(256)]);iou=box_iou(pred,vy);score=float(iou.mean())
        improved=score>best;best=max(best,score);patience=0 if improved else patience+1
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),val_mean_iou=score,elapsed_sec=time.perf_counter()-t))
        state=dict(head=head.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,best=best,patience=patience,history=history,rng=rng_state(),config=cfg,feature_identity=ident)
        atomic_checkpoint(run/'last.pt',state)
        if improved:atomic_checkpoint(run/'best.pt',state)
        save(run/'history.json',history);save(run/'run.json',dict(status='training',epoch=epoch+1,best_val_iou=best,updated=now()))
        print(history[-1],flush=True)
        if patience>=cfg['patience']:break
    state=torch.load(run/'best.pt',map_location=device,weights_only=False);head.load_state_dict(state['head']);head.eval()
    with torch.inference_mode():pred=torch.cat([head(a) for a in vx.split(256)]);iou=box_iou(pred,vy).cpu().numpy()
    per=[dict(video_id=k[0],frame_id=k[1],iou=float(v),predicted_box=p.tolist(),label_box=g.tolist()) for k,v,p,g in zip(keys,iou,pred.cpu(),vy.cpu())]
    save(run/'val_predictions.json',per);save(run/'metrics.json',dict(mean_iou=float(iou.mean()),iou50_recall=float((iou>=.5).mean()),iou75_recall=float((iou>=.75).mean()),frames=len(iou),epoch=state['epoch'],checkpoint_sha256=sha(run/'best.pt')))
    save(run/'run.json',dict(status='trained_pending_visualization',finished=now(),best_val_iou=best))

if __name__=='__main__':main()
