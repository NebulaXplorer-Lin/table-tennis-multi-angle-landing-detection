"""Train GRU candidate association on PP sequences, with rollout and masked targets."""
import argparse
import json
import random
from pathlib import Path
import time
import numpy as np
import torch
from ttnet.common import DATA,save,sha,now,jsonl
from ttnet.evaluation.ball_corpus import BallCorpus
from ttnet.tracking.autoregressive import CandidateGRU,candidate_inputs,INPUT_DIM
from ttnet.events.train import atomic_checkpoint,rng_state,set_rng


def load_sequences(cache,part,device,use_table):
    corpus=BallCorpus(DATA,part);result=[]
    for vid in corpus.split[part]:
        with np.load(cache/f'{vid}.npz') as z:features={int(i):f for i,f in zip(z['frame_ids'],z['features'])}
        records={r['frame_id']:r for r in jsonl(cache/f'{vid}.jsonl')};meta=corpus.videos[vid]['meta'];n=meta['n_frames']
        xs=np.zeros((n,8,INPUT_DIM),np.float32);mask=np.zeros((n,8),bool);target=np.zeros((n,8),bool);nearest=np.full(n,7,np.int64);positions=np.zeros((n,8,2),np.float32)
        mask[:,-1]=True
        for fid,row in enumerate(corpus.rows[vid]):
            rec=records[fid];raw=rec['raw_observations'];cand=[dict(p) for p in raw['global_candidates'] if p.get('confidence',0)>=.1][:5]
            cand += [dict(p) for p in raw['local'] if p.get('valid')][:2]
            for p in cand:p.setdefault('appearance',[0.,0.,0.])
            dt=row['timestamp_sec']-corpus.rows[vid][fid-1]['timestamp_sec'] if fid else 0.
            xs[fid,:len(cand)]=candidate_inputs(cand,features[fid][:128],rec.get('table'),meta,dt,use_table);mask[fid,:len(cand)]=True
            if cand:positions[fid,:len(cand)]=[[p['x'],p['y']] for p in cand]
            if row['label_status']=='positive':
                distances=np.linalg.norm(positions[fid,:len(cand)]-np.array(row['ball_centers'][0]),axis=-1)
                target[fid,:len(cand)]=distances<=10.
                if target[fid].any():nearest[fid]=int(distances.argmin())
                else:target[fid,-1]=True
            elif row['label_status']=='negative':target[fid,-1]=True
        result.append(dict(video=vid,x=torch.from_numpy(xs).to(device),mask=torch.from_numpy(mask).to(device),target=torch.from_numpy(target).to(device),nearest=torch.from_numpy(nearest).to(device)))
    return result


def run_epoch(model,sequences,opt=None,teacher=0.,epoch=0):
    device=next(model.parameters()).device;training=opt is not None;model.train(training)
    order=np.random.default_rng(42+epoch).permutation(len(sequences)) if training else np.arange(len(sequences))
    total=0.;count=correct=0
    # Videos are independent recurrent streams, padded only within a mini-batch.
    for start in range(0,len(order),8):
        seq=[sequences[i] for i in order[start:start+8]];b=len(seq);length=max(len(s['x']) for s in seq)
        x=torch.zeros(b,length,8,INPUT_DIM,device=device);mask=torch.zeros(b,length,8,dtype=torch.bool,device=device);mask[:,:,-1]=True
        targets=torch.zeros_like(mask);nearest=torch.full((b,length),7,dtype=torch.long,device=device);valid=torch.zeros(b,length,dtype=torch.bool,device=device)
        for i,s in enumerate(seq):
            n=len(s['x']);x[i,:n]=s['x'];mask[i,:n]=s['mask'];targets[i,:n]=s['target'];nearest[i,:n]=s['nearest'];valid[i,:n]=True
        h=torch.zeros(b,128,device=device)
        for begin in range(0,length,32):
            losses=[]
            if training:opt.zero_grad()
            for t in range(begin,min(begin+32,length)):
                logp,emb=model.distribution(x[:,t],mask[:,t],h)
                supervised=targets[:,t].any(-1)&valid[:,t]
                # Gather supervised rows before logsumexp: avoid all -inf gradients on unknown rows.
                if supervised.any():
                    loss=-logp[supervised].masked_fill(~targets[supervised,t],-torch.inf).logsumexp(-1)
                    losses.append(loss);total+=float(loss.detach().sum());count+=int(supervised.sum())
                    predicted=logp.argmax(-1);correct+=int(targets[supervised,t].gather(1,predicted[supervised,None]).sum())
                chosen=logp.argmax(-1)
                if training and teacher:
                    force=(torch.rand(b,device=device)<teacher)&supervised;chosen=torch.where(force,nearest[:,t],chosen)
                updated=model.advance(emb,h,chosen);h=torch.where(valid[:,t,None],updated,h)
            if training and losses:
                loss=torch.cat(losses).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            h=h.detach()
    return dict(loss=total/max(1,count),candidate_accuracy=correct/max(1,count),supervised_frames=count)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--no-table',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    device=torch.device(args.device);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    if args.run.exists() and not args.resume:raise FileExistsError(args.run)
    args.run.mkdir(parents=True,exist_ok=True)
    cfg=dict(seed=42,hidden=128,epochs=30,patience=5,batch_videos=8,truncated_steps=32,lr=3e-4,teacher_epochs=5,rollout_ramp_epochs=5,match_px=10,use_table=not args.no_table,cache_identity=json.loads((args.cache/'identity.json').read_text()))
    save(args.run/'config.yaml',cfg)
    train=load_sequences(args.cache,'train',device,not args.no_table);val=load_sequences(args.cache,'val',device,not args.no_table)
    model=CandidateGRU().to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    start=0;best=float('inf');patience=0;history=[]
    if args.resume and (args.run/'last.pt').exists():
        s=torch.load(args.run/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);start=s['epoch'];best=s['best'];patience=s['patience'];history=s['history'];set_rng(s['rng'])
    for epoch in range(start,cfg['epochs']):
        teacher=1. if epoch<5 else max(.5,1.-.1*(epoch-4));t=time.perf_counter()
        tr=run_epoch(model,train,opt,teacher,epoch)
        with torch.no_grad():va=run_epoch(model,val)
        # Keep training through the teacher-forcing to rollout transition.
        improved=va['loss']<best
        if improved:best=va['loss'];patience=0
        elif epoch>=10:patience+=1
        history.append(dict(epoch=epoch+1,teacher_probability=teacher,train=tr,val=va,elapsed_sec=time.perf_counter()-t))
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,best=best,patience=patience,history=history,config=cfg,rng=rng_state())
        atomic_checkpoint(args.run/'last.pt',state)
        if improved:atomic_checkpoint(args.run/'best.pt',state)
        save(args.run/'history.json',history);save(args.run/'run.json',dict(status='training',epoch=epoch+1,updated=now()));print(history[-1],flush=True)
        if patience>=cfg['patience']:break
    save(args.run/'run.json',dict(status='trained_pending_end_to_end_eval',finished=now(),best_validation_loss=best,checkpoint_sha256=sha(args.run/'best.pt')))

if __name__=='__main__':main()
