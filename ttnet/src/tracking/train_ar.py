"""Train GRU candidate association on PP sequences, with rollout and masked targets."""
import argparse
import json
import random
import subprocess
from pathlib import Path
import time
import numpy as np
import torch
from ttnet.common import DATA,PROJECT,save,sha,now,jsonl
from ttnet.evaluation.ball_corpus import BallCorpus
from ttnet.tracking.autoregressive import CandidateGRU,candidate_inputs,INPUT_DIM
from ttnet.events.train import atomic_checkpoint,rng_state,set_rng


def compatible_resume_config(previous,current):
    # Legacy runs predate these opt-in controls; their absent values mean disabled.
    previous=dict(previous)
    defaults=dict(null_loss_weight=1.,candidate_dropout_prob=0.,candidate_dropout_span=3,
                  selection_metric='val_loss',reference_checkpoint_sha256=None,minimum_positive_accuracy=0.)
    for key,value in defaults.items():previous.setdefault(key,value)
    return previous==current


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


def drop_correct_candidates(mask,targets,nearest,valid,probability=0.,span=3):
    """Drop all correct observations in sampled blocks; unknown frames stay unknown."""
    if not 0 <= probability <= 1 or span < 1:
        raise ValueError('invalid candidate dropout configuration')
    if not probability:
        return 0
    b,length=valid.shape
    blocks=torch.rand(b,(length+span-1)//span,device=mask.device)<probability
    selected=blocks.repeat_interleave(span,dim=1)[:,:length]&valid&targets[:,:,:-1].any(-1)
    removed=targets&selected[:,:,None]
    mask[removed]=False
    targets[selected]=False
    targets[:,:,-1]|=selected
    nearest[selected]=mask.shape[-1]-1
    return int(selected.sum())


def run_epoch(model,sequences,opt=None,teacher=0.,epoch=0,null_loss_weight=1.,
              candidate_dropout_prob=0.,candidate_dropout_span=3):
    device=next(model.parameters()).device;training=opt is not None;model.train(training)
    order=np.random.default_rng(42+epoch).permutation(len(sequences)) if training else np.arange(len(sequences))
    if not np.isfinite(null_loss_weight) or null_loss_weight<=0:raise ValueError('invalid null loss weight')
    total=weighted_total=weight_sum=0.;count=correct=0
    positive=positive_correct=null_targets=null_tp=null_fp=dropped=0
    # Videos are independent recurrent streams, padded only within a mini-batch.
    for start in range(0,len(order),8):
        seq=[sequences[i] for i in order[start:start+8]];b=len(seq);length=max(len(s['x']) for s in seq)
        x=torch.zeros(b,length,8,INPUT_DIM,device=device);mask=torch.zeros(b,length,8,dtype=torch.bool,device=device);mask[:,:,-1]=True
        targets=torch.zeros_like(mask);nearest=torch.full((b,length),7,dtype=torch.long,device=device);valid=torch.zeros(b,length,dtype=torch.bool,device=device)
        for i,s in enumerate(seq):
            n=len(s['x']);x[i,:n]=s['x'];mask[i,:n]=s['mask'];targets[i,:n]=s['target'];nearest[i,:n]=s['nearest'];valid[i,:n]=True
        if training:
            dropped+=drop_correct_candidates(mask,targets,nearest,valid,candidate_dropout_prob,candidate_dropout_span)
        h=torch.zeros(b,128,device=device)
        for begin in range(0,length,32):
            losses=[];weights=[]
            if training:opt.zero_grad()
            for t in range(begin,min(begin+32,length)):
                logp,emb=model.distribution(x[:,t],mask[:,t],h)
                supervised=targets[:,t].any(-1)&valid[:,t]
                # Gather supervised rows before logsumexp: avoid all -inf gradients on unknown rows.
                if supervised.any():
                    loss=-logp[supervised].masked_fill(~targets[supervised,t],-torch.inf).logsumexp(-1)
                    null=targets[supervised,t,-1]
                    w=torch.where(null,loss.new_tensor(null_loss_weight),loss.new_tensor(1.))
                    losses.append(loss*w);weights.append(w)
                    total+=float(loss.detach().sum());count+=int(supervised.sum())
                    weighted_total+=float((loss.detach()*w).sum());weight_sum+=float(w.sum())
                    predicted=logp.argmax(-1);p=predicted[supervised]
                    matched=targets[supervised,t].gather(1,p[:,None]).squeeze(1)
                    correct+=int(matched.sum());positive+=int((~null).sum())
                    positive_correct+=int((matched&~null).sum());null_targets+=int(null.sum())
                    null_tp+=int((null&(p==7)).sum());null_fp+=int((~null&(p==7)).sum())
                chosen=logp.argmax(-1)
                if training and teacher:
                    force=(torch.rand(b,device=device)<teacher)&supervised;chosen=torch.where(force,nearest[:,t],chosen)
                updated=model.advance(emb,h,chosen);h=torch.where(valid[:,t,None],updated,h)
            if training and losses:
                loss=torch.cat(losses).sum()/torch.cat(weights).sum();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            h=h.detach()
    null_fn=null_targets-null_tp
    return dict(loss=total/max(1,count),weighted_loss=weighted_total/max(1,weight_sum),
                candidate_accuracy=correct/max(1,count),supervised_frames=count,
                positive=positive,positive_correct=positive_correct,positive_accuracy=positive_correct/max(1,positive),
                null_targets=null_targets,null_tp=null_tp,null_fp=null_fp,null_fn=null_fn,
                null_precision=null_tp/max(1,null_tp+null_fp),null_recall=null_tp/max(1,null_targets),
                null_f1=2*null_tp/max(1,2*null_tp+null_fp+null_fn),augmented_null_frames=dropped)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--no-table',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--null-loss-weight',type=float,default=1.)
    p.add_argument('--candidate-dropout-prob',type=float,default=0.)
    p.add_argument('--candidate-dropout-span',type=int,default=3)
    p.add_argument('--selection-metric',choices=('val_loss','null_f1'),default='val_loss')
    p.add_argument('--reference-checkpoint',type=Path)
    args=p.parse_args()
    device=torch.device(args.device);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    if args.run.exists() and not args.resume:raise FileExistsError(args.run)
    args.run.mkdir(parents=True,exist_ok=True)
    cfg=dict(seed=42,hidden=128,epochs=30,patience=5,batch_videos=8,truncated_steps=32,lr=3e-4,teacher_epochs=5,rollout_ramp_epochs=5,match_px=10,use_table=not args.no_table,cache_identity=json.loads((args.cache/'identity.json').read_text()),
             null_loss_weight=args.null_loss_weight,candidate_dropout_prob=args.candidate_dropout_prob,
             candidate_dropout_span=args.candidate_dropout_span,selection_metric=args.selection_metric,
             reference_checkpoint_sha256=sha(args.reference_checkpoint) if args.reference_checkpoint else None)
    train=load_sequences(args.cache,'train',device,not args.no_table);val=load_sequences(args.cache,'val',device,not args.no_table)
    model=CandidateGRU().to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    minimum_positive_accuracy=0.
    if args.reference_checkpoint:
        reference=CandidateGRU().to(device)
        reference.load_state_dict(torch.load(args.reference_checkpoint,map_location=device,weights_only=False)['model'])
        with torch.no_grad():reference_metrics=run_epoch(reference,val)
        minimum_positive_accuracy=reference_metrics['positive_accuracy']-.01
        save(args.run/'reference_validation.json',reference_metrics)
        del reference
    elif args.selection_metric=='null_f1':raise ValueError('null F1 selection requires a reference checkpoint')
    cfg['minimum_positive_accuracy']=minimum_positive_accuracy
    if args.resume and not compatible_resume_config(json.loads((args.run/'config.yaml').read_text()),cfg):
        raise ValueError('AR resume config mismatch')
    save(args.run/'config.yaml',cfg)
    attempts=args.run/'attempts';attempts.mkdir(exist_ok=True)
    attempt=attempts/f'{len(list(attempts.iterdir()))+1:03d}';attempt.mkdir()
    save(attempt/'environment.json',dict(started=now(),command=__import__('sys').argv,
         torch=torch.__version__,cuda=torch.version.cuda,device=str(device),
         gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
         git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=PROJECT).strip(),
         git_status=subprocess.check_output(['git','status','--short'],text=True,cwd=PROJECT),
         train_code_sha256=sha(Path(__file__))))
    start=0;best=float('inf');patience=0;history=[]
    if args.resume and (args.run/'last.pt').exists():
        s=torch.load(args.run/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);start=s['epoch'];best=s['best'];patience=s['patience'];history=s['history'];set_rng(s['rng'])
    for epoch in range(start,cfg['epochs']):
        teacher=1. if epoch<5 else max(.5,1.-.1*(epoch-4));t=time.perf_counter()
        tr=run_epoch(model,train,opt,teacher,epoch,args.null_loss_weight,args.candidate_dropout_prob,args.candidate_dropout_span)
        with torch.no_grad():va=run_epoch(model,val)
        # Keep training through the teacher-forcing to rollout transition.
        selection=va['loss'] if args.selection_metric=='val_loss' else -va['null_f1']
        eligible=args.selection_metric=='val_loss' or va['positive_accuracy']>=minimum_positive_accuracy
        improved=eligible and selection<best
        if improved:best=selection;patience=0
        elif epoch>=10:patience+=1
        history.append(dict(epoch=epoch+1,teacher_probability=teacher,train=tr,val=va,
                            selection_eligible=eligible,elapsed_sec=time.perf_counter()-t))
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,best=best,patience=patience,history=history,config=cfg,rng=rng_state())
        atomic_checkpoint(args.run/'last.pt',state)
        if improved:atomic_checkpoint(args.run/'best.pt',state)
        save(args.run/'history.json',history);save(args.run/'run.json',dict(status='training',epoch=epoch+1,updated=now()));print(history[-1],flush=True)
        if patience>=cfg['patience']:break
    best_path=args.run/'best.pt'
    if best_path.exists():
        selected=torch.load(best_path,map_location='cpu',weights_only=False)
        metrics=selected['history'][-1]['val']
        if 'null_recall' not in metrics:
            legacy=CandidateGRU().to(device);legacy.load_state_dict(selected['model'])
            with torch.no_grad():metrics=run_epoch(legacy,val)
        passed=metrics['null_recall']>=.2 and metrics['positive_accuracy']>=minimum_positive_accuracy
    else:metrics=None;passed=False
    status='trained_pending_end_to_end_eval' if best_path.exists() else 'gate_failed'
    if args.selection_metric=='null_f1' and not passed:status='gate_failed'
    save(args.run/'run.json',dict(status=status,
         finished=now(),epochs_completed=len(history),best_selection=best if np.isfinite(best) else None,
         selection_metric=args.selection_metric,best_validation_loss=metrics['loss'] if metrics else None,
         checkpoint_sha256=sha(best_path) if best_path.exists() else None,
         selected_validation=metrics,end_to_end_gate_passed=passed))

if __name__=='__main__':main()
