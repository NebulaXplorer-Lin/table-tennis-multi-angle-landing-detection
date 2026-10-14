from __future__ import annotations

import argparse
from contextlib import redirect_stdout,redirect_stderr
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from ttnet.common import DATA,PROJECT,ROOT,identity,jsonl,now,save,save_jsonl,sha
from ttnet.evaluation.inference import load_model,predict,prediction_map
from ttnet.evaluation.metrics import aggregate,select_threshold
from ttnet.models.localizer import BallLocalizer,LocalizationLoss
from .dataset import BallDataset
from .scales import MultiScaleBatchSampler,input_size,training_scales


class Tee:
    def __init__(self,*files): self.files=files
    def write(self,value):
        for file in self.files: file.write(value);file.flush()
    def flush(self):
        for file in self.files: file.flush()


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed);cv2.setNumThreads(1)


def checkpoint(path,value):
    temp=path.with_suffix(".tmp")
    torch.save(value,temp);temp.replace(path)


def train(args,run,config,attempt):
    seed=config["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(config["torch_threads"]);cv2.setNumThreads(1)
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run authorized GPU training outside device-restricted sandbox")
    meta=json.loads((args.dataset/"metadata/dataset.json").read_text())
    if meta["status"]!="completed":
        raise RuntimeError("full dataset audit must complete before training")
    global_model=None;global_info=None;global_maps={};global_candidates=None
    if args.stage=="local":
        if not args.global_checkpoint: raise ValueError("local stage requires --global-checkpoint")
        global_model,global_info=load_model(args.global_checkpoint,device)
        if global_info["dataset_id"]!=meta["dataset_id"]: raise ValueError("global checkpoint dataset differs")
        for p in global_model.parameters(): p.requires_grad_(False)
        cache_dir=run/"crop_indexes";cache_dir.mkdir(exist_ok=True)
        gid=sha(args.global_checkpoint)
        for split in ["train","val"]:
            cache=cache_dir/f"{split}.jsonl"
            cache_identity=cache_dir/f"{split}_identity.json"
            expected=dict(dataset_id=meta["dataset_id"],global_sha256=gid,threshold=global_info["threshold"],
                          input_size=list(input_size(global_info.get("config"),"global")),precision="AMP" if config["amp"] else "FP32")
            if cache.exists() and cache_identity.exists() and json.loads(cache_identity.read_text())==expected:
                stored=jsonl(cache);rows=jsonl(args.dataset/"labels"/f"{split}.jsonl")
                if len(stored)!=len(rows) or any((s["video_id"],s["frame_id"])!=(r["video_id"],r["frame_id"]) for s,r in zip(stored,rows)):
                    raise RuntimeError("global prediction cache row mismatch")
                candidates=[s["candidate"] for s in stored]
            else:
                rows,candidates,_=predict(global_model,args.dataset,split,device,batch=config["micro_batch"],workers=config["workers"],amp=config["amp"],
                                          input_size=input_size(global_info.get("config"),"global"))
                save_jsonl(cache,[dict(video_id=r["video_id"],frame_id=r["frame_id"],candidate=p) for r,p in zip(rows,candidates)])
                save(cache_identity,expected)
            global_maps[split]=prediction_map(rows,candidates,global_info["threshold"])
            if split=="val": global_candidates=candidates
        print("Frozen global prediction/crop indexes saved",flush=True)
        global_model.cpu();del global_model
        if device.type=="cuda":torch.cuda.empty_cache()
    ds=BallDataset(args.dataset,"train",args.stage,augment=True,sigma=config["sigma"],predictions=global_maps.get("train"),
                   input_size=input_size(config,args.stage),crop_side_at_1080=config.get("local_crop_side_1080",256))
    if args.smoke:
        # Spread over videos while keeping a genuinely fixed small training set.
        indices=np.linspace(0,len(ds)-1,min(8,len(ds))).round().astype(int).tolist()
        ds.augment=False;ds.gt_crops=True
        loader=DataLoader(Subset(ds,indices),batch_size=min(config["micro_batch"],len(indices)),num_workers=0)
    else:
        scales=training_scales(config,args.stage)
        if len(scales)>1 or scales[0]!=ds.input_size:
            loader=DataLoader(ds,batch_sampler=MultiScaleBatchSampler(len(ds),config["micro_batch"],scales),
                              num_workers=config["workers"],pin_memory=device.type=="cuda",worker_init_fn=seed_worker)
        else:
            loader=DataLoader(ds,batch_size=config["micro_batch"],shuffle=True,num_workers=config["workers"],
                              pin_memory=device.type=="cuda",worker_init_fn=seed_worker)
    model=BallLocalizer().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    criterion=LocalizationLoss()
    scaler=torch.amp.GradScaler("cuda",enabled=config["amp"] and device.type=="cuda")
    start_epoch=0;best_key=(-1.,-1e12);patience=0;best_epoch=None;threshold=None
    if args.resume:
        resume=torch.load(run/"checkpoints/last.pt",map_location="cpu",weights_only=False)
        if resume["dataset_id"]!=meta["dataset_id"]:raise RuntimeError("resume data differs")
        expected_global=sha(args.global_checkpoint) if args.global_checkpoint else None
        if resume.get("global_checkpoint_sha256")!=expected_global:raise RuntimeError("resume frozen Global checkpoint differs")
        model.load_state_dict(resume["model"]);optimizer.load_state_dict(resume["optimizer"]);scaler.load_state_dict(resume["scaler"])
        start_epoch=resume["epoch"]+1;best_key=tuple(resume["best_key"]);patience=resume["patience"]
        best_epoch=resume["best_epoch"];threshold=resume["threshold"]
        torch.set_rng_state(resume["torch_rng"]);np.random.set_state(resume["numpy_rng"]);random.setstate(resume["python_rng"])
        if device.type=="cuda" and resume.get("cuda_rng") is not None:torch.cuda.set_rng_state_all(resume["cuda_rng"])
    if args.smoke:
        model.train();history=[]
        batches=list(loader)
        for step in range(100):
            x,y,valid,_=batches[step%len(batches)]
            optimizer.zero_grad(set_to_none=True)
            heat,_=model(x.to(device));loss=criterion(heat,y.to(device),valid.to(device));loss.backward();optimizer.step()
            history.append(float(loss.detach()))
        checkpoint(run/"checkpoints/last.pt",dict(model=model.state_dict(),stage=args.stage,dataset_id=meta["dataset_id"]))
        initial=float(np.mean(history[:10]));final=float(np.mean(history[-10:]))
        save(run/"evaluations/smoke.json",dict(initial_loss=initial,final_loss=final,losses=history,
                                              passed=final<initial*.5,steps=100,samples=len(indices)))
        if final>=initial*.5:raise RuntimeError(f"small sample failed to overfit: {initial} -> {final}")
        print(f"Smoke passed: {initial:.5f} -> {final:.5f}",flush=True)
        return dict(status="completed_smoke",dataset_id=meta["dataset_id"],samples=len(indices))
    epochs=config["epochs"]
    for epoch in range(start_epoch,epochs):
        ds.gt_crops=args.stage=="local" and epoch<config["gt_crop_epochs"]
        if args.stage=="local" and epoch==config["gt_crop_epochs"]:
            patience=0
            # Selection must now compare checkpoints exposed to deployment crops.
            best_key=(-1.,-1e12)
        model.train();optimizer.zero_grad(set_to_none=True)
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();total_loss=0.;total_samples=0
        accumulated=0
        for step,(x,y,valid,_) in enumerate(loader):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);valid=valid.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,enabled=config["amp"] and device.type=="cuda"):
                heat,_=model(x)
                loss=criterion(heat,y,valid)
            count=len(x)
            scaler.scale(loss*count/config["effective_batch"]).backward()
            accumulated+=count
            if accumulated>=config["effective_batch"] or step==len(loader)-1:
                if accumulated!=config["effective_batch"]:
                    scaler.unscale_(optimizer)
                    for p in model.parameters():
                        if p.grad is not None:p.grad.mul_(config["effective_batch"]/accumulated)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);accumulated=0
            total_loss+=float(loss.detach())*count;total_samples+=count
            if step%400==0:print(f"epoch {epoch+1}/{epochs} batch {step+1}/{len(loader)} loss={float(loss.detach()):.4f}",flush=True)
        rows,candidates,_=predict(model,args.dataset,"val",device,args.stage,global_maps.get("val"),
                                  batch=config["micro_batch"],workers=config["workers"],amp=config["amp"],
                                  input_size=input_size(config,args.stage),crop_side_at_1080=config.get("local_crop_side_1080",256))
        tolerance=config["global_best_tolerance_px"] if args.stage=="global" else config["local_best_tolerance_px"]
        selected=select_threshold(rows,candidates,tolerance,global_candidates,
                                   global_info["threshold"] if global_info else None)
        metrics=selected["metrics"];error=metrics["accepted_error_median_px"]
        key=(metrics["f1"],-(error if error is not None else 1e12))
        improved=key>best_key
        if improved:best_key=key;best_epoch=epoch;threshold=selected["threshold"];patience=0
        else:patience+=1
        record=dict(epoch=epoch+1,train_loss=total_loss/max(total_samples,1),learning_rate=optimizer.param_groups[0]["lr"],
                    crop_mode="gt_jitter" if ds.gt_crops else ("predicted_global" if args.stage=="local" else None),
                    threshold=selected["threshold"],val=metrics,elapsed_sec=time.perf_counter()-start,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else None)
        with (attempt/"epochs.jsonl").open("a") as f:f.write(json.dumps(record,allow_nan=False)+"\n")
        save(run/"evaluations/latest_threshold_sweep.json",selected["sweep"])
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                   stage=args.stage,epoch=epoch,best_epoch=best_epoch,best_key=best_key,patience=patience,
                   threshold=threshold,dataset_id=meta["dataset_id"],config=config,
                   global_checkpoint_sha256=sha(args.global_checkpoint) if args.global_checkpoint else None,
                   global_threshold=global_info["threshold"] if global_info else None,
                   torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else None)
        checkpoint(run/"checkpoints/last.pt",state)
        if improved:
            checkpoint(run/"checkpoints/best.pt",state)
            save(run/"evaluations/best_selection.json",dict(epoch=epoch+1,threshold=threshold,tolerance_px=tolerance,
                                                           metrics=metrics,sweep=selected["sweep"]))
        save(run/"run.json",dict(status="training",stage=args.stage,started=args.started,updated=now(),epoch=epoch+1,
                                  best_epoch=best_epoch+1,dataset_id=meta["dataset_id"],attempt=str(attempt.relative_to(run))))
        print(f"epoch {epoch+1}: loss={record['train_loss']:.5f} val F1@{tolerance}px={metrics['f1']:.4f} "
              f"threshold={selected['threshold']} time={record['elapsed_sec']:.1f}s patience={patience}",flush=True)
        if patience>=config["patience"] and not (args.stage=="local" and epoch<config["gt_crop_epochs"]):break
    return dict(status="trained_pending_eval",stage=args.stage,started=args.started,finished=now(),
                epochs_completed=epoch+1,best_epoch=best_epoch+1,threshold=threshold,dataset_id=meta["dataset_id"],
                best_sha256=sha(run/"checkpoints/best.pt"),last_sha256=sha(run/"checkpoints/last.pt"))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["global","local"],required=True)
    p.add_argument("--dataset",type=Path,help="defaults to current dataset; --resume reuses the run's original dataset")
    p.add_argument("--config",type=Path,default=ROOT/"configs/default.json")
    p.add_argument("--global-checkpoint",type=Path)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args();args.started=now();run=args.run
    if args.dataset is None:
        args.dataset=Path(json.loads((run/"config.yaml").read_text())["data"]) if args.resume else DATA
    config=json.loads(args.config.read_text())
    for stage in ["global","local"]:
        input_size(config,stage);training_scales(config,stage)
    data_meta=json.loads((args.dataset/"metadata/dataset.json").read_text())
    data_counts={s:len(jsonl(args.dataset/"labels"/f"{s}.jsonl")) for s in ["train","val"]}
    config.update(stage=args.stage,smoke=args.smoke,device=args.device,data=str(args.dataset),frame_stride=1,
                  frame_window=1,feature_window_slots=9,batch=config["micro_batch"],gradient_accumulation=config["effective_batch"]//config["micro_batch"],
                  global_checkpoint=str(args.global_checkpoint) if args.global_checkpoint else None,
                  dataset_id=data_meta["dataset_id"],split_id="ttnet_ball_v1_frozen_yolo_pilot",
                  split_hashes=data_meta["split_hashes"],train_videos=data_meta["train_videos"],val_videos=data_meta["val_videos"],
                  train_frames=data_counts["train"],val_frames=data_counts["val"],
                  data_policy="all decoded frames preserved; only positive/confirmed negative labels supervised; unknown/ignore masked")
    if run.exists() and not args.resume:raise RuntimeError("existing run: use --resume or a new run id")
    if args.resume:
        stored=json.loads((run/"config.yaml").read_text())
        # Observational metadata can be enriched; training controls must remain identical.
        observations={"dataset_id","split_id","split_hashes","train_videos","val_videos","train_frames","val_frames",
                      "data_policy","evaluated_best_checkpoint_sha256","best_epoch"}
        if {k:v for k,v in stored.items() if k not in observations}!={k:v for k,v in config.items() if k not in observations}:
            raise RuntimeError("resume configuration differs")
    for name in ["checkpoints","logs","evaluations"]:(run/name).mkdir(parents=True,exist_ok=True)
    # JSON is a YAML 1.2 subset: no additional dependency required.
    save(run/"config.yaml",config)
    attempts=sorted((run/"logs").glob("attempt_*"))
    attempt=run/"logs"/f"attempt_{len(attempts)+1:03d}";attempt.mkdir()
    save(run/"environment.json",dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                    git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip(),
                                    git_status=subprocess.check_output(["git","status","--short"],cwd=PROJECT,text=True),
                                    command=sys.argv,code_hashes={str(x.relative_to(ROOT)):sha(x) for x in ROOT.rglob("*.py")}))
    with (attempt/"console.log").open("w") as f,redirect_stdout(Tee(sys.stdout,f)),redirect_stderr(Tee(sys.stderr,f)):
        try:
            result=train(args,run,config,attempt);save(run/"run.json",result)
        except BaseException as e:
            save(run/"run.json",dict(status="failed",stage=args.stage,started=args.started,finished=now(),error=repr(e)))
            traceback.print_exc();raise


if __name__=="__main__":main()
