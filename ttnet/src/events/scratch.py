"""Preflight, train from random initialization, then evaluate the locked baseline.

No frozen descriptor cache is read or written. Long runs can be restarted with
--resume; the training command resumes only its own checkpoint and RNG state.
"""
import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from ttnet.common import PROJECT, SRC, DATA, now, save, sha
from ttnet.events.data import EventCorpus, assemble_window
from ttnet.events.continuous import attach_pp, mixed_samples
from ttnet.events.engine import EventSystem
from ttnet.events.train import online_forward, state_digest, atomic_checkpoint
from ttnet.models.event import event_loss


def read(path):return json.loads(Path(path).read_text())


def modules(system):
    return dict(global_model=system.global_model,local_model=system.local_model,table=system.table_head,event=system.head)


def preflight(cfg, root):
    torch.set_num_threads(cfg['torch_threads']);cv2.setNumThreads(1)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    corpus=EventCorpus(cfg['dataset']);attach_pp(corpus,cfg['pp_annotations'])
    system=EventSystem(cfg,torch.device('cuda'));system.configure('joint',True)
    initial={k:state_digest(m) for k,m in modules(system).items()}
    sample=next(s for s in corpus.samples['train'] if s.get('source')=='pp' and s['target'][0]>0)
    optimizer=torch.optim.AdamW(system.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.cuda.amp.GradScaler(init_scale=cfg.get('amp_init_scale',1024.))
    schedule=mixed_samples(corpus.samples['train'],cfg['seed'],0,cfg)
    history=[];event_grads={};capture={}
    def capture_window(*args,**kwargs):
        capture['records']=args[4]
        return assemble_window(*args,**kwargs)
    torch.cuda.reset_peak_memory_stats()
    for step in range(12):
        # Repeated real positive for a learning check, then shuffled sources for timing.
        chosen=sample if step<8 else schedule[step-8]
        torch.cuda.synchronize();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',enabled=cfg['amp']),patch('ttnet.events.joint.assemble_window',side_effect=capture_window):
            logits,aux=online_forward(system,corpus,chosen)
            event=event_loss(logits,torch.tensor([chosen['target']],device='cuda'),torch.tensor([chosen['supervised']],device='cuda'))
            loss=event+sum(cfg['loss_weights'][k]*v for k,v in aux.items())
        if step==0:
            for name,model in [('global',system.global_model),('local',system.local_model)]:
                g=torch.autograd.grad(event,next(model.encoder.parameters()),retain_graph=True)[0]
                event_grads[name]=float(g.detach().abs().sum())
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        finite=all(torch.isfinite(p.grad).all().item() for p in system.parameters() if p.grad is not None)
        if not finite:raise FloatingPointError(f'nonfinite preflight gradient at {step}')
        norms={k:sum(float(p.grad.detach().float().norm()) for p in m.parameters() if p.grad is not None) for k,m in modules(system).items()}
        torch.nn.utils.clip_grad_norm_(system.parameters(),cfg['gradient_clip_norm'])
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        row=dict(step=step,source=chosen.get('source','ttgames'),seconds=time.perf_counter()-started,
                 event=float(event.detach()),aux={k:float(v.detach()) for k,v in aux.items()},gradient_norms=norms)
        history.append(row);print(json.dumps(row),flush=True)
        if step==0:
            changed={k:state_digest(m)!=initial[k] for k,m in modules(system).items()}
            if not all(changed.values()) or not all(g>0 for g in event_grads.values()):raise AssertionError('first joint update failed')
        if step==7:
            rec=capture['records'][sample['frame_id']];frame=corpus.videos[sample['video_id']]['frames'][sample['frame_id']]
            image=cv2.cvtColor(corpus.image(sample['video_id'],sample['frame_id']).copy(),cv2.COLOR_RGB2BGR)
            cv2.drawMarker(image,tuple(round(v) for v in frame['xy']),(60,240,60),cv2.MARKER_CROSS,24,2)
            pred=rec.get('prediction')
            if pred and pred.get('valid'):cv2.circle(image,(round(pred['x']),round(pred['y'])),10,(60,60,255),2)
            for crop in rec['roi_transforms']:
                x,y,side=crop['left'],crop['top'],crop['side'];cv2.rectangle(image,(x,y),(x+side,y+side),(0,220,255),2)
            cv2.putText(image,'PREFLIGHT only: green GT / red prediction / yellow training ROI',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            dest=root/'preflight/training_overlay.jpg';dest.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(dest),image):raise OSError(dest)
    if history[7]['event']>=history[0]['event']:raise AssertionError('repeated real-sample event loss did not decrease')
    # A full checkpoint restores all four modules without any pretrained files.
    path=root/'preflight/checkpoint.pt'
    atomic_checkpoint(path,dict(system=system.state_dict(),config=cfg,diagnostic_only=True))
    clone=EventSystem(cfg,torch.device('cuda'),initialize=False)
    clone.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['system'])
    assert all(state_digest(modules(clone)[k])==state_digest(m) for k,m in modules(system).items())
    timings=[r['seconds'] for r in history[8:]]
    result=dict(passed=True,diagnostic_weights_not_used_for_training=True,history=history,
                first_update_changed=changed,event_encoder_gradient_l1=event_grads,initial_state_sha256=initial,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),samples_per_epoch=len(schedule),
                pp_samples=sum(s.get('source')=='pp' for s in schedule),
                train_video_count=sum(v.startswith('pp_') for v in corpus.split['train']),
                val_video_count=sum(v.startswith('pp_') for v in corpus.split['val']),
                data_hashes={p:sha(DATA/f'splits/{p}.jsonl') for p in ('train','val')},
                median_shuffled_sample_seconds=float(np.median(timings)),
                estimated_train_epoch_seconds=float(np.mean(timings))*len(schedule),
                note='Four shuffled windows give an approximate runtime only; full epochs include 40-video validation.')
    save(root/'preflight/metrics.json',result)
    return result


def command(root,name,args):
    marker=root/'steps'/f'{name}.json'
    if marker.exists() and read(marker)['exit_code']==0:return
    logs=root/'logs';logs.mkdir(exist_ok=True)
    log=logs/f'{name}_{time.time_ns()}.log'
    cmd=[sys.executable,'-u','-m',*map(str,args)]
    save(root/'pipeline.json',dict(status='running',stage=name,started=now(),log=str(log),command=cmd))
    publish(root,PROJECT/'result'/root.name)
    started=time.perf_counter()
    with log.open('w') as handle:
        process=subprocess.Popen(cmd,cwd=PROJECT,stdout=handle,stderr=subprocess.STDOUT)
        save(root/'active_process.json',dict(stage=name,pid=process.pid,log=str(log),command=cmd,started=now()))
        code=process.wait()
    save(marker,dict(command=cmd,log=str(log),exit_code=code,elapsed_sec=time.perf_counter()-started,finished=now()))
    if code:raise RuntimeError(f'{name} exited {code}; see {log}')


def audit(root):
    run=root/'train';checkpoint=run/'checkpoints/best.pt'
    lock=read(root/'locked_selection.json');cfg=read(run/'config.yaml')
    dest=run/'evaluations/continuous';metrics=read(dest/'metrics.json')
    external=read(run/'evaluations/external/metrics.json')
    pre=read(root/'preflight/metrics.json')
    digest=sha(checkpoint)
    checks=dict(checkpoint_identity=digest==lock['checkpoint_sha256']==metrics['checkpoint_sha256']==external['identity']['checkpoint_sha256'],
                locked_thresholds=lock['thresholds']==external['identity']['thresholds'] and lock['thresholds'][0]==metrics['metrics']['threshold'],
                full_40_videos=len(metrics['metrics']['per_video'])==40 and len(metrics['performance'])==40,
                full_40_overlays=len(list(dest.glob('*_overlay.mp4')))==40,
                errors_indexed=(dest/'error_index.jsonl').exists() and (dest/'index.html').exists(),
                best_and_last=checkpoint.exists() and (run/'checkpoints/last.pt').exists(),
                all_random=cfg['initialization']=='random' and not any(cfg.get(k) for k in ('global_checkpoint','local_checkpoint','table_checkpoint','ar_checkpoint')),
                first_step_all_modules_updated=all(pre['first_update_changed'].values()))
    checks['fresh_formal_initialization']=all(cfg['initial_state_sha256'][k]==pre['initial_state_sha256'][v]
                    for k,v in {'global':'global_model','local':'local_model','table':'table','event':'event'}.items())
    save(root/'artifact_check.json',dict(passed=all(checks.values()),checks=checks,checkpoint_sha256=digest,checked=now()))
    if not all(checks.values()):raise AssertionError(checks)


def publish(root,archive):
    archive.mkdir(parents=True,exist_ok=True)
    for name in ('config.yaml','pipeline.json','locked_selection.json','artifact_check.json'):
        if (root/name).exists():shutil.copyfile(root/name,archive/name)
    if (root/'train/config.yaml').exists():shutil.copyfile(root/'train/config.yaml',archive/'actual_config.yaml')
    if (root/'preflight/metrics.json').exists():shutil.copyfile(root/'preflight/metrics.json',archive/'preflight.json')
    run=root/'train';eval_path=run/'evaluations/continuous/metrics.json';external=run/'evaluations/external/metrics.json'
    lines=['# 全随机初始化联合训练 baseline','',
           f"状态：`{read(root/'pipeline.json')['status']}`。",'',
           'Global、Local、桌框头、TCN 全部从随机初始化开始，第 1 轮联合更新；固定 seed42 和 PP 114/40 视频划分。',
           '规则关联与硬坐标/ROI不求导；事件损失通过视觉描述符更新两路编码器，定位与桌框监督训练对应头。',
           '不读取旧 checkpoint 或冻结描述符缓存。预检权重丢弃，正式训练重新设 seed42 初始化。','',
           '13 帧原生帧号窗口、真实 PTS 时间编码、桌框 EMA 0.15s；PP/TTGames 各50%，沿用原始采样。',
           'AdamW，所有网络 lr=3e-4，effective batch8，AMP，最多30轮；第15轮起 patience8。',
           '损失 Event + Global + Local + 0.2 Table；缺失/未知标签屏蔽。',
           'Local GT+jitter 裁剪在前3轮概率1，第10轮为0，此后只用预测；推理始终不使用GT。',
           '训练在每个窗口重置规则追踪器，并额外保留首帧前驱；验证从视频起点连续更新。',
           '每轮完整40个PP验证视频，按三视角落点宏F1选best，同分FP更少；最终锁阈值后做原视频评估与外部评估。','',
           f'- [实时训练状态]({run}/run.json)',f'- [配置]({archive}/config.yaml)',
           f'- [训练权重]({run}/checkpoints/best.pt)',f'- [last / 恢复点]({run}/checkpoints/last.pt)',
           f'- [预检标签/ROI叠加]({root}/preflight/training_overlay.jpg)',
           f'- [训练日志目录]({root}/logs)','']
    probe=root/'preflight/metrics.json'
    if probe.exists():
        p=read(probe);lines += [f"预检通过：四个模块首步均更新，事件梯度到达 Global/Local，checkpoint 可完整恢复；估计每轮纯训练 {p['estimated_train_epoch_seconds']/60:.1f} 分钟。",'']
    if eval_path.exists():
        result=read(eval_path);m=result['metrics']
        shutil.copyfile(eval_path,archive/'validation_metrics.json')
        lines += [f"验证：宏 F1 {m['macro_f1']:.2%}，TP/FP/FN {m['landing']['tp']}/{m['landing']['fp']}/{m['landing']['fn']}，阈值 {m['threshold']}，原视频 FPS {result['e2e_fps']:.2f}。",
                  f"[全部错误叠加]({eval_path.parent}/index.html)",'']
        row=dict(experiment_id=root.name,eval_id='continuous',macro_f1=m['macro_f1'],tp=m['landing']['tp'],fp=m['landing']['fp'],fn=m['landing']['fn'],e2e_fps=result['e2e_fps'],epoch=result['epoch'],checkpoint_sha256=result['checkpoint_sha256'],status='evaluated')
        with (archive/'comparison.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    else:lines += ['尚无完成的验证结果，不能据此判断优于原方案。','']
    if external.exists():
        shutil.copyfile(external,archive/'external_metrics.json');m=read(external)['metrics']
        lines += [f"锁定后外测：TTGames sparse anchor 宏 F1 {m['ttgames_test']['macro_f1']:.2%}；PP public 落点宏 F1 {m['pp_public']['macro_f1']:.2%}。",'']
    lines += ['该实验同时改变初始化和优化范围，是新的完整训练基线，不能单独归因于联合训练。',
              '完整视频 FPS 含解码、全部网络、关联、后处理、JSON 序列化，初始化和磁盘写入另计。',
              '外部数据已在历史实验查看过，不称为全新盲测；外测结果不再用于本轮调参。']
    (archive/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=args.run.resolve();archive=PROJECT/'result'/root.name
    if (root/'pipeline.json').exists() and not args.resume:raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True);cfg=read(args.config)
    if cfg.get('initialization')!='random' or not cfg.get('joint_window_training'):raise ValueError('scratch configuration required')
    if (root/'config.yaml').exists() and read(root/'config.yaml')!=cfg:raise ValueError('pipeline configuration changed')
    save(root/'config.yaml',cfg)
    save(root/'pipeline.json',dict(status='preflight',started=now(),initialization='random'))
    # Preserve the exact source used by this long-running experiment.
    snapshot=root/'source_snapshot'
    for path in SRC.rglob('*.py'):
        target=snapshot/path.relative_to(SRC)
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    publish(root,archive)
    try:
        if not (root/'preflight/metrics.json').exists():preflight(cfg,root)
        torch.cuda.empty_cache()
        publish(root,archive)
        run=root/'train'
        train_args=['ttnet.events.train','--config',root/'config.yaml','--run',run,'--dataset',cfg['dataset'],'--mode','joint','--device','cuda']
        if run.exists():train_args+=['--resume']
        command(root,'train',train_args)
        checkpoint=run/'checkpoints/best.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        lock=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),epoch=state['epoch'],thresholds=state['thresholds'],locked=now(),criterion='PP validation macro F1, fewer FP on ties; no external tuning')
        if (root/'locked_selection.json').exists():
            old=read(root/'locked_selection.json')
            if any(old[k]!=lock[k] for k in ('checkpoint_sha256','thresholds')):raise ValueError('selection changed after lock')
        else:save(root/'locked_selection.json',lock)
        dest=run/'evaluations/continuous'
        if dest.exists() and not (dest/'metrics.json').exists():dest.rename(dest.with_name('continuous_incomplete_'+str(time.time_ns())))
        command(root,'continuous',['ttnet.events.comparison','evaluate','--checkpoint',checkpoint,'--annotations',cfg['pp_annotations'],'--output',dest,'--device','cuda','--scheme','scratch_joint_baseline','--use-checkpoint-threshold'])
        # external.py evaluates TTGames test and public exactly once at locked thresholds.
        command(root,'external',['ttnet.scene.external','--run',run])
        audit(root)
        save(root/'pipeline.json',dict(status='completed',finished=now(),checkpoint_sha256=lock['checkpoint_sha256']))
        status=read(run/'run.json');save(run/'run.json',dict(status,status='completed',evaluation=str(dest/'metrics.json'),finished=now()))
    except BaseException as exc:
        save(root/'pipeline.json',dict(status='partial',error=repr(exc),finished=now()))
        raise
    finally:publish(root,archive)


if __name__=='__main__':main()
