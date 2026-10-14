"""Audited YOLO11n ball-only pilot: smoke, training, localisation metrics and speed."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'out/yolo-deps'))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'out/yolo_setup/config')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'out/yolo_setup/matplotlib')
import cv2
import numpy as np
import torch
import ultralytics
import yaml
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def match(gt, pred, threshold, kind='center'):
    if not len(gt) or not len(pred):
        return []
    if kind == 'center':
        dist = np.linalg.norm(((gt[:,:2]+gt[:,2:])/2)[:,None] - ((pred[:,:2]+pred[:,2:])/2)[None],axis=2)
        valid, cost = dist <= threshold, dist
    else:
        inter = np.maximum(0, np.minimum(gt[:,None,2:], pred[None,:,2:])-np.maximum(gt[:,None,:2],pred[None,:,:2])).prod(2)
        area1 = (gt[:,2:]-gt[:,:2]).prod(1)
        area2 = (pred[:,2:]-pred[:,:2]).prod(1)
        iou = inter/(area1[:,None]+area2[None]-inter+1e-9)
        valid, cost = iou >= threshold, 1-iou
    rows, cols = linear_sum_assignment(np.where(valid,cost,1e9))
    return [(int(i),int(j),float(cost[i,j])) for i,j in zip(rows,cols) if valid[i,j]]


def pr(tp,fp,fn):
    p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
    return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(p+r,1e-12))


def evaluate(model, dest):
    records = [json.loads(x) for x in (ROOT/'experiments/splits/yolo_ball_v1/val.jsonl').read_text().splitlines()]
    errors, counts, videos = {10:[],50:[]}, {10:[0,0,0],50:[0,0,0],'iou':[0,0,0]}, {}
    failures=[]
    with (dest/'predictions.jsonl').open('w') as out:
        for k,r in enumerate(records):
            result=model.predict(str(ROOT/r['image']),imgsz=640,conf=.25,iou=.7,max_det=10,device=0,half=False,verbose=False)[0]
            pred=result.boxes.xyxy.cpu().numpy(); scores=result.boxes.conf.cpu().numpy(); gt=np.array(r['boxes'])
            metrics={}
            for th in [10,50,'iou']:
                matches=match(gt,pred, .5 if th=='iou' else th, 'iou' if th=='iou' else 'center')
                tp=len(matches); vals=[tp,len(pred)-tp,len(gt)-tp]
                counts[th]=[a+b for a,b in zip(counts[th],vals)]
                metrics[str(th)]=dict(zip(['tp','fp','fn'],vals))
                if th!='iou': errors[th].extend(m[2] for m in matches)
            v=videos.setdefault(r['video'],dict(tp=0,fp=0,fn=0,max_consecutive_missed=0,run=0,last=-2))
            for key,val in metrics['10'].items(): v[key]+=val
            v['run']=(v['run']+1 if r['frame']==v['last']+1 else 1) if metrics['10']['fn'] else 0
            v['max_consecutive_missed']=max(v['max_consecutive_missed'],v['run']); v['last']=r['frame']
            row=dict(r,predicted_boxes=pred.tolist(),confidence=scores.tolist(),metrics=metrics)
            out.write(json.dumps(row)+'\n')
            if metrics['10']['fn'] or metrics['10']['fp']: failures.append((r,pred,scores))
            if k%500==0: print(f'localisation eval {k}/{len(records)}',flush=True)
    for v in videos.values():
        v.pop('run');v.pop('last');v.update(pr(v['tp'],v['fp'],v['fn']))
    save(dest/'per_video.json',videos)
    preview=dest/'errors';preview.mkdir(exist_ok=True)
    # Deterministic samples distributed across all failing frames, not only the first video.
    for j,idx in enumerate(np.linspace(0,len(failures)-1,min(12,len(failures)),dtype=int)):
        r,pred,scores=failures[idx]; im=cv2.imread(str(ROOT/r['image']))
        for b in r['boxes']: cv2.rectangle(im,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(0,255,0),2)
        for b,s in zip(pred,scores):
            cv2.rectangle(im,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(0,0,255),2)
            cv2.putText(im,f'{s:.2f}',tuple(map(round,b[:2])),0,.5,(0,0,255),1)
        cv2.imwrite(str(preview/f'{j:02d}_{Path(r["image"]).stem}.jpg'),im)
    result={str(t):dict(pr(*counts[t]),error_median_px=float(np.median(errors[t])) if errors[t] else None,
                        error_p90_px=float(np.percentile(errors[t],90)) if errors[t] else None,
                        coverage=counts[t][0]/max(counts[t][0]+counts[t][2],1)) for t in [10,50]}
    result['bbox_iou50_fixed_conf']=pr(*counts['iou'])
    result['max_consecutive_missed_10px']=max(v['max_consecutive_missed'] for v in videos.values())
    result['known_background_false_positives']=None
    return result


def benchmark(model,dest):
    import copy
    from thop import profile
    # Prediction/validation may fuse model.model in place; reload for unfused counts.
    net=YOLO(model.ckpt_path).model.float().eval().to('cuda')
    x=torch.zeros(1,3,640,640,device='cuda')
    params=sum(p.numel() for p in net.parameters())
    macs,_=profile(copy.deepcopy(net),inputs=(x,),verbose=False)
    net.fuse(verbose=False)
    times=[]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(50):net(x)
        torch.cuda.synchronize()
        for repeat in range(3):
            for _ in range(200):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();net(x);end.record();end.synchronize();times.append(dict(repeat=repeat,ms=start.elapsed_time(end)))
    save(dest/'latencies.json',times)
    memory=torch.cuda.max_memory_allocated()/2**20
    del net,x
    torch.cuda.empty_cache()
    audit=json.loads((ROOT/'experiments/splits/yolo_ball_v1/audit.json').read_text())
    selected={}
    for r in audit:
        if r['split']=='val':selected.setdefault(r['group'],r['video'])
    durations=[]
    for group,path in selected.items():
        torch.cuda.synchronize();start=time.perf_counter();cap=cv2.VideoCapture(str(ROOT/path));n=0
        with (dest/f'e2e_{group}.jsonl').open('w') as f:
            while True:
                ok,im=cap.read()
                if not ok:break
                r=model.predict(im,imgsz=640,conf=.25,iou=.7,max_det=10,device=0,half=False,verbose=False)[0]
                f.write(json.dumps(dict(frame=n,boxes=r.boxes.xyxy.cpu().tolist(),conf=r.boxes.conf.cpu().tolist()))+'\n');n+=1
        cap.release();torch.cuda.synchronize();elapsed=time.perf_counter()-start
        durations.append(dict(video=path,frames=n,seconds=elapsed,fps=n/elapsed))
    save(dest/'e2e_videos.json',durations)
    return dict(shape=[1,3,640,640],batch=1,precision='FP32',backend='PyTorch CUDA',gpu=torch.cuda.get_device_name(0),
                model_p50_ms=float(np.median([t['ms'] for t in times])),model_p95_ms=float(np.percentile([t['ms'] for t in times],95)),
                warmup=50,repeats=3,iterations=200,e2e_fps=sum(x['frames'] for x in durations)/sum(x['seconds'] for x in durations),
                e2e_scope='six complete validation videos; CPU decode + preprocessing + H2D + model + NMS + D2H + JSONL writes; initialization excluded; predictor rect=True',
                peak_memory_mib=memory,peak_memory_scope='torch.cuda.max_memory_allocated during forward benchmark; includes resident YOLO and benchmark clone',
                params_m=params/1e6,gmac=macs/1e9,gflops=2*macs/1e9,compute_tool='ultralytics-thop 2.0.18',
                compute_scope='THOP supported modules incl Conv/Linear; functional operations such as attention matmul may be omitted; not an exact total; unfused model; FLOPs=2*MACs')


def comparison(exp,status,metrics=None):
    m=metrics or {}; ap=m.get('bbox_ap',{}); center=m.get('center',{}).get('10',{}); b=m.get('benchmark',{})
    row=dict(experiment_id=exp,eval_id='best_fp32_640',status=status,task='ball_localisation',parent='',change='YOLO11n single-frame pilot',
             data_version='pp_train_data annotated positives',split_id='yolo_ball_v1',protocol='ball_v1',model='YOLO11n',checkpoint=m.get('checkpoint',''),
             checkpoint_sha256=m.get('checkpoint_sha256',''),best_epoch=m.get('best_epoch',''),seed=42,shape='1x3x640x640',window=1,stride=1,batch=1,
             ap50=ap.get('metrics/mAP50(B)',''),ap50_95=ap.get('metrics/mAP50-95(B)',''),ap_precision=ap.get('metrics/precision(B)',''),ap_recall=ap.get('metrics/recall(B)',''),
             center_p=center.get('precision',''),center_r=center.get('recall',''),center_f1=center.get('f1',''),center_median_px=center.get('error_median_px',''),
             center_p90_px=center.get('error_p90_px',''),touch_f1='',gpu=b.get('gpu',''),backend=b.get('backend',''),precision='FP32',
             model_p50_ms=b.get('model_p50_ms',''),model_p95_ms=b.get('model_p95_ms',''),e2e_fps=b.get('e2e_fps',''),peak_memory_mib=b.get('peak_memory_mib',''),
             params_m=b.get('params_m',''),gmac=b.get('gmac',''),gflops=b.get('gflops',''),compute_tool=b.get('compute_tool',''),compute_scope=b.get('compute_scope',''),
             config=f'experiments/records/{exp}/config.yaml',log=f'out/experiments/{exp}/attempts/001/console.log',
             evaluation=f'experiments/records/{exp}/evaluations/best_fp32_640.json' if metrics else '',predictions=f'out/experiments/{exp}/eval/best_fp32_640/predictions.jsonl' if metrics else '',
             notes='positive-labelled frames only; grouped pilot; no background or bounce scoring')
    path=ROOT/'experiments/comparison.csv';rows=list(csv.DictReader(path.open())) if path.exists() else []
    rows=[r for r in rows if (r['experiment_id'],r['eval_id'])!=(exp,row['eval_id'])]+[row]
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(row));w.writeheader();w.writerows(rows)
    tmp=path.with_suffix('.tmp');tmp.write_text(buf.getvalue());tmp.replace(path)
    keys=['experiment_id','status','ap50','ap50_95','center_f1','model_p50_ms','e2e_fps','params_m','gflops']
    (ROOT/'experiments/comparison.md').write_text('# 球定位实验（所有 AP / P / R / F1 为 0–1）\n\n'+'| '+' | '.join(keys)+' |\n'+'|'+'|'.join(['---']*len(keys))+'|\n'+'\n'.join('| '+' | '.join(str(r[k]) or '—' for k in keys)+' |' for r in rows)+'\n\n触桌指标：本轮不执行。\n')


def main():
    exp=sys.argv[1];record=ROOT/'experiments/records'/exp;out=ROOT/'out/experiments'/exp
    record.mkdir(parents=True,exist_ok=False);attempt=out/'attempts/001';attempt.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',started=now(),command=' '.join(sys.argv),cwd=str(ROOT),pid=os.getpid(),
               python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,ultralytics=ultralytics.__version__,
               gpu=torch.cuda.get_device_name(0),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),uv_lock_sha256=sha(ROOT/'uv.lock'),
               pretrained_sha256=sha(ROOT/'out/yolo_setup/weights/yolo11n.pt'),
               pretrained_source='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
               split_hashes={s:sha(ROOT/f'experiments/splits/yolo_ball_v1/{s}.jsonl') for s in ['train','val']})
    save(record/'run.json',state);comparison(exp,'running')
    for p in ['prepare_yolo_ball.py','run_yolo_ball.py']:
        shutil.copy2(ROOT/'scripts'/p,attempt/p)
    (attempt/'workspace.patch').write_text(subprocess.check_output(['git','diff','--no-ext-diff'],cwd=ROOT,text=True))
    shutil.copy2(ROOT/'out/yolo_setup/prepare.log',attempt/'prepare.log')
    base=dict(imgsz=640,batch=16,device=0,workers=4,seed=42,deterministic=True,amp=False,optimizer='AdamW',
              lr0=.001,lrf=.01,weight_decay=.0005,patience=5,pretrained=True,save=True,plots=True,cache=False,
              mosaic=0.,mixup=0.,scale=.2,translate=.1,fliplr=.5,flipud=0.,degrees=0.,close_mosaic=0,project=str(attempt),verbose=False)
    from ultralytics.cfg import DEFAULT_CFG_DICT
    (record/'config.yaml').write_text(yaml.safe_dump(dict(DEFAULT_CFG_DICT, **dict(base, epochs=15, name='train', data=str(ROOT/'out/datasets/yolo_ball_v1/data.yaml'), model=str(ROOT/'out/yolo_setup/weights/yolo11n.pt')))))
    try:
        print('SMOKE: 64 train / 32 val, one epoch',flush=True)
        smoke=YOLO(str(ROOT/'out/yolo_setup/weights/yolo11n.pt'))
        smoke.train(data=str(ROOT/'out/datasets/yolo_ball_v1/smoke.yaml'),epochs=1,name='smoke',**base)
        assert (attempt/'smoke/weights/last.pt').exists()
        del smoke;torch.cuda.empty_cache()
        print('MAIN: 15-epoch positive-labelled source-grouped pilot',flush=True)
        model=YOLO(str(ROOT/'out/yolo_setup/weights/yolo11n.pt'))
        def epoch_record(trainer):
            with (attempt/'epoch_resources.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=trainer.epoch+1, global_step=(trainer.epoch+1)*len(trainer.train_loader), epoch_seconds=trainer.epoch_time, gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))+'\n')
            torch.cuda.reset_peak_memory_stats()
        model.add_callback('on_fit_epoch_end',epoch_record)
        model.train(data=str(ROOT/'out/datasets/yolo_ball_v1/data.yaml'),epochs=15,name='train',**base)
        shutil.copy2(attempt/'train/args.yaml',record/'config.yaml')
        shutil.copy2(attempt/'train/results.csv',attempt/'train_metrics.csv')
        weights=attempt/'train/weights';shutil.copytree(weights,attempt/'checkpoints')
        state['status']='trained_pending_eval';save(record/'run.json',state);comparison(exp,state['status'])
        del model;torch.cuda.empty_cache()
        model=YOLO(str(weights/'best.pt'));evaldir=out/'eval/best_fp32_640';evaldir.mkdir(parents=True)
        val=model.val(data=str(ROOT/'out/datasets/yolo_ball_v1/data.yaml'),imgsz=640,batch=16,device=0,half=False,
                      conf=.001,iou=.7,max_det=10,workers=4,plots=True,project=str(evaldir),name='ap',verbose=False)
        m=dict(bbox_ap={k:float(v) for k,v in val.results_dict.items()},checkpoint=str((weights/'best.pt').relative_to(ROOT)),
               checkpoint_sha256=sha(weights/'best.pt'),protocol='ball_v1',
               bbox_protocol='Ultralytics AP IoU .50:.95, conf floor .001, NMS IoU .7, max_det 10; P/R at best F1 curve threshold, not fixed .25',
               center_protocol='original pixel Euclidean distance 10 and 50; max-cardinality/min-distance Hungarian matching; confidence .25, NMS IoU .7, max_det 10; positives only',
               unknown_policy='excluded, never treated as background; missing frames break consecutive miss runs',touch_metrics=None)
        m['center']=evaluate(model,evaldir)
        benchdir=out/'benchmark/best_fp32_640';benchdir.mkdir(parents=True)
        m['benchmark']=benchmark(model,benchdir)
        history=list(csv.DictReader((attempt/'train/results.csv').open()))
        best=max(history,key=lambda r:float(r['metrics/mAP50-95(B)']))
        m['best_epoch']=int(best['epoch']);m['epochs_run']=len(history)
        save(record/'evaluations/best_fp32_640.json',m)
        state.update(status='completed',finished=now(),exit_code=0);save(record/'run.json',state);comparison(exp,'completed',m)
        c=m['center']['10'];b=m['benchmark']
        (record/'summary.md').write_text(f'''# YOLO11n 球定位首轮测试\n\n只测试单帧球定位，不调用触桌模块。预训练初始化，640 输入，最多 15 epoch，seed=42。\n\n- AP50: {m['bbox_ap']['metrics/mAP50(B)']:.6f}\n- AP50–95: {m['bbox_ap']['metrics/mAP50-95(B)']:.6f}\n- 10px 球心 P/R/F1: {c['precision']:.6f} / {c['recall']:.6f} / {c['f1']:.6f}\n- 匹配误差中位数/P90: {c['error_median_px']} / {c['error_p90_px']} px；覆盖率 {c['coverage']:.6f}\n- 前向 p50 / p95: {b['model_p50_ms']:.3f} / {b['model_p95_ms']:.3f} ms；六段完整视频 E2E {b['e2e_fps']:.2f} FPS\n- 参数 {b['params_m']:.4f} M；THOP {b['gmac']:.4f} GMAC / {b['gflops']:.4f} GFLOPs（有算子覆盖限制）\n\n有球标注帧上的初步结果；无确定背景标签，因此不报告无球误报率。按场景保守分组，但缺少原始来源元数据，不能证明完全独立比赛。验证集用于 best checkpoint 选择，没有独立测试集。首轮无相同协议对照，不能宣称优于热图模型。\n\n训练日志、best/last、逐帧预测、逐视频结果、错误画面和测速原始样本在 `{out.relative_to(ROOT)}`。本机 RTX 5070 Ti / PyTorch FP32，未做 ONNX/TensorRT 导出，不代表赛事 4090 性能。\n''')
        print(json.dumps(m,ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',finished=now(),exit_code=1,error=traceback.format_exc())
        save(record/'run.json',state);comparison(exp,state['status'])
        (record/'summary.md').write_text(state['error']);raise

if __name__=='__main__':main()
