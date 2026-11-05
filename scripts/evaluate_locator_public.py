"""Frozen A/B locators through the original public runner and landing heuristic."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import av
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttnet.common import PROJECT, now, save, sha

PUBLIC = PROJECT / 'pingpang/public_data'
MIXED = PROJECT / 'out/experiments/20261103-140723_mixed_ball_comparison'
PROBE = PROJECT / 'out/results/20261103_mixed_ball_export_recovery/final'


def alive(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return path.exists() and bool(path.read_bytes())


def yield_gpu(task, output):
    state = json.loads((task / 'task.json').read_text())
    pid = state.get('coordinator_pid')
    if not pid or not alive(pid):
        return None
    argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if b'ttnet.training.mixed_experiment' not in argv:
        raise RuntimeError('coordinator identity differs')
    if state['status'] != 'evaluating':
        raise RuntimeError('only interrupt read-only evaluation, never active training')
    # Let any exclusive benchmark/export subprocess finish before yielding.
    deadline = time.monotonic() + 180
    while Path(f'/proc/{pid}/task/{pid}/children').read_text().strip():
        if time.monotonic() > deadline:
            raise TimeoutError('evaluation subprocess did not finish')
        time.sleep(1)
    save(output / 'mixed_before.json', state)
    launch = PROJECT / 'out/launches' / task.name
    previous = json.loads((launch / 'launch.json').read_text())
    os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + 30
    while alive(pid):
        if time.monotonic() > deadline:
            raise TimeoutError('evaluation did not release GPU')
        time.sleep(.2)
    save(output / 'gpu_yield.json', dict(reason='user requested public locator integration; isolate GPU timing',
                                       at=now(),previous_coordinator_pid=pid,task=str(task),launch=previous))
    return previous


def resume_mixed(task, previous, output):
    if previous is None:
        return
    state = json.loads((task / 'task.json').read_text())
    pid = state.get('coordinator_pid')
    if pid and alive(pid):
        return
    launch = PROJECT / 'out/launches' / task.name
    recovery = task / 'recoveries' / (output.name + '_public_gpu_yield')
    recovery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task / 'task.json', recovery / 'task_interrupted.json')
    shutil.copy2(output / 'gpu_yield.json', recovery / 'recovery.json')
    state.setdefault('recovery_history', []).append(str((recovery / 'recovery.json').relative_to(PROJECT)))
    state.update(status='resuming_evaluation',updated=now())
    state.pop('failed',None);state.pop('error',None)
    save(task / 'task.json', state)
    with (launch / 'driver.log').open('a') as stream:
        stream.write('\nRESUME after public integration GPU tests: ' + str(output) + '\n');stream.flush()
        child = subprocess.Popen(previous['command'], cwd=PROJECT, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    previous.update(pid=child.pid,resumed=now(),recovery=str(recovery),
                    previous_coordinator_pids=previous.get('previous_coordinator_pids',[])+[pid],
                    note='read-only evaluation resumed after exclusive public integration tests')
    save(launch / 'launch.json', previous)
    save(output / 'mixed_resumed.json', dict(pid=child.pid,at=now()))


def freeze_configs(output):
    mixed = json.loads((MIXED / 'task.json').read_text())
    configs = {}
    for variant, runs in mixed['runs'].items():
        checkpoints = {s:Path(r)/'checkpoints/best.pt' for s,r in runs.items()}
        data = {s:torch.load(p,map_location='cpu',weights_only=False) for s,p in checkpoints.items()}
        if data['local']['global_checkpoint_sha256'] != sha(checkpoints['global']):
            raise ValueError('Global/Local pair differs')
        cfg = dict(variant=variant,global_threshold=data['global']['threshold'],local_threshold=data['local']['threshold'],
                   global_input=data['global']['config']['global_input'],local_input=data['local']['config']['local_input'],
                   local_crop_side_1080=data['local']['config']['local_crop_side_1080'],
                   training_dataset_id=data['global']['dataset_id'],threshold_selection='PP only; public never used to tune',
                   timestamps_path=str(output/'public_timestamps.json'),timestamps_sha256=sha(output/'public_timestamps.json'),
                   inference='ORT CUDA FP32, TF32 disabled; independent peak and trained Local, Global fallback')
        for stage in ('global','local'):
            source = PROBE / f'{variant}_{stage}'
            parity = json.loads((source/'parity.json').read_text())
            if not parity['parity']['passed'] or parity['checkpoint_sha256'] != sha(checkpoints[stage]):
                raise ValueError('export checkpoint identity or parity differs')
            destination = output/'models'/variant/f'{stage}.onnx';destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/f'{stage}.onnx',destination)
            if sha(destination) != parity['onnx_sha256']:
                raise ValueError('export ONNX identity differs')
            cfg.update({stage+'_onnx':str(destination),stage+'_onnx_sha256':sha(destination),
                        stage+'_checkpoint':str(checkpoints[stage]),stage+'_checkpoint_sha256':sha(checkpoints[stage])})
        path=output/'configs'/f'{variant}.json';save(path,cfg);configs[variant]=path
    return configs


def source_identity(output):
    manifest=json.loads((PUBLIC/'manifest.json').read_text())
    pp={}
    for path in (PROJECT/'data/datasets/pp_ball_v2/metadata/videos').glob('*.json'):
        row=json.loads(path.read_text());pp[row['video_sha256']]=dict(video_id=row['video_id'],split=row['split'])
    rows=[];timestamps={}
    for video in manifest['videos']:
        path=PUBLIC/video['path'];digest=sha(path)
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            times=[float(frame.pts*frame.time_base) for frame in container.decode(stream)]
        if len(times)!=video['n_frames'] or not np.all(np.diff(times)>0):
            raise ValueError('public source timestamps or frame count invalid')
        timestamps[video['video_id']]=[t-times[0] for t in times]
        rows.append(dict(video_id=video['video_id'],path=video['path'],sha256=digest,frames=len(times),
                         fps=video['fps'],interval_min_sec=float(np.diff(times).min()),
                         interval_max_sec=float(np.diff(times).max()),exact_pp_match=pp.get(digest)))
    value=dict(manifest_sha256=sha(PUBLIC/'manifest.json'),gt_sha256=sha(PUBLIC/'gt_public.jsonl'),videos=rows,
               exact_train_video_matches=[r for r in rows if r['exact_pp_match'] and r['exact_pp_match']['split']=='train'],
               scope='exact byte hashes; resized/reencoded or same-source clip overlap is not established by this check')
    save(output/'public_identity.json',value)
    save(output/'public_timestamps.json',timestamps)
    return manifest


def score(prediction_path, manifest):
    predictions=[json.loads(l) for l in prediction_path.read_text().splitlines() if l.strip()]
    gt=[json.loads(l) for l in (PUBLIC/'gt_public.jsonl').read_text().splitlines() if l.strip()]
    per_video={};angles=defaultdict(lambda:dict(tp=0,fp=0,fn=0));errors=[];matches=[]
    def finish(m):
        p=m['tp']/max(1,m['tp']+m['fp']);r=m['tp']/max(1,m['tp']+m['fn'])
        return dict(m,precision=p,recall=r,f1=2*p*r/max(1e-12,p+r))
    for video in manifest['videos']:
        vid=video['video_id'];all_gt=[g for g in gt if g['video_id']==vid]
        valid=[g for g in all_gt if g['valid']];ignored=[g for g in all_gt if not g['valid']]
        pp=[p for p in predictions if p['video_id']==vid]
        usable=[p for p in pp if not any(abs(p['frame_id']-g['frame_id'])<=1 and np.hypot(p['x']-g['x'],p['y']-g['y'])<=50 for g in ignored)]
        cost=np.full((len(valid),len(usable)),1e9)
        for i,g in enumerate(valid):
            for j,p in enumerate(usable):
                distance=float(np.hypot(p['x']-g['x'],p['y']-g['y']))
                if abs(p['frame_id']-g['frame_id'])<=1 and distance<=50:cost[i,j]=distance
        pairs=[(i,j) for i,j in zip(*linear_sum_assignment(cost)) if cost[i,j]<1e9]
        used_gt={i for i,j in pairs};used_pred={j for i,j in pairs}
        for i,j in pairs:matches.append(dict(video_id=vid,gt=valid[i],prediction=usable[j],error_px=float(cost[i,j])))
        for i,g in enumerate(valid):
            if i not in used_gt:errors.append(dict(video_id=vid,frame_id=g['frame_id'],kind='FN',event=g))
        for j,p in enumerate(usable):
            if j not in used_pred:errors.append(dict(video_id=vid,frame_id=p['frame_id'],kind='FP',event=p))
        metric=dict(tp=len(pairs),fp=len(usable)-len(pairs),fn=len(valid)-len(pairs),
                    predictions=len(pp),valid_gt=len(valid),ignored_predictions=len(pp)-len(usable))
        per_video[vid]=finish(metric)
        for key in ('tp','fp','fn'):angles[video['angle']][key]+=metric[key]
    per_angle={a:finish(m) for a,m in angles.items()}
    overall=finish({k:sum(m[k] for m in per_video.values()) for k in ('tp','fp','fn')})
    return dict(protocol='public local scorer: valid GT, ignored-event suppression, Hungarian one-to-one, +/-1 source frame and <=50px',
                overall=overall,per_angle=per_angle,per_video=per_video,
                macro_f1=float(np.mean([m['f1'] for m in per_angle.values()])),matches=matches,errors=errors)


def run_case(name, config, output, manifest):
    destination=output/'eval'/name;destination.mkdir(parents=True,exist_ok=False)
    entry='participant.solution_local:Solution' if name=='YOLO' else 'participant.solution_locator_local:Solution'
    command=[sys.executable,'pingpang/run.py','--input',str(PUBLIC),'--output',str(destination/'runner'),
             '--solution',entry,'--decoder','cpu']
    env=dict(os.environ)
    if config:env['PINGPONG_LOCATOR_CONFIG']=str(config)
    print('RUN '+name+' '+' '.join(command),flush=True)
    with (destination/'console.log').open('w') as log:
        result=subprocess.run(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(destination/'exit.json',dict(command=command,exit_code=result.returncode,finished=now()))
    if result.returncode:raise RuntimeError(f'{name} runner failed; see {destination}/console.log')
    status=json.loads((destination/'runner/run_status.json').read_text())
    if status['status']!='ok' or len(status['videos'])!=len(manifest['videos']) or any(v['status']!='ok' for v in status['videos']):
        raise ValueError('not all public videos completed')
    if status['performance']['total_frames']!=sum(v['n_frames'] for v in manifest['videos']) or status['validation']['n_errors']:
        raise ValueError('frame count or competition output validation failed')
    if config:
        for v in manifest['videos']:
            rows=[json.loads(l) for l in (destination/f'runner/locator_trace/{v["video_id"]}.jsonl').read_text().splitlines()]
            if [r['frame_id'] for r in rows]!=list(range(v['n_frames'])):raise ValueError('centre frame drift')
    metric=score(destination/'runner/predictions.jsonl',manifest)
    metric.update(name=name,run_status=status,configuration=json.loads(config.read_text()) if config else dict(locator='current YOLO entry'),
                  performance_scope='exclusive RTX 5070 Ti, original runner CPU decode + inference + heuristic + outputs; initialization separate; not official TensorRT timing')
    save(destination/'metrics.json',metric)
    print(name,json.dumps(dict(macro_f1=metric['macro_f1'],overall=metric['overall'],per_angle=metric['per_angle'],
                              performance=status['performance']),ensure_ascii=False),flush=True)
    return metric


def visualize(name, metric, output, manifest):
    destination=output/'eval'/name;pred=[json.loads(l) for l in (destination/'runner/predictions.jsonl').read_text().splitlines()]
    gt=[json.loads(l) for l in (PUBLIC/'gt_public.jsonl').read_text().splitlines()]
    images={};examples=[]
    for v in manifest['videos']:
        vid=v['video_id'];wanted={e['frame_id'] for e in metric['errors'] if e['video_id']==vid}
        if not wanted:
            candidates=[g['frame_id'] for g in gt if g['video_id']==vid and g['valid']]
            if candidates:wanted.add(candidates[0])
        trace={}
        path=destination/f'runner/locator_trace/{vid}.jsonl'
        if path.exists():trace={r['frame_id']:r for r in (json.loads(l) for l in path.read_text().splitlines())}
        with av.open(str(PUBLIC/v['path'])) as container:
            for fid,frame in enumerate(container.decode(video=0)):
                if fid not in wanted:continue
                im=cv2.cvtColor(frame.to_ndarray(format='rgb24'),cv2.COLOR_RGB2BGR)
                cv2.putText(im,f'{name} {vid} frame={fid} | GT green / ignored yellow / landing pred red / ball cyan',
                            (25,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                kinds=sorted({e['kind'] for e in metric['errors'] if e['video_id']==vid and e['frame_id']==fid})
                cv2.putText(im,','.join(kinds) if kinds else 'sample', (25,70),0,.9,(0,0,255),2)
                for rows,is_gt in ((gt,True),(pred,False)):
                    for row in rows:
                        if row['video_id']!=vid or abs(row['frame_id']-fid)>1:continue
                        color=((0,255,0) if row.get('valid',True) else (0,255,255)) if is_gt else (0,0,255)
                        point=(round(row['x']),round(row['y']));cv2.circle(im,point,15,color,3)
                        cv2.putText(im,('GT' if is_gt else 'P')+f' f={row["frame_id"]}',(point[0]+18,point[1]),0,.6,color,2)
                ball=trace.get(fid,{}).get('prediction')
                if ball:cv2.circle(im,(round(ball['x']),round(ball['y'])),7,(255,255,0),2)
                path=destination/'overlays'/vid/f'{fid:06d}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(path),im,[cv2.IMWRITE_JPEG_QUALITY,86]):raise OSError(path)
                images[vid,fid]=str(path.relative_to(destination));examples.append(dict(video_id=vid,frame_id=fid,kinds=kinds,image=images[vid,fid]))
    for error in metric['errors']:error['visualization']=images[error['video_id'],error['frame_id']]
    save(destination/'error_index.json',dict(errors=metric['errors'],coverage='all scored FP and FN events',examples=examples))
    (destination/'index.html').write_text('<meta charset="utf-8"><h1>'+name+' public errors</h1>'+''.join(
        f'<figure><img loading="lazy" width="640" src="{e["image"]}"><figcaption>{e["video_id"]}:{e["frame_id"]} {e["kinds"]}</figcaption></figure>' for e in examples))
    save(destination/'metrics.json',metric)


def report(output, record, results, identity):
    dest=PROJECT/'out/results'/output.name;dest.mkdir(parents=True,exist_ok=True)
    fields=['name','macro_f1','micro_f1','precision','recall','tp','fp','fn','0度','45度','90度','fps','prepare_sec','error_index']
    rows=[]
    for name,m in results.items():
        rows.append(dict(name=name,macro_f1=m['macro_f1'],micro_f1=m['overall']['f1'],
                         **{k:m['overall'][k] for k in ('precision','recall','tp','fp','fn')},
                         **{a:m['per_angle'][a]['f1'] for a in ('0度','45度','90度')},
                         fps=m['run_status']['performance']['infer_fps'],prepare_sec=m['run_status']['timing']['prepare_sec'],
                         error_index=str((output/f'eval/{name}/index.html').relative_to(PROJECT))))
    with (record/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    shutil.copy2(record/'comparison.csv',dest/'comparison.csv')
    text='# Frozen locators + unchanged competition landing heuristic\n\n'
    text+='All 15 public videos / 2,109 source frames. Fixed MIN_GAP=10, FALL_AMP=2, RISE_AMP=1, CONF_TH=0.05. No event head, association or public tuning. Locator thresholds and weights selected on PP only.\n\n'
    text+='| Model | 0° F1 | 45° F1 | 90° F1 | Macro F1 | Micro F1 | P | R | TP/FP/FN | Local FPS |\n|---|---:|---:|---:|---:|---:|---:|---:|---|---:|\n'
    for r in rows:
        text+=f'| {r["name"]} | '+ ' | '.join(f'{r[k]:.2%}' for k in ('0度','45度','90度','macro_f1','micro_f1','precision','recall'))+f' | {r["tp"]}/{r["fp"]}/{r["fn"]} | {r["fps"]:.1f} |\n'
    text+='\nLocal ORT CUDA / CPU decoder on RTX 5070 Ti; inference time includes decoder, locator, original heuristic and outputs; initialization separate. This is not an official 4090/TensorRT acceptance test. The fixed3 locator adds one source-frame lookahead while preserving original event frame IDs.\n\n'
    text+='Public scores use the existing local protocol: valid GT, ignore predictions near invalid GT, one-to-one ±1 frame / 50px. Exact video-hash matches are listed in public_identity.json; resized/reencoded source overlap has not been excluded. Public is diagnostic validation, not a new claim of independent test generalization.\n\n'
    for name in results:
        target=os.path.relpath(output/f'eval/{name}/index.html',dest)
        text+=f'- [{name}: all scored FP/FN overlays]({target})\n'
    (dest/'summary.md').write_text(text);(record/'summary.md').write_text(text)
    header='<meta charset="utf-8"><h1>Public: locator + original heuristic</h1><pre>'+html.escape(text)+'</pre>'
    (dest/'index.html').write_text(header+''.join(f'<p><a href="{os.path.relpath(output/f"eval/{name}/index.html",dest)}">{name} all errors</a></p>' for name in results))
    save(dest/'metrics.json',dict(results=results,identity=identity))
    return dest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    stamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S')
    output=(args.output or PROJECT/f'out/experiments/{stamp}_locator_public_heuristic').resolve();output.mkdir(parents=True,exist_ok=False)
    record=PROJECT/'experiments/records'/output.name;record.mkdir(parents=True,exist_ok=False)
    state=dict(status='prepared',started=now(),stage='public_landing',experiment_id=output.name,command=sys.argv)
    save(output/'run.json',state);save(record/'run.json',state)
    previous=None
    try:
        manifest=source_identity(output);configs=freeze_configs(output)
        files=['pingpang/participant/solution.py','pingpang/participant/solution_local.py','pingpang/participant/solution_locator_local.py',
               'ttnet/src/data_tools/geometry.py','ttnet/src/tracking/independent.py',__file__]
        hashes={str(p):sha(PROJECT/p) for p in files}
        save(record/'config.yaml',dict(cases={v:json.loads(p.read_text()) for v,p in configs.items()},
                                      heuristic=dict(MIN_GAP=10,FALL_AMP=2,RISE_AMP=1,CONF_TH=.05),
                                      baseline_onnx_sha256=sha(PROJECT/'pingpang/participant/weights/yolo_ball_full_v1.onnx'),source_hashes=hashes))
        (record/'purpose.md').write_text('# Public locator integration\n\nReplace only ball centres with final A/B Global+Local; inherit existing landing heuristic without parameter changes. Compare current YOLO entry on identical 15 complete public videos.\n')
        subprocess.run([sys.executable,'integrity.py','--verify'],cwd=PROJECT/'pingpang',check=True)
        previous=yield_gpu(MIXED,output)
        state.update(status='running_public',updated=now());save(output/'run.json',state);save(record/'run.json',state)
        try:
            results={name:run_case(name,configs.get(name),output,manifest) for name in ('YOLO','A','B')}
        finally:
            resume_mixed(MIXED,previous,output);previous=None
        for name,metric in results.items():visualize(name,metric,output,manifest)
        if any(sha(PROJECT/p)!=value for p,value in hashes.items()):raise ValueError('source changed during frozen evaluation')
        subprocess.run([sys.executable,'integrity.py','--verify'],cwd=PROJECT/'pingpang',check=True)
        destination=report(output,record,results,json.loads((output/'public_identity.json').read_text()))
        state.update(status='completed',finished=now(),result=str(destination),results={n:dict(macro_f1=m['macro_f1'],overall=m['overall']) for n,m in results.items()})
    except BaseException as error:
        state.update(status='failed',finished=now(),error=repr(error))
        raise
    finally:
        if previous is not None:resume_mixed(MIXED,previous,output)
        save(output/'run.json',state);save(record/'run.json',state)
    print(json.dumps(state,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
