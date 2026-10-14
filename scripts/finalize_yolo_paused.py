"""Persist comparison and event-level error index for the paused evaluation."""
import csv
import io
import json
import shutil
import numpy as np
from scipy.optimize import linear_sum_assignment
from run_yolo_ball_full import ROOT, save

exp = '20261013-215749_yolo11n_full1280_public'
eval_id = 'epoch032_public_paused'
record = ROOT / 'experiments/records' / exp
dest = ROOT / 'out/experiments' / exp / 'eval' / eval_id
metrics = json.loads((record / 'evaluations' / f'{eval_id}.json').read_text())
gt = [json.loads(l) for l in (ROOT / 'pingpang/public_data/gt_public.jsonl').read_text().splitlines()]
pred = [json.loads(l) for l in (dest / 'runner/predictions.jsonl').read_text().splitlines()]
errors = []
for vid in metrics['per_video']:
    valid = [g for g in gt if g['video_id'] == vid and g['valid']]
    ignored = [g for g in gt if g['video_id'] == vid and not g['valid']]
    pp = [p for p in pred if p['video_id'] == vid and not any(abs(p['frame_id']-g['frame_id'])<=1 and np.hypot(p['x']-g['x'],p['y']-g['y'])<=50 for g in ignored)]
    cost = np.full((len(valid),len(pp)),1e9)
    for i,g in enumerate(valid):
        for j,p in enumerate(pp):
            d = np.hypot(p['x']-g['x'],p['y']-g['y'])
            if abs(p['frame_id']-g['frame_id'])<=1 and d<=50:
                cost[i,j] = d
    pairs = [(i,j) for i,j in zip(*linear_sum_assignment(cost)) if cost[i,j]<1e9]
    for rows, matched, kind in [(valid,{i for i,j in pairs},'FN'),(pp,{j for i,j in pairs},'FP')]:
        for i,r in enumerate(rows):
            if i not in matched:
                errors.append(dict(r,error_type=kind,overlay=str((dest/'overlays'/f'{vid}.mp4').relative_to(ROOT))))
save(dest / 'event_errors.json', {'errors':errors,'counts':{'FP':sum(e['error_type']=='FP' for e in errors),'FN':sum(e['error_type']=='FN' for e in errors)}})
for name in ['solution.py','solution_local.py','yolo_ball_detector.py']:
    shutil.copy2(ROOT/'pingpang/participant'/name,dest/name)
shutil.copy2(ROOT/'scripts/evaluate_yolo_paused.py',dest/'evaluate_yolo_paused.py')
shutil.copy2(ROOT/'scripts/finalize_yolo_paused.py',dest/'finalize_yolo_paused.py')
path = ROOT/'experiments/comparison.csv'
rows = list(csv.DictReader(path.open()))
fields = list(rows[0])
row = {k:'' for k in fields}
row.update(experiment_id=exp,eval_id=eval_id,status='evaluated_training_paused',task='landing_detection',change='full train YOLO11n 1280, epoch32 + original heuristic',
           data_version='pp_train_data all 20846 positive frames; public landing validation',split_id='yolo_ball_full_v1',protocol='local_public_landing_ignore_first',
           model='YOLO11n',checkpoint=metrics['checkpoint'],checkpoint_sha256=metrics['checkpoint_sha256'],seed=42,shape='1x3x1280x1280',window=1,stride=1,batch=1,
           touch_f1=metrics['macro_f1'],gpu='NVIDIA GeForce RTX 5070 Ti',backend='ONNX Runtime CUDA',precision='FP32',
           e2e_fps=metrics['run_status'].get('performance',{}).get('infer_fps',''),config=f'experiments/records/{exp}/config.yaml',
           log=str((dest/'runner_console.log').relative_to(ROOT)),evaluation=f'experiments/records/{exp}/evaluations/{eval_id}.json',
           predictions=str((dest/'runner/predictions.jsonl').relative_to(ROOT)),notes='touch_f1=angle macro; micro F1=0.398148; conf=.05; 15 complete videos; training remains SIGSTOP paused')
rows = [r for r in rows if (r['experiment_id'],r['eval_id'])!=(exp,eval_id)] + [row]
buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=fields);w.writeheader();w.writerows(rows);path.write_text(buf.getvalue())
keys=['experiment_id','eval_id','task','status','center_f1','touch_f1','e2e_fps']
(ROOT/'experiments/comparison.md').write_text('# Experiment comparison\n\nBall localisation and landing detection are different tasks. touch_f1 is angle macro F1 under the recorded local public protocol.\n\n| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'+'\n'.join('| '+' | '.join(str(r[k]) or '—' for k in keys)+' |' for r in rows)+'\n')
print('Saved event errors, source snapshots and comparison row.')
