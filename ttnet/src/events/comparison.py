"""Continuous PP evaluation, historical A0, and auditable four-way result tables."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ttnet.common import DATA, PROJECT, identity, jsonl, save, save_jsonl, sha
from ttnet.events.continuous import validate_annotations, select_continuous_threshold, score_continuous, known
from ttnet.events.engine import EventSystem
from ttnet.events.inference import predict_video
from ttnet.evaluation.ball_corpus import BallCorpus
from ttnet.evaluation.candidate_audit import overlay, classify_error
from ttnet.evaluation.metrics import aggregate


def localization_diagnostics(rows, records, annotation):
    predictions=[dict(records[r['frame_id']]['prediction'],crop_transform=records[r['frame_id']]['crop_transform'])
                 if records[r['frame_id']]['prediction'] else None for r in rows]
    near=[(r,p) for r,p in zip(rows,predictions) if any(abs(r['frame_id']-e['frame_id'])<=1 for e in annotation['events'] if e['kind']=='bounce')]
    jumps=0
    for (a,p),(b,q) in zip(zip(rows,predictions),zip(rows[1:],predictions[1:])):
        if b['frame_id']!=a['frame_id']+1 or not (p and q and p['valid'] and q['valid'] and a['ball_centers'] and b['ball_centers']):continue
        gt=np.array(b['ball_centers'][0])-a['ball_centers'][0]
        delta=np.array([q['x']-p['x'],q['y']-p['y']])
        jumps+=int(np.linalg.norm(delta-gt)>50)
    coverage=[any(c['left']<=r['ball_centers'][0][0]<c['left']+c['side'] and c['top']<=r['ball_centers'][0][1]<c['top']+c['side']
                  for c in records[r['frame_id']].get('roi_transforms',[records[r['frame_id']]['crop_transform']]) if c)
              for r in rows if r['label_status']=='positive']
    return dict(by_tolerance={str(t):aggregate(rows,predictions,t) for t in (10,50)},
                bounce_neighborhood=aggregate([r for r,p in near],[p for r,p in near],50),
                wrong_jumps=jumps,jump_definition='observed displacement minus GT displacement exceeds 50 original pixels on adjacent annotated frames',
                any_roi_coverage=float(np.mean(coverage)) if coverage else None)


def render_errors(corpus,vid,rows,records,annotation,score,dest):
    kinds=defaultdict(list);landing=score['landing']
    for i in landing['false_positives']:kinds[score['predictions'][i]['frame_id']].append('landing_false_positive')
    for i in landing['false_negatives']:kinds[score['ground_truth'][i]['frame_id']].append('landing_false_negative')
    for i in score['continuous_bounce']['false_positives']:
        kinds[score['detected'][i]['frame_id']].append('continuous_false_positive')
    for i in score['continuous_bounce']['false_negatives']:
        kinds[score.get('temporal_ground_truth',score['ground_truth'])[i]['frame_id']].append('continuous_false_negative')
    for row in corpus.rows[vid]:
        if records[row['frame_id']].get('global_candidates'):
            kinds[row['frame_id']].extend(classify_error(row,records[row['frame_id']]))
        else:
            p=records[row['frame_id']]['prediction'];xy=row['ball_centers']
            if row['label_status']=='positive' and (not p or not p['valid'] or np.hypot(p['x']-xy[0][0],p['y']-xy[0][1])>10):
                kinds[row['frame_id']].append('ball_localization_error')
            if row['label_status']=='negative' and p and p['valid']:kinds[row['frame_id']].append('ball_false_positive')
    path=dest/f'{vid}_overlay.mp4';writer=None
    event_gt=defaultdict(list)
    for e in annotation['events']:
        if e['kind']=='bounce' and e.get('spatial_valid',True):event_gt[e['frame_id']].append(e)
    event_probs={r['frame_id']:r['probabilities'][0] for r in rows}
    try:
        for row in corpus.rows[vid]:
            fid=row['frame_id'];rgb=corpus.image(vid,fid).copy()
            for e in event_gt[fid]:cv2.drawMarker(rgb,(round(e['x']),round(e['y'])),(255,255,0),cv2.MARKER_DIAMOND,24,2)
            note=kinds[fid]+[f"Bounce={event_probs[fid]:.3f}", 'event_reviewed' if known(annotation,fid) else 'event_UNKNOWN']
            im=overlay(rgb,row,records[fid],note)
            if writer is None:
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),corpus.videos[vid]['meta']['fps'],(im.shape[1],im.shape[0]))
                if not writer.isOpened():raise OSError(path)
            writer.write(im)
    finally:
        if writer is not None:writer.release()
    return [dict(video_id=vid,frame_id=f,kinds=k,visualization=path.name) for f,k in kinds.items() if k]


def evaluate_pp(checkpoint, annotations, output, device='cpu', scheme='A0', thresholds=None):
    document=json.loads(Path(annotations).read_text());audit=validate_annotations(document,DATA)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=state['config']
    if cfg.get('best_metric')=='pp_landing_macro_f1' and scheme!='A0':
        # A trained comparison checkpoint must use the exact training annotations.
        if identity(json.loads(Path(cfg['pp_annotations']).read_text()))!=audit['annotation_id']:
            raise ValueError('annotation version differs from training configuration')
        from ttnet.events.data import EventCorpus
        from ttnet.events.continuous import attach_pp
        source_corpus=EventCorpus(cfg['dataset']);attach_pp(source_corpus,annotations)
        if source_corpus.identity!=cfg['dataset_id']:raise ValueError('trained annotation/data identity changed')
    dest=Path(output)
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    torch.set_num_threads(cfg.get('torch_threads',4));cv2.setNumThreads(1)
    system=EventSystem(cfg,torch.device(device),initialize=False);system.load_state_dict(state['system']);system.configure('frozen',False)
    corpus=BallCorpus(DATA);sources={r['video_id']:PROJECT/r['video'] for r in jsonl(DATA/'splits/val.jsonl')}
    rows_by_video={};records_by_video={};ann={};performance=[];loc={}
    for a in document['videos']:
        if a['split']!='val':continue
        vid=a['video_id'];meta=corpus.videos[vid]['meta']
        rows,records,speed=predict_video(system,sources[vid],vid,target_fps=None)
        if len(rows)!=meta['n_frames']:raise ValueError('source frame mapping changed')
        rows_by_video[vid]=rows;records_by_video[vid]=records;performance.append(speed)
        ann[vid]=dict(a,resolution=f"{meta['width']}x{meta['height']}")
        loc[vid]=localization_diagnostics(corpus.rows[vid],records,a)
        save_jsonl(dest/f'{vid}_predictions.jsonl',rows);save_jsonl(dest/f'{vid}_trajectory.jsonl',records.values())
    if thresholds is None:
        metrics,sweep=select_continuous_threshold(rows_by_video,ann,cfg['threshold_grid'],cfg['nms_sec'])
    else:metrics=score_continuous(rows_by_video,ann,thresholds[0],cfg['nms_sec']);sweep=None
    errors=[]
    for vid in ann:errors.extend(render_errors(corpus,vid,rows_by_video[vid],records_by_video[vid],ann[vid],metrics['per_video'][vid],dest))
    fps=sum(r['frames'] for r in performance)/sum(r['elapsed_sec'] for r in performance)
    result=dict(status='evaluated',scheme=scheme,seed=cfg['seed'],annotation_id=audit['annotation_id'],
                dataset_id=cfg.get('dataset_id'),checkpoint_sha256=sha(checkpoint),epoch=state.get('epoch'),
                tracker=cfg.get('tracker_type','independent'),event_head=cfg.get('event_head_type','tcn'),
                config=cfg,metrics=metrics,threshold_sweep=sweep,localization=loc,performance=performance,e2e_fps=fps,
                meets_50_fps=fps>=50,benchmark_hardware=torch.cuda.get_device_name() if system.device.type=='cuda' else 'CPU',
                backend='PyTorch',precision='AMP' if cfg['amp'] and system.device.type=='cuda' else 'FP32',
                protocol_id=identity(dict(annotation=audit['annotation_id'],nms_sec=cfg['nms_sec'],frame_tolerance=1,pixel_tolerance=50)),
                implementation={str(p.relative_to(PROJECT)):sha(p) for p in [Path(__file__),PROJECT/'ttnet/src/events/continuous.py',PROJECT/'ttnet/src/events/inference.py']})
    save(dest/'metrics.json',result);save_jsonl(dest/'error_index.jsonl',errors)
    links=''.join(f'<li><a href="{v}_overlay.mp4">{v}</a></li>' for v in ann)
    (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Continuous PP errors</h1><p>Green: ball GT; red: observed ball; yellow diamond: contact GT. Unknown event times are not scored.</p><ul>'+links+'</ul>')
    return result


def report(paths, dest):
    results=[json.loads(Path(p).read_text()) for p in paths]
    if len({r['protocol_id'] for r in results})>1:raise ValueError('cannot compare different labels or postprocessing')
    keys=[(r['scheme'],r['seed']) for r in results]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate scheme/seed results')
    protocol_keys=('epochs','patience','lr','effective_batch','optimizer','sampling_protocol','window',
                   'global_checkpoint','local_checkpoint','target_fps','threshold_grid','mode')
    protocols={identity({k:r['config'].get(k) for k in protocol_keys}) for r in results if r['scheme']!='A0'}
    if len(protocols)>1:raise ValueError('four-way training protocols differ')
    table=[]
    for r,p in zip(results,paths):
        table.append(dict(scheme=r['scheme'],seed=r['seed'],macro_f1=r['metrics']['macro_f1'],fp=r['metrics']['landing']['fp'],
            e2e_fps=r['e2e_fps'],hardware=r['benchmark_hardware'],backend=r['backend'],precision=r['precision'],status=r['status'],result=str(Path(p).resolve())))
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    with (dest/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['scheme','seed','macro_f1','fp','e2e_fps','hardware','backend','precision','status','result']);writer.writeheader();writer.writerows(table)
    seed42={r['scheme']:r for r in table if r['seed']==42}
    complete=all(k in seed42 for k in 'ABCD')
    candidate=max((seed42[k] for k in 'ABCD'),key=lambda r:(r['macro_f1'],-r['fp'],-'ABCD'.index(r['scheme'])))['scheme'] if complete else None
    stability={}
    for name in ('A',candidate):
        if name is None:continue
        scores=[r['macro_f1'] for r in table if r['scheme']==name and r['seed'] in (42,43,44)]
        seeds={r['seed'] for r in table if r['scheme']==name}
        stability[name]=dict(seeds=sorted(seeds),mean=float(np.mean(scores)) if scores else None,std=float(np.std(scores)) if scores else None,complete={42,43,44}<=seeds)
    same_speed_protocol=len({(r['hardware'],r['backend'],r['precision']) for r in table})<=1
    selection=dict(status='pending_multiseed_review' if complete else 'incomplete',seed42_candidate=candidate,
                   retained_default='A',stability=stability,speed_comparable=same_speed_protocol,
                   conclusion='Review per-view regressions, false positives, multi-seed variation and speed before adopting a new model.')
    save(dest/'selection.json',selection)
    lines=['# Four-way event comparison','','| Scheme | Seed | Landing macro F1 | FP | Full FPS | Result |','|---|---:|---:|---:|---:|---|']
    lines.extend(f"| {r['scheme']} | {r['seed']} | {r['macro_f1']:.4f} | {r['fp']} | {r['e2e_fps']:.2f} | {r['result']} |" for r in table)
    lines+=['',selection['conclusion'],f"Status: {selection['status']}; retain A until review."]
    (dest/'comparison.md').write_text('\n'.join(lines)+'\n')
    return selection


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--checkpoint',type=Path,required=True);e.add_argument('--annotations',type=Path,required=True)
    e.add_argument('--output',type=Path,required=True);e.add_argument('--device',default='cpu');e.add_argument('--scheme',required=True)
    e.add_argument('--use-checkpoint-threshold',action='store_true')
    r=sub.add_parser('report');r.add_argument('--results',type=Path,nargs='+',required=True);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='report':print(report(a.results,a.output))
    else:
        thresholds=None
        if a.use_checkpoint_threshold:thresholds=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['thresholds']
        result=evaluate_pp(a.checkpoint,a.annotations,a.output,a.device,a.scheme,thresholds)
        print(json.dumps(dict(macro_f1=result['metrics']['macro_f1'],fps=result['e2e_fps'])))


if __name__=='__main__':main()
