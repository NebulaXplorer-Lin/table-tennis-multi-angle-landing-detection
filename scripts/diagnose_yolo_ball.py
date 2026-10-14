"""Post-hoc score diagnostics; does not tune or replace the primary evaluation."""
import sys
from pathlib import Path
import json
import time
import numpy as np
import cv2
from run_yolo_ball import ROOT, YOLO, save, match, pr, now, sha


def main():
    exp=sys.argv[1]; record=ROOT/'experiments/records'/exp; dest=ROOT/'out/experiments'/exp/'eval/score_diagnostic'
    dest.mkdir(parents=True,exist_ok=False)
    model=YOLO(str(ROOT/'out/experiments'/exp/'attempts/001/train/weights/best.pt'))
    records=[json.loads(x) for x in (ROOT/'experiments/splits/yolo_ball_v1/val.jsonl').read_text().splitlines()]
    counts={str(t):[0,0,0] for t in [.001,.005,.01,.05,.1,.25]}
    best_scores=[]; bins={}; examples=[]
    with (dest/'predictions.jsonl').open('w') as f:
        for i,r in enumerate(records):
            out=model.predict(str(ROOT/r['image']),imgsz=640,conf=.001,iou=.7,max_det=10,device=0,half=False,verbose=False)[0]
            pred=out.boxes.xyxy.cpu().numpy();scores=out.boxes.conf.cpu().numpy(); gt=np.array(r['boxes'])
            best_scores.append(float(max(scores)) if len(scores) else 0.)
            row=dict(r,predicted_boxes=pred.tolist(),confidence=scores.tolist())
            f.write(json.dumps(row)+'\n')
            for t in counts:
                p=pred[scores>=float(t)];pairs=match(gt,p,10);n=len(pairs)
                counts[t]=[a+b for a,b in zip(counts[t],[n,len(p)-n,len(gt)-n])]
            matches=match(gt,pred,10)
            if matches and r['group'] not in bins:
                bins[r['group']]=True
                im=cv2.imread(str(ROOT/r['image']))
                for b in gt:cv2.rectangle(im,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(0,255,0),2)
                for b,s in zip(pred,scores):
                    cv2.rectangle(im,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(0,0,255),2)
                    cv2.putText(im,f'{s:.4f}',(round(b[0]),max(12,round(b[1])-4)),0,.5,(0,0,255),1)
                name=f'{r["group"]}_matched.jpg';cv2.imwrite(str(dest/name),im);examples.append(name)
            if i%1000==0:print(f'diagnostic {i}/{len(records)}',flush=True)
    result=dict(created=now(),protocol='post-hoc diagnostic on selection val, not independent test; fixed conf floor .001; center distance<=10 original pixels; NMS=.7; max_det=10',
                per_confidence={k:pr(*v) for k,v in counts.items()},
                top_score_percentiles=dict(zip(['p0','p50','p90','p99','max'],np.percentile(best_scores,[0,50,90,99,100]).tolist())),
                no_predictions_at_floor=sum(s==0 for s in best_scores),examples=examples)
    save(record/'evaluations/score_diagnostic.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
