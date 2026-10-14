"""Audit AP interpolation from persisted detections (no new inference)."""
import sys
import json
import numpy as np
from run_yolo_ball import ROOT,save
from ultralytics.utils.metrics import compute_ap


def main():
    exp=sys.argv[1];base=ROOT/'out/experiments'/exp
    rows=[json.loads(s) for s in (base/'eval/score_diagnostic/predictions.jsonl').read_text().splitlines()]
    thresholds=np.linspace(.5,.95,10); entries=[];total=sum(len(r['boxes']) for r in rows)
    for row in rows:
        gt=np.array(row['boxes']);pred=np.array(row['predicted_boxes']).reshape(-1,4);scores=np.array(row['confidence'])
        order=np.argsort(-scores,kind='stable');pred=pred[order];scores=scores[order]
        used=[set() for _ in thresholds]
        for box,score in zip(pred,scores):
            inter=np.maximum(0,np.minimum(gt[:,2:],box[2:])-np.maximum(gt[:,:2],box[:2])).prod(1)
            iou=inter/((gt[:,2:]-gt[:,:2]).prod(1)+(box[2:]-box[:2]).prod()-inter+1e-9)
            tp=[]
            for j,t in enumerate(thresholds):
                eligible=[i for i in range(len(gt)) if i not in used[j] and iou[i]>=t]
                chosen=max(eligible,key=lambda i:iou[i]) if eligible else None
                tp.append(chosen is not None)
                if chosen is not None:used[j].add(chosen)
            entries.append((float(score),tp))
    entries.sort(key=lambda r:-r[0]); flags=np.array([e[1] for e in entries],dtype=float)
    ap=[];legacy=[];maxrec=[]
    for i,t in enumerate(thresholds):
        tp=np.cumsum(flags[:,i]);fp=np.cumsum(1-flags[:,i]);rec=tp/total;prec=tp/(tp+fp)
        envelope=np.maximum.accumulate(prec[::-1])[::-1]
        ap.append(float(np.mean([max(envelope[rec>=r],default=0.) for r in np.linspace(0,1,101)])))
        legacy.append(float(compute_ap(rec,prec)[0]));maxrec.append(float(rec[-1]) if len(rec) else 0.)
    result=dict(protocol='local 101 recall points, max precision for achieved recall>=r, zero beyond maximum achieved recall; score-ordered greedy IoU matching per image; conf floor .001; not full pycocotools COCO protocol',
                predictions=len(entries),ground_truth=total,iou_thresholds=thresholds.tolist(),
                ap_by_iou=ap,ap50=ap[0],ap50_95=float(np.mean(ap)),max_recall_by_iou=maxrec,
                ultralytics_linear_endpoint_ap_by_iou=legacy,
                explanation='Ultralytics 8.3.225 appends (recall=1,precision=0), then linearly interpolates and integrates across the unachieved recall region. With only a few low-score detections this can greatly exceed zero-beyond-achieved-recall AP. Primary library metrics retained for traceability.')
    save(ROOT/'experiments/records'/exp/'evaluations/ap_audit.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
