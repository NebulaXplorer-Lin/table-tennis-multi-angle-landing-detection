from __future__ import annotations

from collections import defaultdict
import numpy as np
from ttnet.tracking.independent import decode_heatmap


def accept(pred, threshold, resolution):
    if pred is None:
        return None
    x,y,c = pred["x"],pred["y"],pred["confidence"]
    valid = bool(np.isfinite([x,y,c]).all() and c>=threshold and 0<=x<resolution["width"] and 0<=y<resolution["height"])
    return dict(pred,valid=valid)


def aggregate(rows, predictions, tolerance=10):
    tp=fp=fn=ignored=0
    matched,accepted_errors,miss_runs = [],[],[]
    run,last_video,last_frame = 0,None,-2
    roi_inside=roi_total=0
    for row,p in zip(rows,predictions):
        status=row["label_status"]
        if status not in ["positive","negative"]:
            ignored+=1; run=0
            last_video,last_frame=row["video_id"],row["frame_id"]
            continue
        positive=status=="positive"
        accepted=bool(p and p["valid"])
        error=float(np.linalg.norm(np.asarray(row["ball_centers"][0])-[p["x"],p["y"]])) if positive and accepted else None
        hit=positive and accepted and error<=tolerance
        tp+=int(hit); fp+=int(accepted and not hit); fn+=int(positive and not hit)
        if error is not None: accepted_errors.append(error)
        if hit: matched.append(error)
        contiguous=row["video_id"]==last_video and row["frame_id"]==last_frame+1
        run=(run+1 if contiguous else 1) if positive and not hit else 0
        miss_runs.append(run)
        last_video,last_frame=row["video_id"],row["frame_id"]
        if positive:
            roi_total+=1
            if p and p.get("crop_transform"):
                t=p["crop_transform"]; x,y=row["ball_centers"][0]
                roi_inside+=int(t["left"]<=x<t["left"]+t["side"] and t["top"]<=y<t["top"]+t["side"])
    precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
    quantile=lambda values,q:float(np.percentile(values,q)) if values else None
    return dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=2*precision*recall/max(1e-12,precision+recall),
                matched_coverage=recall,error_median_px=quantile(matched,50),error_p90_px=quantile(matched,90),
                accepted_error_median_px=quantile(accepted_errors,50),accepted_error_p90_px=quantile(accepted_errors,90),
                max_consecutive_missed=max(miss_runs,default=0),ignored_frames=ignored,
                roi_coverage=roi_inside/max(1,roi_total) if roi_total else None,bbox_ap=None)


def select_threshold(rows, candidates, tolerance, globals_=None, global_threshold=None):
    choices=[]
    for threshold in [.01,.02,.05,.1,.2,.3,.5,.7,.9]:
        predictions=[]
        for i,(r,p) in enumerate(zip(rows,candidates)):
            local=accept(p,threshold,r["resolution"])
            if globals_ is not None:
                g=accept(globals_[i],global_threshold,r["resolution"])
                chosen=local if local and local["valid"] else g
                if chosen is not None:
                    chosen=dict(chosen,crop_transform=p.get("crop_transform") if p else None)
            else: chosen=local
            predictions.append(chosen)
        metrics=aggregate(rows,predictions,tolerance)
        error=metrics["accepted_error_median_px"]
        choices.append((metrics["f1"],-(error if error is not None else 1e9),threshold,metrics,predictions))
    best=max(choices,key=lambda x:(x[0],x[1],x[2]))
    return dict(threshold=best[2],metrics=best[3],predictions=best[4],
                sweep=[dict(threshold=x[2],metrics=x[3]) for x in choices])
