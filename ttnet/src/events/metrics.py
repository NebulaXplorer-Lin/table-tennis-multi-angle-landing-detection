"""Sparse anchor classification is distinct from continuous landing evaluation."""
from __future__ import annotations

import numpy as np


def binary_metrics(truth, predicted):
    truth=np.asarray(truth,bool);predicted=np.asarray(predicted,bool)
    tp=int((truth&predicted).sum()); fp=int((~truth&predicted).sum());fn=int((truth&~predicted).sum())
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn)
    return dict(tp=tp,fp=fp,fn=fn,tn=int((~truth&~predicted).sum()),precision=p,recall=r,f1=2*p*r/max(1e-12,p+r))


def classification(rows, thresholds):
    result={kind:binary_metrics([r['target'][k]>.5 for r in rows if r.get('supervised',[True,True])[k]],
                               [r['probabilities'][k]>=thresholds[k] for r in rows if r.get('supervised',[True,True])[k]])
            for k,kind in enumerate(('bounce','net'))}
    result['macro_f1']=sum(result[k]['f1'] for k in ('bounce','net'))/2
    confusion=np.zeros((3,3),int)
    for r in rows:
        if not all(r.get('supervised',[True,True])):continue
        truth=int(np.argmax(r['target'])) if max(r['target'])>.5 else 2
        accepted=[k for k in range(2) if r['probabilities'][k]>=thresholds[k]]
        pred=max(accepted,key=lambda k:r['probabilities'][k]) if accepted else 2
        confusion[truth,pred]+=1
    result['confusion_bounce_net_empty']=confusion.tolist()
    result['samples']=len(rows)
    return result


def select_thresholds(rows, grid):
    thresholds=[]
    for k in range(2):
        known=[r for r in rows if r.get('supervised',[True,True])[k]]
        best=max(grid,key=lambda t:(binary_metrics([r['target'][k]>.5 for r in known],
                                                   [r['probabilities'][k]>=t for r in known])['f1'],t))
        thresholds.append(best)
    return thresholds


def peaks(rows, thresholds, nms_sec=.08):
    result=[]
    for k,kind in enumerate(('bounce','net')):
        candidates=[]
        for i,r in enumerate(rows):
            p=r['probabilities'][k]
            if p<thresholds[k]:continue
            if i and rows[i-1]['frame_id']==r['frame_id']-1 and rows[i-1]['probabilities'][k]>p:continue
            if i+1<len(rows) and rows[i+1]['frame_id']==r['frame_id']+1 and rows[i+1]['probabilities'][k]>p:continue
            candidates.append(dict(r,kind=kind,confidence=p))
        chosen=[]
        for r in sorted(candidates,key=lambda x:(-x['confidence'],x['frame_id'])):
            if all(abs(r['timestamp_sec']-c['timestamp_sec'])>nms_sec+1e-9 for c in chosen):chosen.append(r)
        result.extend(chosen)
    return sorted(result,key=lambda r:(r['frame_id'],r['kind']))


def landing_match(predictions, gt, time_tolerance=1, distance=50):
    """Maximum-cardinality one-to-one matching; valid GT before ignored distractors."""
    valid=[g for g in gt if g.get('valid',True)];ignored=[g for g in gt if not g.get('valid',True)]
    def near(p,g):return abs(p['frame_id']-g['frame_id'])<=time_tolerance and np.hypot(p['x']-g['x'],p['y']-g['y'])<=distance
    edges={i:sorted([j for j,g in enumerate(valid) if near(p,g)],
                    key=lambda j:(abs(p['frame_id']-valid[j]['frame_id']),np.hypot(p['x']-valid[j]['x'],p['y']-valid[j]['y'])))
           for i,p in enumerate(predictions)}
    owners={}
    def augment(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owners or augment(owners[j],seen):owners[j]=i;return True
        return False
    for i in range(len(predictions)):augment(i,set())
    matched=set(owners.values());ignored_ids={i for i,p in enumerate(predictions) if i not in matched and any(near(p,g) for g in ignored)}
    tp=len(owners);fp=len(predictions)-tp-len(ignored_ids);fn=len(valid)-tp
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn)
    return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(1e-12,p+r),ignored=len(ignored_ids),
                matches=[dict(prediction=i,ground_truth=j) for j,i in sorted(owners.items())],
                false_positives=[i for i in range(len(predictions)) if i not in matched|ignored_ids],
                false_negatives=[j for j in range(len(valid)) if j not in owners])
