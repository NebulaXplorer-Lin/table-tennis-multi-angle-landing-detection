"""Local approximate scorer, not the organizer's scorer. Standard library only."""
import json
import math
from collections import defaultdict
from typing import TypedDict


class Metrics(TypedDict):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class AngleMetrics(Metrics):
    ignored: int


class EvaluationSummary(TypedDict):
    protocol: str
    caveat: str
    overall: Metrics
    by_angle: dict[str, AngleMetrics]
    macro_f1: float


class EvaluationResult(EvaluationSummary, total=False):
    performance: dict[str, object]
    prediction_sha256: str


def records(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def qualifies(p, g):
    return abs(p['frame_id']-g['frame_id']) <= 1 and math.hypot(p['x']-g['x'], p['y']-g['y']) <= 50


def maximum_match(predictions, truths):
    edges = [[j for j, g in enumerate(truths) if qualifies(p, g)] for p in predictions]
    owners = {}
    def visit(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or visit(owners[j], seen):
                owners[j] = i
                return True
        return False
    for i in range(len(predictions)):
        visit(i, set())
    return set(owners.values())


def measure(tp: int, fp: int, fn: int) -> Metrics:
    return Metrics(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)


def evaluate(pred_path, data) -> EvaluationResult:
    videos = json.loads((data/'manifest.json').read_text(encoding='utf-8'))['videos']
    pred, gt = defaultdict(list), defaultdict(list)
    for r in records(pred_path):
        pred[r['video_id']].append(r)
    for r in records(data/'gt_public.jsonl'):
        gt[r['video_id']].append(r)
    assert set(pred).issubset({v['video_id'] for v in videos}), 'Unknown predicted video'
    counts = defaultdict(lambda: [0, 0, 0, 0])
    for v in videos:
        ps = pred[v['video_id']]
        gs = [g for g in gt[v['video_id']] if g.get('valid', True)]
        invalid = [g for g in gt[v['video_id']] if not g.get('valid', True)]
        matched = maximum_match(ps, gs)
        ignored = sum(i not in matched and any(qualifies(p,g) for g in invalid) for i,p in enumerate(ps))
        for j,value in enumerate((len(matched),len(ps)-len(matched)-ignored,len(gs)-len(matched),ignored)):
            counts[v['angle']][j] += value
    angles = {a: AngleMetrics(ignored=c[3], **measure(*c[:3])) for a,c in counts.items()}
    total = measure(*(sum(c[j] for c in counts.values()) for j in range(3)))
    return EvaluationResult(protocol='local_max_match_f1_frame1_px50_v1',
                caveat='Valid GT matched first; remaining predictions near invalid GT ignored. Official ignore handling may differ.',
                overall=total, by_angle=angles, macro_f1=sum(a['f1'] for a in angles.values())/len(angles))
