"""Score official JSONL with the documented +/-1 frame, 50px 1:1 protocol.

This is a local accuracy report, not an organizer-provided scoring program.
Optionally render all frames with prediction/label overlays and error indexes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ttnet.events.metrics import landing_match


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def aggregate(rows):
    tp, fp, fn = [sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn')]
    p, r = tp/max(1, tp+fp), tp/max(1, tp+fn)
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r,
                f1=2*p*r/max(1e-12, p+r))


def evaluate(manifest, gt_path, predictions):
    videos = json.loads(Path(manifest).read_text())['videos']
    gt, pred = defaultdict(list), defaultdict(list)
    for row in read_rows(gt_path):
        gt[row['video_id']].append(row)
    for row in read_rows(predictions):
        pred[row['video_id']].append(row)
    by_view = defaultdict(list)
    per_video = {}
    for meta in videos:
        vid = meta['video_id']
        result = landing_match(pred[vid], gt[vid], time_tolerance=1, distance=50)
        per_video[vid] = dict(angle=meta['angle'], **result)
        by_view[meta['angle']].append(result)
    views = {view: aggregate(rows) for view, rows in by_view.items()}
    return dict(macro_f1=sum(r['f1'] for r in views.values()) / len(views),
                by_view=views, total=aggregate(list(per_video.values())), per_video=per_video,
                protocol=dict(frame_tolerance=1, distance_px=50,
                              matching='maximum-cardinality one-to-one; valid GT first; invalid GT absorbs unmatched predictions',
                              macro='unweighted mean over views', organizer_scorer=False))


def compare(prior, current):
    old = {(r['video_id'], r['frame_id']): (r['x'], r['y']) for r in read_rows(prior)}
    new = {(r['video_id'], r['frame_id']): (r['x'], r['y']) for r in read_rows(current)}
    shared = old.keys() & new.keys()
    max_delta = max((max(abs(a-b) for a, b in zip(old[k], new[k])) for k in shared), default=0.)
    return dict(prior=str(prior), prior_count=len(old), current_count=len(new),
                missing=sorted(old.keys()-new.keys()), added=sorted(new.keys()-old.keys()),
                max_coordinate_delta_px=max_delta, identical=old == new)


def render(manifest, gt_path, predictions, input_dir, output_dir, results):
    import cv2
    videos = json.loads(Path(manifest).read_text())['videos']
    gt, pred = defaultdict(list), defaultdict(list)
    for row in read_rows(gt_path):
        if row.get('valid', True):
            gt[row['video_id']].append(row)
    for row in read_rows(predictions):
        pred[row['video_id']].append(row)
    dest = Path(output_dir) / 'overlays'
    dest.mkdir(parents=True, exist_ok=True)
    errors = []
    for meta in videos:
        vid = meta['video_id']
        result = results['per_video'][vid]
        matched_p = {r['prediction'] for r in result['matches']}
        matched_g = {r['ground_truth'] for r in result['matches']}
        overlay = dest / f'{vid}.mp4'
        for kind, indexes, rows in [('FP', result['false_positives'], pred[vid]),
                                     ('FN', result['false_negatives'], gt[vid])]:
            errors.extend(dict(video_id=vid, type=kind, frame_id=rows[i]['frame_id'],
                               x=rows[i]['x'], y=rows[i]['y'], overlay=str(overlay)) for i in indexes)
        cap = cv2.VideoCapture(str(Path(input_dir) / meta['path']))
        w, h = meta['width'], meta['height']
        scale = min(960/w, 540/h)
        size = (int(w*scale)//2*2, int(h*scale)//2*2)
        writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*'mp4v'), meta['fps'], size)
        if not cap.isOpened() or not writer.isOpened():
            raise RuntimeError(f'Cannot render {vid}')
        fid = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            image = cv2.resize(image, size)
            cv2.rectangle(image, (0, 0), (size[0], 65), (0, 0, 0), -1)
            cv2.putText(image, f'{vid} frame {fid} | GT green; pred blue; FP red; FN yellow',
                        (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
            cv2.putText(image, 'Events shown +/-1 frame; localizer confidence is not in official JSONL',
                        (12, 47), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
            for kind, rows, matched in [('GT', gt[vid], matched_g), ('P', pred[vid], matched_p)]:
                for i, row in enumerate(rows):
                    if abs(row['frame_id']-fid) > 1:
                        continue
                    color = ((0, 255, 0) if i in matched else (0, 255, 255)) if kind == 'GT' else ((255, 160, 0) if i in matched else (0, 0, 255))
                    point = (round(row['x']*scale), round(row['y']*scale))
                    cv2.circle(image, point, 10 if kind == 'GT' else 6, color, 2)
                    cv2.putText(image, f'{kind}@{row["frame_id"]}', (point[0]+10, point[1]+(12 if kind == 'GT' else -10)),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            writer.write(image)
            fid += 1
        cap.release()
        writer.release()
        if fid != meta['n_frames']:
            raise ValueError(f'Incomplete overlay video: {vid}:{fid}')
    (Path(output_dir) / 'error_index.json').write_text(json.dumps(errors, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('manifest', 'gt', 'predictions', 'output'):
        parser.add_argument('--' + arg, required=True, type=Path)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--input', type=Path, help='Render overlays from this video root')
    args = parser.parse_args()
    result = evaluate(args.manifest, args.gt, args.predictions)
    if args.prior:
        result['comparison'] = compare(args.prior, args.predictions)
        result['prior_metrics'] = evaluate(args.manifest, args.gt, args.prior)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input:
        render(args.manifest, args.gt, args.predictions, args.input, args.output, result)
    (args.output / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_video', 'prior_metrics')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
