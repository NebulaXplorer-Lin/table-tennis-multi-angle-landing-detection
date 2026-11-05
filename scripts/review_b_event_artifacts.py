"""Decode the study's complete error videos and render actual-result figures.

This CPU-only review reads saved predictions; it never runs or changes models.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from ttnet.common import save, sha

ARMS = ('F0', 'geometry', 'concat')


def read(path):
    return json.loads(Path(path).read_text())


def resolve(path):
    return Path(path) if Path(path).is_absolute() else PROJECT/path


def open_video(path):
    return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])


def decode_video(job):
    arm, vid, path, expected = job
    capture = open_video(path)
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); count = 0; shape = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if shape is not None and frame.shape != shape:
            raise ValueError(f'video size changes: {path}')
        shape = frame.shape
        count += 1
    capture.release()
    return dict(arm=arm, video_id=vid, path=str(path), sha256=sha(path),
                declared_frames=declared, decoded_frames=count, expected_frames=expected,
                shape=list(shape) if shape is not None else None,
                passed=count == expected == declared and count > 0)


def plots(task, analysis, state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'F0':'#2673b8', 'geometry':'#d67c18', 'concat':'#30854c'}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for arm in ARMS:
        epochs = {}
        for path in sorted(resolve(state['runs'][arm]).glob('attempts/*/epochs.jsonl')):
            for line in path.read_text().splitlines():
                row = json.loads(line); epochs[row['epoch']] = row
        rows = [epochs[key] for key in sorted(epochs)]
        axes[0].plot([r['epoch'] for r in rows], [r['train']['event'] for r in rows], label=arm, color=colors[arm])
        axes[1].plot([r['epoch'] for r in rows], [r['continuous']['macro_f1'] for r in rows], label=arm, color=colors[arm])
        best = analysis['arms'][arm]['training']['best_epoch']
        axes[1].scatter([best], [epochs[best]['continuous']['macro_f1']], marker='*', s=90, color=colors[arm])
    axes[0].set_ylabel('Train event BCE'); axes[0].set_yscale('log')
    axes[1].set_ylabel('PP validation landing macro F1'); axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.set_xlabel('Epoch'); axis.grid(alpha=.2); axis.legend()
    figure.suptitle(f"B locators + frozen 13-frame TCN (seed{state.get('seed',42)}); internal selection")
    for extension in ('png', 'svg'):
        figure.savefig(task/f'training_curves.{extension}', dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    views = ('side', 'diagonal', 'front'); positions = np.arange(3)
    for i, arm in enumerate(ARMS):
        values = [analysis['arms'][arm]['by_view'][view]['f1'] for view in views]
        axis.bar(positions+(i-1)*.25, values, .24, label=arm, color=colors[arm])
    axis.set_xticks(positions, views); axis.set_ylim(0, 1); axis.set_ylabel('Landing F1')
    axis.set_title(f"Complete PP validation videos; checkpoint thresholds; seed{state.get('seed',42)}")
    axis.legend(); axis.grid(axis='y', alpha=.2)
    for extension in ('png', 'svg'):
        figure.savefig(task/f'validation_views.{extension}', dpi=160)
    plt.close(figure)


def cases(task, analysis):
    events = analysis['paired']['events']; chosen = []; seen = set()
    predicates = (
        ('concat_recovers_F0', lambda e:not e['matched']['F0'] and e['matched']['concat']),
        ('concat_loses_F0', lambda e:e['matched']['F0'] and not e['matched']['concat']),
        ('group22_without_ROI', lambda e:e['group']=='22' and e['no_roi_13window']),
        ('old_FN_exact_frame_repair', lambda e:e['exact_frame_repair']),
    )
    for label, predicate in predicates:
        count = 0
        for event in events:
            key = (event['video_id'], event['frame_id'])
            if key in seen or not predicate(event):
                continue
            chosen.append((label, event)); seen.add(key); count += 1
            if count == 2:
                break
    output = task/'review_cases'; output.mkdir(exist_ok=True)
    rows = []
    for label, event in chosen:
        vid, fid = event['video_id'], event['frame_id']; panels = []
        for arm in ARMS:
            video = Path(analysis['arms'][arm]['errors']['overlays'][vid])
            capture = open_video(video); capture.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = capture.read(); capture.release()
            if not ok:
                raise OSError(f'cannot read {video}:{fid}')
            frame = cv2.resize(frame, (640, round(frame.shape[0]*640/frame.shape[1])))
            frame = cv2.copyMakeBorder(frame, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))
            note = f"{arm} | {vid}:{fid} | event matched={event['matched'][arm]}"
            cv2.putText(frame, note, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            panels.append(frame)
        height = max(p.shape[0] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, height-p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
        path = output/f'{label}_{vid}_{fid:06d}.jpg'
        if not cv2.imwrite(str(path), np.concatenate(panels, axis=1)):
            raise OSError(path)
        rows.append(dict(label=label, event=event, path=str(path)))
    cards = ''.join(f'<p>{r["label"]} {r["event"]["video_id"]}:{r["event"]["frame_id"]}</p><a href="{Path(r["path"]).name}"><img width="1200" src="{Path(r["path"]).name}"></a>' for r in rows)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Saved complete-video overlays: paired review</h1><p>GREEN ball GT; RED observed ball; YELLOW contact GT. PINK boxes are ball-localizer search crops; the no-ROI case label refers to missing YOLO table ROI only. Match refers to original-frame ±1 / 50px event scoring. Cases selected by deterministic rules; all errors remain in complete-video indexes.</p>'+cards)
    return rows


def review(task):
    cv2.setNumThreads(1)
    task = resolve(task); analysis = read(task/'analysis.json'); state = read(task/'task.json')
    jobs = []
    for arm in ARMS:
        directory = Path(analysis['arms'][arm]['evaluation']).parent
        for vid, video in analysis['arms'][arm]['errors']['overlays'].items():
            expected = sum(1 for _ in (directory/f'{vid}_predictions.jsonl').open())
            jobs.append((arm, vid, Path(video), expected))
    with ThreadPoolExecutor(max_workers=4) as pool:
        videos = list(pool.map(decode_video, jobs))
    result = dict(status='passed' if all(v['passed'] for v in videos) else 'failed',
                  scope='Every saved full-video overlay decoded frame by frame; compare prediction row count. Human semantic image review is separate.',
                  videos=videos, total_decoded_frames=sum(v['decoded_frames'] for v in videos))
    save(task/'visual_qa.json', result)
    if result['status'] != 'passed':
        raise ValueError('incomplete or undecodable overlay videos; see visual_qa.json')
    plots(task, analysis, state)
    save(task/'review_cases.json', cases(task, analysis))
    print(json.dumps(dict(status='passed', videos=len(videos), frames=result['total_decoded_frames'], task=str(task))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    review(parser.parse_args().task)
