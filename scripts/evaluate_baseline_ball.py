"""Evaluate the unchanged baseline BallDetector on the frozen ball_v1 videos.

Run with .venv/bin/python scripts/evaluate_baseline_ball.py. No training or
landing-event inference. Large artifacts go to out/, experiment records remain
under experiments/. Unknown labels are never counted as background.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import cv2
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pingpang'))
from participant.ball_detector import BallDetector, SCORE_TH, OUT_CHANNEL
from participant.solution import CONF_TH

SPLIT = ROOT / 'experiments/splits/yolo_ball_v1'
MODEL = ROOT / 'pingpang/participant/weights/ball.onnx'
EVAL = 'onnx_fp32_ball'


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def rates(tp, fp, fn):
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r,
                f1=2 * p * r / max(p + r, 1e-12), coverage=r)


def score(boxes, pred, distance, confidence):
    """One output point: nearest valid GT is max-cardinality/min-distance match."""
    accepted = pred is not None and pred['conf'] >= confidence
    centers = [(np.array(b[:2]) + b[2:]) / 2 for b in boxes]
    errors = [float(np.linalg.norm(c - [pred['x'], pred['y']])) for c in centers] if accepted else []
    nearest = min(errors) if errors else None
    tp = int(accepted and nearest is not None and nearest <= distance)
    return dict(tp=tp, fp=int(accepted) - tp, fn=len(boxes) - tp,
                nearest_error_px=nearest, matched_error_px=nearest if tp else None)


def aggregate(rows, distance=10, confidence=SCORE_TH):
    counts = [0, 0, 0]
    matched = []
    longest = run = 0
    last_video, last_frame = None, -2
    for row in rows:
        # Missing ball annotations are unknown, not negative examples.
        if row['label_status'] != 'positive':
            run = 0
            continue
        m = score(row['boxes'], row['prediction'], distance, confidence)
        counts = [a + m[k] for a, k in zip(counts, ['tp', 'fp', 'fn'])]
        if m['tp']:
            matched.append(m['matched_error_px'])
        contiguous = row['video'] == last_video and row['frame'] == last_frame + 1
        if m['fn']:
            run = run + 1 if contiguous else 1
        else:
            run = 0
        longest = max(longest, run)
        last_video, last_frame = row['video'], row['frame']
    return dict(rates(*counts), error_median_px=float(np.median(matched)) if matched else None,
                error_p90_px=float(np.percentile(matched, 90)) if matched else None,
                max_consecutive_missed=longest)


def overlay(frame, row):
    im = frame.copy()
    gt_color, pred_color = (70, 240, 90), (80, 80, 255)
    for b in row['boxes']:
        cv2.rectangle(im, tuple(map(round, b[:2])), tuple(map(round, b[2:])), gt_color, 2)
        c = tuple(round(v) for v in ((np.array(b[:2]) + b[2:]) / 2))
        cv2.drawMarker(im, c, gt_color, cv2.MARKER_CROSS, 18, 2)
    pred = row['prediction']
    if pred is not None:
        point = (round(pred['x']), round(pred['y']))
        cv2.circle(im, point, 10, pred_color, 2)
        cv2.putText(im, f"pred {pred['conf']:.3f}", (max(0, point[0]-30), max(90, point[1]-15)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, pred_color, 2)
    m = row['metrics_10px']
    status = 'UNKNOWN LABEL' if m is None else ('MISS' if pred is None else ('TP <=10px' if m['tp'] else 'LOCATION ERROR: FP+FN'))
    error = '' if m is None or m['nearest_error_px'] is None else f" | error {m['nearest_error_px']:.2f}px"
    cv2.rectangle(im, (0, 0), (im.shape[1], 83), (12, 22, 30), -1)
    name = '/'.join(Path(row['video']).parts[-2:])
    cv2.putText(im, f'{name} | frame {row["frame"]} | {status}{error}', (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
    cv2.putText(im, 'GREEN: GT box/center | RED: predicted center | heat >0.7 | aligned input frame-1',
                (12, 63), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    if row['tail_unavailable']:
        cv2.putText(im, 'FINAL FRAME: no future input in baseline (counted as missing)', (12, 111),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (80, 200, 255), 2)
    return im


def encoder(path, w, h, fps, log):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
           '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
           '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
           '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)


def evaluate(det, audits, labels, dest):
    rows, per_video, errors, examples = [], {}, [], []
    unknown = startup_discarded = 0
    with (dest / 'predictions.jsonl').open('w') as predictions, (dest / 'encoder.log').open('w') as encoder_log:
        for index, audit in enumerate(audits):
            video = audit['video']
            cap = cv2.VideoCapture(str(ROOT / video))
            assert cap.isOpened(), video
            w, h = audit['width'], audit['height']
            assert (int(cap.get(3)), int(cap.get(4))) == (w, h)
            name = f"{audit['group']}_{Path(video).stem}"
            visual = dest / 'videos' / f'{name}.mp4'
            writer = encoder(visual, w, h, cap.get(5) or 25, encoder_log)
            det.reset()
            previous, fid, video_rows = None, 0, []
            sample_types = set()

            def emit(frame, frame_id, pred, tail=False):
                nonlocal unknown
                boxes = labels.get((video, frame_id), [])
                row = dict(video=video, frame=frame_id, width=w, height=h, boxes=boxes,
                           label_status='positive' if boxes else 'unknown', prediction=pred,
                           input_frame=None if tail else frame_id + 1, tail_unavailable=tail)
                m = score(boxes, pred, 10, SCORE_TH) if boxes else None
                row['metrics_10px'] = m
                predictions.write(json.dumps(row) + '\n')
                rows.append(row)
                video_rows.append(row)
                painted = overlay(frame, row)
                writer.stdin.write(painted.tobytes())
                if boxes:
                    kind = 'miss' if pred is None else ('tp' if m['tp'] else 'location_error')
                    if m['fp'] or m['fn']:
                        errors.append(dict(video=video, frame=frame_id, kind=kind,
                                           nearest_error_px=m['nearest_error_px'],
                                           visualization=str(visual.relative_to(ROOT)),
                                           time_seconds=frame_id/(cap.get(5) or 25), tail_unavailable=tail))
                    if kind not in sample_types:
                        sample_types.add(kind)
                        image = dest / 'examples' / f'{name}_{frame_id:06d}_{kind}.jpg'
                        image.parent.mkdir(parents=True, exist_ok=True)
                        assert cv2.imwrite(str(image), painted)
                        examples.append(dict(video=video, frame=frame_id, kind=kind,
                                             image=str(image.relative_to(dest))))
                else:
                    unknown += 1

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    pred = det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), w, h)
                    if pred is not None:
                        assert all(np.isfinite(pred[k]) for k in ['x', 'y', 'conf'])
                        assert 0 <= pred['x'] < w and 0 <= pred['y'] < h and SCORE_TH < pred['conf'] <= 1
                    if previous is not None:
                        emit(previous, fid-1, pred)
                    elif pred is not None:
                        startup_discarded += 1  # output ch1 refers to the prefilled blank
                    previous = frame
                    fid += 1
                assert fid == audit['frames'], (video, fid, audit['frames'])
                if previous is not None:
                    # Official detector/finish_video do not infer a future blank frame.
                    emit(previous, fid-1, None, tail=True)
            finally:
                cap.release()
                writer.stdin.close()
                result = writer.wait()
            assert result == 0, visual
            assert len(video_rows) == fid
            per_video[video] = dict(frames=fid, labelled=sum(bool(r['boxes']) for r in video_rows),
                                   center={str(t): aggregate(video_rows, t) for t in [10, 50]},
                                   visualization=str(visual.relative_to(ROOT)))
            print(f"video {index+1}/{len(audits)} {name}: {fid} frames; "
                  f"10px F1={per_video[video]['center']['10']['f1']:.4f}", flush=True)
    save(dest / 'per_video.json', per_video)
    save(dest / 'errors.json', errors)
    save(dest / 'examples.json', examples)
    by_group = {group: {str(t): aggregate([r for r in rows if Path(r['video']).parent.name == group], t)
                       for t in [10, 50]} for group in sorted({a['group'] for a in audits})}
    save(dest / 'per_group.json', by_group)
    items = ''.join(f'<figure><a href="{html.escape(e["image"])}"><img loading="lazy" src="{html.escape(e["image"])}"></a>'
                    f'<figcaption>{html.escape(e["video"])} / frame {e["frame"]}: {e["kind"]}</figcaption></figure>' for e in examples)
    links = ''.join(f'<li><a href="videos/{Path(v["visualization"]).name}">{html.escape(k)}</a> '
                    + (f'— {v["labelled"]} labelled frames; F1(10px) {v["center"]["10"]["f1"]:.3f}'
                       if v['labelled'] else '— 无球标注，未评分（不能作为无球负样本）')
                    + '</li>' for k, v in per_video.items())
    (dest / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>Baseline 小球定位</title><style>body{font-family:system-ui;background:#111d28;color:#eef3fa;'
        'max-width:1280px;margin:24px auto;padding:20px}a{color:#77bcff}img{width:100%}figure{margin:30px 0}</style>'
        '<h1>Baseline 小球定位验证</h1><p>绿色框/十字为 GT，红色圆为预测球心。热图阈值 &gt;0.7，'
        '中间帧输出已对齐到输入帧号−1。未知标注帧不计 FP；末帧没有下一帧输入，按无输出评估。'
        '以下叠加视频覆盖所有验证帧和错误；图片每视频抽取首个成功、漏检和定位错误。</p><ul>' + links + '</ul>' + items)
    return dict(center={str(t): aggregate(rows, t) for t in [10, 50]},
                downstream_conf075={str(t): aggregate(rows, t, CONF_TH) for t in [10, 50]},
                decoded_frames=len(rows), labelled_frames=sum(bool(r['boxes']) for r in rows),
                unknown_frames=unknown, tail_labelled_frames=sum(r['tail_unavailable'] and bool(r['boxes']) for r in rows),
                startup_predictions_discarded=startup_discarded, error_frames=len(errors),
                no_output_labelled_frames=sum(bool(r['boxes']) and r['prediction'] is None for r in rows),
                per_group=by_group, known_background_false_positives=None, bbox_ap=None, touch_metrics=None)


def benchmark(session, det, audits, dest):
    dest.mkdir(parents=True, exist_ok=True)
    # Device-bound input/output exclude H2D/D2H; synchronized wall timing includes dispatch overhead.
    x = ort.OrtValue.ortvalue_from_numpy(np.zeros((1, 9, 288, 512), np.float32), 'cuda', 0)
    y = ort.OrtValue.ortvalue_from_shape_and_type([1, 3, 288, 512], np.float32, 'cuda', 0)
    binding = session.io_binding()
    binding.bind_ortvalue_input(session.get_inputs()[0].name, x)
    binding.bind_ortvalue_output(session.get_outputs()[0].name, y)
    for _ in range(50):
        session.run_with_iobinding(binding)
    torch.cuda.synchronize()
    times = []
    for repeat in range(3):
        for _ in range(200):
            torch.cuda.synchronize()
            start = time.perf_counter()
            session.run_with_iobinding(binding)
            torch.cuda.synchronize()
            times.append(dict(repeat=repeat, ms=1000*(time.perf_counter()-start)))
    save(dest / 'latencies.json', times)
    selected = {}
    for a in audits:
        selected.setdefault(a['group'], a)
    durations = []
    for a in selected.values():
        det.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        cap = cv2.VideoCapture(str(ROOT / a['video']))
        assert cap.isOpened()
        fid = 0
        with (dest / f'e2e_{a["group"]}.jsonl').open('w') as output:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                p = det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), a['width'], a['height'])
                output.write(json.dumps(dict(input_frame=fid, output_frame=fid-1, prediction=p))+'\n')
                fid += 1
        cap.release()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        assert fid == a['frames']
        durations.append(dict(video=a['video'], frames=fid, seconds=elapsed, fps=fid/elapsed))
    save(dest / 'e2e_videos.json', durations)
    compute = json.loads((ROOT / 'docs/ball_onnx_graph.json').read_text())['summary']
    assert compute['sha256'] == sha(MODEL)
    return dict(gpu=torch.cuda.get_device_name(0), backend='ONNX Runtime CUDA', precision='FP32 (CUDA EP default use_tf32=1)',
                shape=[1, 9, 288, 512], batch=1, warmup=50, repeats=3, iterations=200,
                model_p50_ms=float(np.median([r['ms'] for r in times])),
                model_p95_ms=float(np.percentile([r['ms'] for r in times], 95)),
                timing_scope='GPU-bound IO; device-synchronized perf_counter; includes CPU dispatch; excludes transfers/preprocessing',
                e2e_fps=sum(r['frames'] for r in durations)/sum(r['seconds'] for r in durations),
                e2e_scope='six complete validation videos: CPU decode, BGR/RGB, normalization, stack, H2D, ORT, D2H, decode heatmap, JSONL write; no overlays/init',
                peak_memory_mib=None, peak_memory_reason='PyTorch allocator cannot measure ORT CUDA arena; not measured',
                params_m=compute['initializer_elements']/1e6, gmac=compute['conv_matmul_macs_batch1']/1e9,
                gflops=compute['conv_matmul_flops_2_per_mac']/1e9,
                compute_tool='local ONNX graph inspector; batch=1', compute_scope=compute['mac_exclusions'])


def comparison(exp, status, result=None):
    result = result or {}
    path = ROOT / 'experiments/comparison.csv'
    rows = list(csv.DictReader(path.open()))
    keys = list(rows[0])
    row = dict.fromkeys(keys, '')
    c, b = result.get('center', {}).get('10', {}), result.get('benchmark', {})
    row.update(experiment_id=exp, eval_id=EVAL, status=status, task='ball_localisation',
               parent='20261013-163117_yolo11n_ball-640', change='official heatmap baseline ball-only evaluation',
               data_version='pp_train_data annotated positives', split_id='yolo_ball_v1', protocol='ball_v1_center',
               model='official HRNet-style heatmap', checkpoint=str(MODEL.relative_to(ROOT)), checkpoint_sha256=sha(MODEL),
               shape='1x9x288x512', window=3, stride=1, batch=1, center_p=c.get('precision', ''),
               center_r=c.get('recall', ''), center_f1=c.get('f1', ''), center_median_px=c.get('error_median_px', ''),
               center_p90_px=c.get('error_p90_px', ''), precision='FP32 (TF32 enabled)',
               config=f'experiments/records/{exp}/config.yaml', log=f'out/experiments/{exp}/attempts/001/console.log',
               evaluation=f'experiments/records/{exp}/evaluations/{EVAL}.json' if result else '',
               predictions=f'out/experiments/{exp}/eval/{EVAL}/predictions.jsonl' if result else '',
               notes='heatmap >.7; all pixels weighted centroid; ch1 aligned -1; tail no output; unknown excluded; no bbox AP; existing frozen split; source pretraining unknown')
    for key in ['gpu', 'backend', 'model_p50_ms', 'model_p95_ms', 'e2e_fps', 'params_m', 'gmac', 'gflops', 'compute_tool', 'compute_scope']:
        row[key] = b.get(key, '')
    rows = [r for r in rows if (r['experiment_id'], r['eval_id']) != (exp, EVAL)] + [row]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue())
    cols = ['experiment_id', 'eval_id', 'status', 'ap50', 'ap50_95', 'center_f1', 'model_p50_ms', 'e2e_fps', 'params_m', 'gflops']
    (ROOT / 'experiments/comparison.md').write_text('# 球定位实验\n\nAP / P / R / F1 存 0–1。球心均以原图 10px 一对一匹配；不同模型的置信度含义不同。热图模型不产生 bbox，AP 留空。YOLO 主行的库 AP 与诊断 AP 积分不同，不能直接横比。速度使用各自原生输入与后端。\n\n| '
        + ' | '.join(cols) + ' |\n|' + '|'.join(['---']*len(cols)) + '|\n'
        + '\n'.join('| ' + ' | '.join(str(r[k]) or '—' for k in cols) + ' |' for r in rows) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id', default=now().strftime('%Y%m%d-%H%M%S')+'_baseline_ball-position')
    args = parser.parse_args()
    exp = args.experiment_id
    assert Path(exp).name == exp
    record, out = ROOT / 'experiments/records' / exp, ROOT / 'out/experiments' / exp
    record.mkdir(parents=True, exist_ok=False)
    attempt = out / 'attempts/001'
    attempt.mkdir(parents=True, exist_ok=True)
    # Tee console output when run directly, without needing shell redirection.
    class Tee:
        def __init__(self, original, log):
            self.original, self.log = original, log
        def write(self, s):
            self.original.write(s)
            self.log.write(s)
        def flush(self):
            self.original.flush()
            self.log.flush()
    log = (attempt / 'console.log').open('w', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    state = dict(status='running', started=now().isoformat(), command=[sys.executable]+sys.argv,
                 cwd=str(ROOT), pid=os.getpid(), python=sys.version, torch=torch.__version__,
                 cuda=torch.version.cuda, onnxruntime=ort.__version__, opencv=cv2.__version__, numpy=np.__version__,
                 git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                 model_sha256=sha(MODEL), uv_lock_sha256=sha(ROOT/'uv.lock'),
                 split_hashes={s: sha(SPLIT/f'{s}.jsonl') for s in ['train', 'val']})
    save(record/'run.json', state)
    (record/'config.yaml').write_text(json.dumps(dict(purpose='baseline ball-only validation',
        checkpoint=str(MODEL.relative_to(ROOT)), checkpoint_sha256=sha(MODEL),
        split_id='yolo_ball_v1', split_hashes=state['split_hashes'], input_shape=[1,9,288,512],
        heat_threshold=SCORE_TH, frame_alignment='ch1 -> input frame-1', train=False), indent=2)+'\n')
    comparison(exp, 'running')
    try:
        cv2.setNumThreads(1)
        assert torch.cuda.is_available()
        baseline_hashes = {p.name: sha(p) for p in [ROOT/'pingpang/participant/ball_detector.py', ROOT/'pingpang/participant/solution.py']}
        state['source_sha256'] = dict(baseline_hashes, evaluator=sha(__file__))
        shutil.copy2(__file__, attempt/Path(__file__).name)
        for name in baseline_hashes:
            shutil.copy2(ROOT/'pingpang/participant'/name, attempt/name)
        (attempt/'workspace.patch').write_text(subprocess.check_output(['git', 'diff', '--no-ext-diff'], cwd=ROOT, text=True))
        previous = json.loads((ROOT/'experiments/records/20261013-163117_yolo11n_ball-640/run.json').read_text())
        assert state['split_hashes'] == previous['split_hashes'], 'Frozen split changed'
        audits = json.loads((SPLIT/'audit.json').read_text())
        train_videos = {a['video'] for a in audits if a['split']=='train'}
        audits = [a for a in audits if a['split']=='val']
        assert not train_videos.intersection(a['video'] for a in audits)
        labels = {}
        for line in (SPLIT/'val.jsonl').read_text().splitlines():
            r = json.loads(line)
            key = (r['video'], r['frame'])
            assert key not in labels
            labels[key] = r['boxes']
        labelled_videos = set(v for v, _ in labels)
        assert labelled_videos <= {a['video'] for a in audits}
        assert all(a['selected'] == 0 for a in audits if a['video'] not in labelled_videos)
        for a in audits:
            assert sha(ROOT/a['video']) == a['video_sha256']
            ann = ROOT/'data/pp_train_data/annotations'/a['group']/(Path(a['video']).stem+'.json')
            assert sha(ann) == a['annotation_sha256']
        config = dict(purpose='test unchanged baseline ball localization only',
                      control='YOLO11n pilot on same frozen val manifest; model/backend/input differ',
                      model='official HRNet-style heatmap, pretrained origin/epoch unavailable',
                      checkpoint=str(MODEL.relative_to(ROOT)), checkpoint_sha256=sha(MODEL),
                      input_shape=[1,9,288,512], window=3, stride=1, batch=1, train=False,
                      optimizer=None, epochs=None, seed=None, augmentation=None, precision='float32; CUDA EP default TF32',
                      split_id='yolo_ball_v1', split_hashes=state['split_hashes'],
                      train_samples=3162, val_samples=len(labels), val_videos=len(audits),
                      decode='OpenCV CPU, full original videos, BGR to RGB',
                      heat_threshold=SCORE_TH, downstream_confidence=CONF_TH, output_channel=OUT_CHANNEL,
                      center_thresholds_px=[10,50], matching='one predicted center: nearest GT; maximum cardinality, minimum distance',
                      frame_alignment='input f -> output f-1; discard startup -1; final frame missing, no synthetic future frame',
                      unknown_policy='no ball box means unknown; exclude from precision/recall; break miss runs',
                      bbox_ap=None, NMS=None, initialization='det.reset per video; two normalized black frames',
                      forward_benchmark=dict(warmup=50,repeats=3,iterations=200,gpu_bound_io=True,synchronized=True))
        # JSON is valid YAML and avoids adding dependencies to the local environment.
        (record/'config.yaml').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
        print(f'Checking {len(audits)} complete videos / {len(labels)} labelled frames; split hashes match YOLO', flush=True)
        ort.preload_dlls()
        start = time.perf_counter()
        session = ort.InferenceSession(str(MODEL), providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        assert session.get_providers()[0]=='CUDAExecutionProvider'
        session.disable_fallback()
        def infer(x):
            return session.run(None, {session.get_inputs()[0].name:x})[0]
        sample = infer(np.zeros((1,9,288,512),np.float32))
        assert sample.shape == (1,3,288,512) and np.isfinite(sample).all()
        det = BallDetector(infer, MODEL)
        det.detect(np.zeros((288,512,3), np.uint8),512,288)
        state.update(initialization_seconds=time.perf_counter()-start,
                     gpu=torch.cuda.get_device_name(0), provider_options=session.get_provider_options())
        config['provider_options'] = state['provider_options']
        (record/'config.yaml').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
        save(record/'run.json',state)
        dest = out/'eval'/EVAL
        dest.mkdir(parents=True)
        result = evaluate(det,audits,labels,dest)
        assert result['labelled_frames']==len(labels)
        result.update(checkpoint=str(MODEL.relative_to(ROOT)), checkpoint_sha256=sha(MODEL),
                      protocol=config, predictions=str((dest/'predictions.jsonl').relative_to(ROOT)),
                      visualizations=str((dest/'index.html').relative_to(ROOT)))
        save(record/'evaluations'/f'{EVAL}.json', result)
        print('All validation videos evaluated. Running 50 warmups + 3x200 forward benchmark and six full E2E videos.', flush=True)
        result['benchmark'] = benchmark(session,det,audits,out/'benchmark'/EVAL)
        for name, digest in baseline_hashes.items():
            assert sha(ROOT/'pingpang/participant'/name)==digest
        subprocess.run([sys.executable,str(ROOT/'pingpang/integrity.py'),'--verify'],check=True)
        save(record/'evaluations'/f'{EVAL}.json',result)
        state.update(status='completed',finished=now().isoformat(),exit_code=0)
        save(record/'run.json',state)
        comparison(exp,'completed',result)
        c, c50, b = result['center']['10'], result['center']['50'], result['benchmark']
        summary = f'''# Baseline 小球位置检测验证

使用原始 ball.onnx 和未改动的 BallDetector，仅执行球定位。三帧 RGB 输入 1×9×288×512、FP32（CUDA EP 默认允许 TF32）、热图 >0.7，全部阈值以上像素的加权质心；ch1 对齐输入帧号−1。未运行触桌/落点规则。

## 数据与结果

复用固定 yolo_ball_v1 验证清单，{len(audits)} 个完整视频，{result['decoded_frames']} 个原始帧，{result['labelled_frames']} 个有球标注帧，{result['unknown_frames']} 个未知帧不计精度。train/val 与已有实验清单 SHA256 一致，视频和标注身份已核对。沿用已有场景分组 pilot，未重新划分。baseline 权重的训练来源和是否包含这些视频未知，因此这是本数据上的验证对照，不能据此证明独立泛化。

| 原图球心距离 | TP | FP | FN | Precision | Recall | F1 | 匹配误差中位数 / P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≤10px | {c['tp']} | {c['fp']} | {c['fn']} | {c['precision']:.6f} | {c['recall']:.6f} | {c['f1']:.6f} | {c['error_median_px']:.3f} / {c['error_p90_px']:.3f}px |
| ≤50px | {c50['tp']} | {c50['fp']} | {c50['fn']} | {c50['precision']:.6f} | {c50['recall']:.6f} | {c50['f1']:.6f} | {c50['error_median_px']:.3f} / {c50['error_p90_px']:.3f}px |

10px 匹配覆盖率 {c['coverage']:.2%}；最长连续未匹配 {c['max_consecutive_missed']} 帧（未知帧处断开）。{result['no_output_labelled_frames']} 个有标签帧没有输出，另有 {c['fp']} 个输出位置未匹配。所有 {result['tail_labelled_frames']} 个有标签末帧均计漏检，因为官方路径没有未来帧输入，不人为补黑帧。中间帧预测不使用当前帧 GT。输出最多一个球心，远离全部 GT 的一个输出同时贡献 FP 和 FN。

热图没有 bbox，mAP 留空；没有可靠无球标签，无球误报率留空。下游 conf≥0.75 的定位诊断另存 evaluations/{EVAL}.json（同一预测，无额外调阈值推理）。

## 速度与计算量

RTX 5070 Ti / ONNX Runtime CUDA：GPU 绑定输入输出的同步前向 p50 {b['model_p50_ms']:.3f}ms、p95 {b['model_p95_ms']:.3f}ms（warmup 50、3×200 次，含 CPU 调度，不含传输/预处理）。六个场景各首段完整视频，CPU 解码至 JSONL 写盘 E2E {b['e2e_fps']:.2f} FPS，包含 RGB 转换、预处理、传输、推理和解码球心，排除初始化及可视化。初始化 {state['initialization_seconds']:.3f}s。

ONNX 权重/偏置元素 {b['params_m']:.6f}M；Conv/MatMul {b['gmac']:.6f} GMAC / {b['gflops']:.6f} GFLOPs（1 MAC=2 FLOPs），未计激活、融合 Add、SE Mul、pool、resize 和形状运算。ORT 显存不由 PyTorch allocator 管理，峰值显存未测，留 null。与 YOLO 的后端和原生输入不同，速度仅作本机各自流水线对照；没有进行赛事 TensorRT/4090 测试。

## 产物

- [全部叠加视频和代表帧]({os.path.relpath(dest/'index.html', record)})：绿色 GT / 红色预测，覆盖所有错误和未知帧。
- [逐视频指标]({os.path.relpath(dest/'per_video.json',record)})、[逐组指标]({os.path.relpath(dest/'per_group.json',record)})、[全部错误索引]({os.path.relpath(dest/'errors.json',record)})。
- [逐帧预测]({os.path.relpath(dest/'predictions.jsonl',record)})、[完整评估](evaluations/{EVAL}.json)、[实际配置](config.yaml)。
- [日志与源码快照]({os.path.relpath(attempt,record)})、[测速原始记录]({os.path.relpath(out/'benchmark'/EVAL,record)})。

未修改官方检测器、solution 或受保护框架，完整性校验通过。未训练或调整 baseline。
'''
        (record/'summary.md').write_text(summary)
        print(json.dumps(dict(experiment=exp,center=result['center'],benchmark=b),ensure_ascii=False,indent=2),flush=True)
    except BaseException:
        state.update(status='failed',finished=now().isoformat(),exit_code=1,error=traceback.format_exc())
        save(record/'run.json',state)
        comparison(exp,'failed')
        (record/'summary.md').write_text(state['error'])
        raise


if __name__ == '__main__':
    main()
