"""Evaluate a frozen training checkpoint with the existing landing heuristic."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from run_yolo_ball_full import ROOT, YOLO, now, save, score_public, sha
import cv2
import torch


def main():
    exp = '20261013-215749_yolo11n_full1280_public'
    record = ROOT / 'experiments/records' / exp
    out = ROOT / 'out/experiments' / exp
    checkpoint = out / 'attempts/001/train/weights/last.pt'
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    epoch = ckpt['epoch'] + 1
    eval_id = f'epoch{epoch:03d}_public_paused'
    dest = out / 'eval' / eval_id
    dest.mkdir(parents=True, exist_ok=True)
    frozen = dest / 'checkpoint.pt'
    if not frozen.exists():
        shutil.copy2(checkpoint, frozen)
    state = json.loads((record / 'run.json').read_text())
    state.update(status='paused', paused_at=now(), completed_epochs=epoch, paused_process_group=229018,
                 paused_checkpoint=str(frozen.relative_to(ROOT)), pause_method='SIGSTOP; training state remains resident')
    save(record / 'run.json', state)
    del ckpt
    model = YOLO(str(frozen))
    exported = Path(model.export(format='onnx', imgsz=1280, dynamic=False, simplify=False, opset=17, device='cpu'))
    target = ROOT / 'pingpang/participant/weights/yolo_ball_full_v1.onnx'
    if target.exists():
        shutil.copy2(target, dest / 'previous_deployed.onnx')
    shutil.copy2(exported, target)
    command = [sys.executable, 'pingpang/run.py', '--input', 'pingpang/public_data', '--output', str(dest / 'runner'),
               '--solution', 'participant.solution_local:Solution', '--decoder', 'cpu']
    with (dest / 'runner_console.log').open('w') as log:
        env = dict(os.environ)
        # CPU export sets CUDA_VISIBLE_DEVICES=-1 in Ultralytics; restore the
        # GPU visibility for the independent ONNX Runtime runner.
        env.pop('CUDA_VISIBLE_DEVICES', None)
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    prediction = dest / 'runner/predictions.jsonl'
    status = json.loads((dest / 'runner/run_status.json').read_text())
    assert len(status['videos']) == 15 and all(v['status'] == 'ok' for v in status['videos']), status
    subprocess.run([sys.executable, 'pingpang/validate.py', '--predictions', str(prediction), '--manifest',
                    'pingpang/public_data/manifest.json'], cwd=ROOT, check=True)
    metrics = score_public(prediction, record / 'evaluations' / f'{eval_id}.json')
    metrics.update(epoch=epoch, checkpoint=str(frozen.relative_to(ROOT)), checkpoint_sha256=sha(frozen),
                   onnx_sha256=sha(target), conf=0.05, nms_iou=0.45, heuristic={'MIN_GAP':10,'FALL_AMP':2.0,'RISE_AMP':1.0},
                   macro_f1=sum(m['f1'] for m in metrics['per_angle'].values())/3,
                   run_status=status, evaluated_at=now())
    # Full videos cover every predicted and labelled event, including misses.
    pred = [json.loads(l) for l in prediction.read_text().splitlines()]
    gt = [json.loads(l) for l in (ROOT / 'pingpang/public_data/gt_public.jsonl').read_text().splitlines()]
    manifest = json.loads((ROOT / 'pingpang/public_data/manifest.json').read_text())
    previews = dest / 'overlays'
    previews.mkdir()
    index = []
    for video in manifest['videos']:
        vid = video['video_id']
        cap = cv2.VideoCapture(str(ROOT / 'pingpang/public_data' / video['path']))
        path = previews / f'{vid}.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), video['fps'], (video['width'],video['height']))
        assert writer.isOpened()
        fid = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            cv2.putText(im, f'{vid} frame={fid} | GT green / ignored yellow / prediction red | conf=0.05',
                        (30,40), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
            for rows, is_gt in [(gt, True), (pred, False)]:
                for row in rows:
                    if row['video_id'] == vid and abs(row['frame_id']-fid) <= 2:
                        color = ((0,255,0) if row.get('valid',True) else (0,255,255)) if is_gt else (0,0,255)
                        pt = (round(row['x']),round(row['y']))
                        cv2.circle(im,pt,15,color,3)
                        cv2.putText(im, f"{'GT' if is_gt else 'P'} f={row['frame_id']}", (pt[0]+18,pt[1]),0,.6,color,2)
            writer.write(im)
            fid += 1
        cap.release()
        writer.release()
        assert fid == video['n_frames']
        index.append({'video_id':vid,'metrics':metrics['per_video'][vid], 'overlay':str(path.relative_to(ROOT))})
    save(dest / 'error_index.json', {'videos':index,'scope':'all events shown in complete videos; green valid GT, yellow ignored GT, red predictions'})
    metrics['visualizations'] = str(previews.relative_to(ROOT))
    save(record / 'evaluations' / f'{eval_id}.json', metrics)
    state['paused_evaluation'] = {'eval_id':eval_id,'overall':metrics['overall'],'macro_f1':metrics['macro_f1']}
    save(record / 'run.json', state)
    (record / 'summary.md').write_text(f"# YOLO11n paused evaluation\n\nTraining paused after {epoch} completed epochs; 80 were planned. All 154 training videos / 20,846 labelled positive frames, 1280 input. Checkpoint: `{frozen.relative_to(ROOT)}`.\n\nPublic landing validation: micro F1={metrics['overall']['f1']:.6f}, macro F1={metrics['macro_f1']:.6f}.\n\n" + '\n'.join(f"- {a}: {m}" for a,m in metrics['per_angle'].items()) + f"\n\nLocal approximate scorer: ±1 frame / 50px; predictions near ignored events removed before valid-GT matching. All 15 complete overlay videos: `{previews.relative_to(ROOT)}`. Training remains paused.\n")
    print(json.dumps({'epoch':epoch,'overall':metrics['overall'],'per_angle':metrics['per_angle'],'macro_f1':metrics['macro_f1'],'performance':status.get('performance')},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
