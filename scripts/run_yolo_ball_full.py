"""Train YOLO on all labelled training frames, export it, then score public landings."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "out/yolo-deps"))
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "out/yolo_setup/config")
os.environ["MPLCONFIGDIR"] = str(ROOT / "out/yolo_setup/matplotlib")

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def score_public(prediction_path: Path, output: Path) -> dict:
    """Public approximation: ±1 frame / 50 px, ignore ``valid:false`` events.

    Predictions within the tolerance of an ignored event are removed before
    scoring, reflecting the task description's instruction that these marked
    secondary/edge events should not affect the score.
    """
    gt_rows = [json.loads(x) for x in (ROOT / "pingpang/public_data/gt_public.jsonl").read_text().splitlines()]
    preds: dict[str, list[dict]] = {}
    for line in prediction_path.read_text().splitlines():
        row = json.loads(line)
        preds.setdefault(row["video_id"], []).append(row)
    by_video: dict[str, list[dict]] = {}
    for row in gt_rows:
        by_video.setdefault(row["video_id"], []).append(row)
    manifest = json.loads((ROOT / "pingpang/public_data/manifest.json").read_text())
    angle = {v["video_id"]: v["angle"] for v in manifest["videos"]}
    per_video, per_angle = {}, {}
    for video, rows in by_video.items():
        valid, ignored = [r for r in rows if r["valid"]], [r for r in rows if not r["valid"]]
        pp = preds.get(video, [])
        usable = [p for p in pp if not any(abs(p["frame_id"] - g["frame_id"]) <= 1 and np.hypot(p["x"]-g["x"], p["y"]-g["y"]) <= 50 for g in ignored)]
        if valid and usable:
            cost = np.full((len(valid), len(usable)), 1e9)
            for i, g in enumerate(valid):
                for j, p in enumerate(usable):
                    if abs(p["frame_id"] - g["frame_id"]) <= 1:
                        d = float(np.hypot(p["x"]-g["x"], p["y"]-g["y"]))
                        if d <= 50:
                            cost[i, j] = d
            rr, cc = linear_sum_assignment(cost)
            tp = sum(cost[i, j] < 1e9 for i, j in zip(rr, cc))
        else:
            tp = 0
        m = {"tp": int(tp), "fp": len(usable) - int(tp), "fn": len(valid) - int(tp),
             "ignored_predictions": len(pp) - len(usable), "predictions": len(pp), "valid_gt": len(valid)}
        per_video[video] = m
        a = per_angle.setdefault(angle[video], {"tp": 0, "fp": 0, "fn": 0})
        for k in ("tp", "fp", "fn"):
            a[k] += m[k]
    def finish(m: dict) -> dict:
        p = m["tp"] / max(1, m["tp"] + m["fp"])
        r = m["tp"] / max(1, m["tp"] + m["fn"])
        return dict(m, precision=p, recall=r, f1=2*p*r/max(1e-12, p+r))
    per_angle = {k: finish(v) for k, v in per_angle.items()}
    total = finish({k: sum(v[k] for v in per_video.values()) for k in ("tp", "fp", "fn")})
    result = {"protocol": "local approximation: valid=true GT, Hungarian one-to-one, |frame delta|<=1 and distance<=50px; predictions near valid=false events ignored", "overall": total, "per_angle": per_angle, "per_video": per_video}
    save(output, result)
    return result


def main() -> None:
    exp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d-%H%M%S") + "_yolo11n_full1280_public"
    record, out = ROOT / "experiments/records" / exp, ROOT / "out/experiments" / exp
    attempt = out / "attempts/001"
    record.mkdir(parents=True)
    attempt.mkdir(parents=True)
    weights = ROOT / "out/yolo_setup/weights/yolo11n.pt"
    state = {"status": "running", "started": now(), "experiment_id": exp,
             "data": {"split_id": "yolo_ball_full_v1", "train_videos": 154, "train_positive_frames": 20846,
                      "public_data_role": "external landing-event validation only; never detector training"},
             "pretrained": str(weights.relative_to(ROOT)), "pretrained_sha256": sha(weights),
             "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
             "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    save(record / "run.json", state)
    config = dict(model=str(weights), data=str(ROOT / "out/datasets/yolo_ball_full_v1/data.yaml"), imgsz=1280,
                  epochs=80, batch=-1, device=0, workers=8, optimizer="AdamW", lr0=0.001, lrf=0.01,
                  weight_decay=0.0005, cos_lr=True, warmup_epochs=3.0, seed=42, deterministic=True,
                  pretrained=True, amp=True, val=False, patience=0, mosaic=0.20, close_mosaic=10,
                  mixup=0.0, scale=0.10, translate=0.05, fliplr=0.5, flipud=0.0, degrees=0.0,
                  project=str(attempt), name="train", save=True, plots=True, verbose=True)
    (record / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    try:
        model = YOLO(str(weights))
        # A complete fixed-length run; no public data or held-out training frame
        # is used to select an epoch.  Ultralytics' auto batch targets 60% VRAM.
        model.train(**config)
        train_dir = attempt / "train"
        final = train_dir / "weights/last.pt"
        if not final.exists():
            raise RuntimeError(f"missing final checkpoint: {final}")
        shutil.copy2(train_dir / "args.yaml", record / "actual_config.yaml")
        shutil.copy2(train_dir / "results.csv", attempt / "train_metrics.csv")
        shutil.copytree(train_dir / "weights", attempt / "checkpoints")
        export_model = YOLO(str(final))
        exported = Path(export_model.export(format="onnx", imgsz=1280, dynamic=False, simplify=True, opset=17))
        target = ROOT / "pingpang/participant/weights/yolo_ball_full_v1.onnx"
        shutil.copy2(exported, target)
        state.update(status="trained_exported", final_checkpoint=str(final.relative_to(ROOT)), final_checkpoint_sha256=sha(final),
                     exported_onnx=str(target.relative_to(ROOT)), exported_onnx_sha256=sha(target))
        save(record / "run.json", state)
        # Execute the actual local submission entry point: YOLO centres flow
        # directly into Solution._emit, which is the original heuristic.
        public_out = out / "public_yolo_heuristic"
        command = [sys.executable, "pingpang/run.py", "--input", "pingpang/public_data", "--output", str(public_out),
                   "--solution", "participant.solution_local:Solution", "--decoder", "cpu"]
        with (attempt / "public_run.log").open("w") as log:
            p = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        if p.returncode:
            raise RuntimeError(f"public runner exited {p.returncode}; see {attempt / 'public_run.log'}")
        public = score_public(public_out / "predictions.jsonl", record / "evaluations/public_landing_f1.json")
        shutil.copy2(ROOT / "pingpang/participant/solution.py", attempt / "solution.py")
        shutil.copy2(ROOT / "pingpang/participant/yolo_ball_detector.py", attempt / "yolo_ball_detector.py")
        state.update(status="completed", finished=now(), exit_code=0, public_overall=public["overall"], public_per_angle=public["per_angle"])
        save(record / "run.json", state)
        (record / "summary.md").write_text(
            f"# YOLO11n 全训练集 → 原落点启发式\n\n"
            f"完整训练：154 视频、20,846 张有球标注帧，1280 方形输入，80 epoch，无 early stopping。"
            f"检测器没有使用 public_data；公开集仅用于最终端到端验证。\n\n"
            f"- public 近似 F1：{public['overall']['f1']:.6f}（P={public['overall']['precision']:.6f}，R={public['overall']['recall']:.6f}）\n"
            + "\n".join(f"- {a}: F1={m['f1']:.6f}, P={m['precision']:.6f}, R={m['recall']:.6f}" for a, m in public['per_angle'].items())
            + f"\n\n部署 ONNX：`pingpang/participant/weights/yolo_ball_full_v1.onnx`。"
              f"公开集原始运行输出：`{public_out.relative_to(ROOT)}`。\n")
        print(json.dumps({"experiment_id": exp, "public": public}, ensure_ascii=False, indent=2))
    except BaseException as exc:
        state.update(status="failed", finished=now(), error=f"{type(exc).__name__}: {exc}")
        save(record / "run.json", state)
        raise


if __name__ == "__main__":
    main()
