#!/usr/bin/env python3
"""输出校验（可独立运行）。

规则（单条非法即报错，不 clip、不自动修正）：
  - 每行必须是 JSON 对象，含 video_id / frame_id / track_id / bbox
  - video_id 必须来自 manifest
  - frame_id 整数且 [0, n_frames)（0 基）
  - track_id 正整数
  - bbox 为 4 个有限数值 [x1,y1,x2,y2]，x1<x2、y1<y2，且在画面 [0,width)x[0,height) 内
  - 同一 (video_id, frame_id, track_id) 只允许一行——重复说明跟踪器状态有 bug
警告（不算错，但影响得分）：
  - 某帧预测数超过上限（默认 15）：按输出顺序保留前 15 个，超出的框在评分时按 FP 计
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.dataset import load_manifest  # noqa: E402
from core.types import VideoMeta  # noqa: E402

# 与 external/score.py 的 --cap 保持一致（每帧预测数上限，超出部分计 FP）
FRAME_PRED_CAP = 15


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_predictions(pred_path: str | Path, videos: list[VideoMeta]) -> dict:
    by_id = {v.video_id: v for v in videos}
    errors: list[str] = []
    warnings: list[str] = []
    # (video_id, frame_id) -> {track_id} 查重；(video_id, frame_id) -> 框数 查上限
    seen_ids: dict[tuple[str, int], set[int]] = defaultdict(set)
    frame_counts: dict[tuple[str, int], int] = defaultdict(int)
    n = 0
    per_video: dict[str, int] = defaultdict(int)

    for lineno, line in enumerate(Path(pred_path).open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {lineno}: 非法 JSON（{e}）")
            continue
        if not isinstance(rec, dict):
            errors.append(f"line {lineno}: 必须是 JSON 对象")
            continue
        missing = [k for k in ("video_id", "frame_id", "track_id", "bbox")
                   if k not in rec]
        if missing:
            errors.append(f"line {lineno}: 缺字段 {missing}")
            continue

        vid = rec["video_id"]
        if vid not in by_id:
            errors.append(f"line {lineno}: 未知 video_id {vid!r}")
            continue
        meta = by_id[vid]

        fid, tid = rec["frame_id"], rec["track_id"]
        if not isinstance(fid, int) or isinstance(fid, bool):
            errors.append(f"line {lineno}: frame_id 必须是整数，得到 {fid!r}")
            continue
        if not isinstance(tid, int) or isinstance(tid, bool) or tid <= 0:
            errors.append(f"line {lineno}: track_id 必须是正整数，得到 {tid!r}")
            continue
        bbox = rec["bbox"]
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(_is_num(v) for v in bbox)):
            errors.append(f"line {lineno}: bbox 必须是 4 个有限数值 [x1,y1,x2,y2]")
            continue
        x1, y1, x2, y2 = bbox
        if not (x1 < x2 and y1 < y2):
            errors.append(f"line {lineno}: bbox 需满足 x1<x2 且 y1<y2，得到 {bbox}")
            continue
        if not (0 <= fid < meta.n_frames):
            errors.append(f"line {lineno}: frame_id {fid} 越界 [0,{meta.n_frames})")
            continue
        if not (0 <= x1 < meta.width and 0 <= x2 <= meta.width
                and 0 <= y1 < meta.height and 0 <= y2 <= meta.height):
            errors.append(
                f"line {lineno}: bbox 越界（画面 {meta.width}x{meta.height}）: {bbox}")
            continue

        if tid in seen_ids[(vid, fid)]:
            errors.append(
                f"line {lineno}: {vid} 帧{fid} track_id={tid} 重复（一条轨迹一帧至多一框）")
            continue
        seen_ids[(vid, fid)].add(tid)
        frame_counts[(vid, fid)] += 1
        per_video[vid] += 1

    for (vid, fid), cnt in sorted(frame_counts.items()):
        if cnt > FRAME_PRED_CAP:
            warnings.append(
                f"{vid} 帧{fid}: 预测数 {cnt} 超过上限 {FRAME_PRED_CAP}，"
                f"评分时按输出顺序保留前 {FRAME_PRED_CAP} 个，超出部分按 FP 计"
                f"（GT 有框帧的未匹配预测本不计 FP，"
                f"但超上限的框一律计 FP——防密集框刷分）")

    return {
        "n_lines": n,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors[:50],
        "cap_exceeded_frames": len(warnings),
        "cap_examples": warnings[:5],
        "videos_with_predictions": len(per_video),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="predictions.jsonl 校验")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args(argv)
    videos = load_manifest(a.manifest)
    report = validate_predictions(a.predictions, videos)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 1 if report["n_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
