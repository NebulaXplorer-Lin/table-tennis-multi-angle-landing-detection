#!/usr/bin/env python3
"""输出校验（可独立运行）。

规则（单条非法即报错，不 clip、不自动修正）：
  - 每行必须是 JSON 对象，含 video_id / frame_id / x / y
  - video_id 必须来自 manifest
  - frame_id 整数且 [0, n_frames)
  - x/y 有限数值且 [0, width) x [0, height)
  - 近邻重复（同视频 |Δframe|<=1 且距离<=50px）只警告不修改
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


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_predictions(pred_path: str | Path, videos: list[VideoMeta]) -> dict:
    by_id = {v.video_id: v for v in videos}
    errors: list[str] = []
    warnings: list[str] = []
    seen: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
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
        missing = [k for k in ("video_id", "frame_id", "x", "y") if k not in rec]
        if missing:
            errors.append(f"line {lineno}: 缺字段 {missing}")
            continue

        vid = rec["video_id"]
        if vid not in by_id:
            errors.append(f"line {lineno}: 未知 video_id {vid!r}")
            continue
        meta = by_id[vid]

        fid, x, y = rec["frame_id"], rec["x"], rec["y"]
        if not isinstance(fid, int) or isinstance(fid, bool):
            errors.append(f"line {lineno}: frame_id 必须是整数，得到 {fid!r}")
            continue
        if not (_is_num(x) and _is_num(y)):
            errors.append(f"line {lineno}: x/y 必须是有限数值")
            continue
        if not (0 <= fid < meta.n_frames):
            errors.append(f"line {lineno}: frame_id {fid} 越界 [0,{meta.n_frames})")
        if not (0 <= x < meta.width):
            errors.append(f"line {lineno}: x={x} 越界 [0,{meta.width})")
        if not (0 <= y < meta.height):
            errors.append(f"line {lineno}: y={y} 越界 [0,{meta.height})")

        if not (0 <= fid < meta.n_frames):
            continue
        for pf, px, py in seen[vid]:
            if abs(pf - fid) <= 1 and math.hypot(px - x, py - y) <= 50:
                warnings.append(
                    f"line {lineno}: 与 {vid} 帧{pf} 的预测近邻重复（|Δf|<=1 且 <=50px）")
                break
        seen[vid].append((fid, x, y))
        per_video[vid] += 1

    return {
        "n_lines": n,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors[:50],
        "near_dup_examples": warnings[:5],
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
