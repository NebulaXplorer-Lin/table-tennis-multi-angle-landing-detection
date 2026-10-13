#!/usr/bin/env python3
"""参赛方案快速自检：跑 1 条小视频端到端（解码→算法→写出→校验）。

    python3 scripts/check_solution.py --input <dataset> --solution participant.solution:Solution

80% 的接入错误（frame_id 语义、RGB/BGR、x/y 越界、score 越界、float 序列化）
会在这一步暴露，不用等全量评测。
"""

from __future__ import annotations

import os
import sys

# 任务根（scripts/ 的上一级）进 path——core / participant / validate 都在那里
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse       # noqa: E402
import json           # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent        # scripts/
COMPETITION = ROOT.parent                       # 任务根（sys.path[0] 已指向这里）

from core.dataset import load_manifest, resolve_manifest  # noqa: E402
from core.logging_utils import RunLogger  # noqa: E402
from core.runner import Runner  # noqa: E402
from run import load_solution  # noqa: E402
from validate import validate_predictions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--decoder", choices=["gpu", "cpu"], default="gpu")
    a = ap.parse_args()

    videos = load_manifest(resolve_manifest(a.input, a.manifest))
    out_dir = Path(__file__).resolve().parent / "out_check_solution"  # 固定包内，免受 CWD 影响
    logger = RunLogger(out_dir)
    runner = Runner(
        load_solution(a.solution), input_dir=a.input, output_dir=out_dir,
        logger=logger, decoder=a.decoder,
    )
    status = runner.run(videos[:1])  # 只跑第一条（最短路径验证）
    report = validate_predictions(out_dir / "predictions.jsonl", videos[:1])

    print(json.dumps({
        "run_status": status.get("status"),
        "video": status["videos"][0] if status["videos"] else None,
        "validation": {k: report[k] for k in ("n_lines", "n_errors", "n_warnings")},
        "errors": report["errors"][:5],
    }, ensure_ascii=False, indent=1))

    if status.get("status") == "ok" and report["n_errors"] == 0:
        print("SOLUTION CHECK PASSED ✅（可提交全量评测）")
        return 0
    print("SOLUTION CHECK FAILED —— 按上面 errors 修完再试")
    return 1


if __name__ == "__main__":
    sys.exit(main())
