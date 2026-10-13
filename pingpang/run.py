#!/usr/bin/env python3
"""标准运行入口。

本地（participant/ 即官方参考实现，直接改它即可）:
    python3 run.py --input public_data --output ../out/pingpang \
        --solution participant.solution:Solution --decoder gpu
正式评测（评测平台）:
    python3 /participant/pingpang/run.py --input <in_dir>/pingpang --output <out_dir>/pingpang \
        --solution participant.solution:Solution --decoder gpu
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 完整性自校验由 run.sh 在启动本脚本前调用（integrity.py --verify）
# 这里不再重复校验，避免循环依赖（run.py 自身也是被校验对象）

from core.dataset import load_manifest, resolve_manifest  # noqa: E402
from core.logging_utils import RunLogger  # noqa: E402
from core.runner import Runner  # noqa: E402


def load_solution(spec: str):
    """spec 形如 ``participant.solution:Solution``。"""
    mod_name, cls_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="乒乓球落点检测比赛 标准运行器")
    ap.add_argument("--input", required=True, help="数据集根目录（manifest 的 path 相对它）")
    ap.add_argument("--output", required=True, help="输出目录（predictions/run_status/run.log）")
    ap.add_argument("--solution", default="participant.solution:Solution", help="pkg.module:Class")
    ap.add_argument("--decoder", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--timeout-sec", type=int, default=7200, help="全局超时（秒）")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # manifest 回退链：<input>/manifest.json → 仓库 public_data/manifest.json
    manifest_path = resolve_manifest(args.input, None)
    videos = load_manifest(manifest_path)

    logger = RunLogger(args.output)
    logger.info(f"manifest={manifest_path} videos={len(videos)} "
                f"decoder={args.decoder} solution={args.solution}")

    solution = load_solution(args.solution)
    runner = Runner(
        solution, input_dir=args.input, output_dir=args.output, logger=logger,
        decoder=args.decoder,
        timeout_sec=args.timeout_sec,
    )

    # 固定流程：Solution.prepare()（含模型转换）→ 逐视频推理 → 写 predictions.jsonl
    status = runner.run(videos)

    # 出厂自检：对刚写出的 predictions.jsonl 跑一遍 schema 校验
    from validate import validate_predictions  # noqa: E402

    pred_path = Path(args.output) / "predictions.jsonl"
    report = validate_predictions(pred_path, videos) if pred_path.exists() \
        else {"n_lines": 0, "n_errors": 1, "errors": ["predictions.jsonl 不存在"]}
    status["validation"] = report
    logger.info(
        f"validate: {report['n_lines']} 行, {report['n_errors']} 非法, "
        f"{report['n_warnings']} 近邻重复警告"
    )

    # 输出额度：/output 总大小超 1GB 直接 fail
    total_bytes = 0
    for root, _dirs, files in os.walk(args.output):
        for f in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    size_gb = total_bytes / (1024 ** 3)
    status["output_size_gb"] = round(size_gb, 3)
    if size_gb > 1.0:
        status["status"] = "failed"
        status["output_size_exceeded"] = {"limit_gb": 1.0,
                                          "actual_gb": round(size_gb, 3)}
        logger.info(f"FAIL: 输出超限 {size_gb:.3f}GB > 1GB")

    # 写出 run_status.json（平台据此判 ok/failed 与各阶段计时）
    logger.dump_status(args.output)
    logger.info(f"完成 status={status['status']} -> {args.output}")
    logger.close()

    # 退出码契约：3=输出超限 / 1=校验有错 / 2=推理失败 / 0=成功
    if status.get("output_size_exceeded"):
        return 3
    if report["n_errors"]:
        return 1
    return 0 if status["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
