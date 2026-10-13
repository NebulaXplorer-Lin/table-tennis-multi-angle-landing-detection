#!/usr/bin/env python3
"""环境契约自检（参赛者提交前 / 评测部署后各跑一次）。

    python3 check_env.py

检查基础栈版本、CUDA/TRT/ORT 可用性、ffmpeg GPU 解码能力。任何一项 FAIL 都应
先修环境再继续——**不要通过升级 torch/numpy/onnxruntime/tensorrt 来"修复"**，
那会破坏 cu118 契约（cuDNN8 依赖 torch/lib 的 LD_LIBRARY_PATH）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

FAILED = []


def check(name, fn):
    try:
        print(f"PASS {name}: {fn()}")
    except Exception as e:
        FAILED.append(name)
        print(f"FAIL {name}: {type(e).__name__}: {e}")


def c_torch():
    import torch
    assert torch.__version__ == "2.1.2+cu118", f"应为 2.1.2+cu118，实际 {torch.__version__}"
    return f"torch {torch.__version__}, cuda 可用={torch.cuda.is_available()}"


def c_numpy():
    import numpy
    assert numpy.__version__ == "1.26.4", f"应为 1.26.4，实际 {numpy.__version__}"
    return f"numpy {numpy.__version__}"


def c_trt():
    import tensorrt as trt
    assert trt.__version__.startswith("8.6."), f"应为 8.6.x，实际 {trt.__version__}"
    return f"TensorRT {trt.__version__}（import 成功即 LD_LIBRARY_PATH 契约生效）"


def c_ort():
    import onnxruntime as ort
    assert ort.__version__ == "1.18.1", f"应为 1.18.1，实际 {ort.__version__}"
    assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA EP 不可用"
    return f"onnxruntime-gpu {ort.__version__}, CUDA EP 可用"


def c_media():
    import av
    import cv2
    return f"av {av.__version__}, opencv {cv2.__version__}"


def c_ffmpeg():
    exe = shutil.which("ffmpeg")
    assert exe, "PATH 里没有 ffmpeg"
    out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"],
                         capture_output=True, text=True, timeout=30).stdout
    assert "cuda" in out, f"ffmpeg 无 cuda hwaccel: {out.strip()}"
    return f"{exe}, cuda hwaccel 可用"


def c_gpu():
    """资源额度展示：GPU 卡数 + 每卡显存（评测配额 2×GPU）。无 GPU 直接 fail。"""
    import torch
    n = torch.cuda.device_count()
    assert n > 0, "无可用 CUDA GPU（评测需 2×GPU）"
    props = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        props.append(f"{p.name} {p.total_memory // (1024**3)}GB")
    return f"{n} 卡: " + ", ".join(props)


def c_disk():
    """资源额度展示：/output 可写空间（评测输出限 1GB）。不足只警告不 fail
    ——评测平台可能挂载更大盘到 /output，容器内见到的未必是评测时实际可用。"""
    target = "/output" if os.path.isdir("/output") else "/"
    free = shutil.disk_usage(target).free
    info = f"{target} 可用 {free / (1024**3):.1f}GB（评测输出限 1GB）"
    if free < 1024 ** 3:
        print(f"WARN c_disk: {info} < 1GB，评测可能写不下输出")
    return info


if __name__ == "__main__":
    check("torch 契约", c_torch)
    check("numpy 契约", c_numpy)
    check("TensorRT", c_trt)
    check("ORT CUDA EP", c_ort)
    check("av/opencv", c_media)
    check("ffmpeg GPU 解码", c_ffmpeg)
    check("GPU 资源", c_gpu)
    check("磁盘额度", c_disk)
    if FAILED:
        print(f"\nENV CHECK FAILED: {FAILED}")
        sys.exit(1)
    print("\nENV CHECK ALL PASSED ✅")
