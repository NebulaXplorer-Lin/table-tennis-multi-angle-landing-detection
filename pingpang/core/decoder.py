"""GPU/CPU 统一解码：产出 RGB uint8 (H,W,3)，帧数与 manifest 强一致。

- gpu: ffmpeg 子进程 `-hwaccel cuda` NVDEC 解码 + 管道读 rawvideo rgb24
- cpu: PyAV 直读（兜底/调试路径）

两条路径都允许与对方有微小数值差异（swscale/hwaccel 实现），验收以标准评测
decoder（gpu）为准；但帧数必须精确等于 manifest.n_frames，否则视为解码器漂移，
该视频直接 fail。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np

from .types import VideoMeta


class DecoderError(RuntimeError):
    """解码失败 / 帧数不一致。"""


def iter_frames_cpu(path: Path) -> Iterator[np.ndarray]:
    import av  # 延迟导入：gpu 路径不需要 av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


def iter_frames_gpu(path: Path, width: int, height: int) -> Iterator[np.ndarray]:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda",
        "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    frame_size = width * height * 3
    # stderr 落临时文件，避免 PIPE 缓冲区满导致死锁
    errf = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf)
    n = 0
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf:
                break
            if len(buf) != frame_size:
                raise DecoderError(f"{path.name}: 截断的第 {n} 帧（{len(buf)} 字节）")
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3).copy()
            n += 1
    finally:
        if proc.poll() is None:
            proc.kill()
    errf.seek(0)
    err = errf.read().decode("utf-8", "replace").strip()
    errf.close()
    rc = proc.wait()
    if rc != 0:
        raise DecoderError(f"ffmpeg 退出码 {rc}: {err[-500:]}")


def decode_video(meta: VideoMeta, input_dir: str | Path, mode: str) -> Iterator[tuple[int, np.ndarray]]:
    """yield (frame_id, rgb)；流结束后校验总帧数 == meta.n_frames。"""
    path = Path(input_dir) / meta.path
    if not path.exists():
        raise DecoderError(f"视频不存在: {path}")
    if mode == "gpu":
        gen = iter_frames_gpu(path, meta.width, meta.height)
    elif mode == "cpu":
        gen = iter_frames_cpu(path)
    else:
        raise ValueError(f"未知 decoder: {mode}")
    n = 0
    try:
        for img in gen:
            yield n, img
            n += 1
    finally:
        gen.close()
    if n != meta.n_frames:
        raise DecoderError(
            f"{meta.video_id}: 解码 {n} 帧 != manifest {meta.n_frames} 帧（{mode}）——"
            f"解码器帧数漂移，该结果不可信"
        )
