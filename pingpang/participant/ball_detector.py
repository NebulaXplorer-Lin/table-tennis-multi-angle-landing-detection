"""球检测模型最小实现（官方示例，教学基线，不追精度）。

参赛者仿写本文件即可完成完整接入（含 TensorRT）：
  1. 权重只带 ONNX（weights/ball.onnx，跨机器可移植的唯一交付格式）
  2. 模型加载 / TRT 转换由参赛者在 Solution.prepare 里自己写——见 solution.py
  3. 本文件只管预处理 + 推理 + 解码

预处理在 CPU（cv2+numpy）做：resize 到 288x512 -> /255 -> ImageNet mean/std
-> 3 帧滑动栈拼 (1,9,288,512)；解码取 ch1 热图 >0.7 的 blob 加权质心，
坐标按 sx=W/512, sy=H/288 回原分辨率

已知简化（与更复杂实现的差异，README 有声明）：
  - cv2.resize 与 F.interpolate 有微小数值差异，不保证逐字节一致
  - 不做多窗口投票，固定解码输出通道 ch1（中间帧，固有 1 帧滞后）
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IN_H, IN_W = 288, 512
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SCORE_TH = 0.7
OUT_CHANNEL = 1  # 3 通道热图对应 buffer 的 [旧, 中, 新] 三帧，取中间帧


class BallDetector:
    """球检测器：依赖一个外部传入的推理函数 _infer（输入 (1,9,H,W)，输出热图）。"""

    def __init__(self, infer_fn, onnx_path: str | Path):
        self._infer = infer_fn
        self.onnx_path = Path(onnx_path)
        self._blank = self._normalize(np.zeros((IN_H, IN_W, 3), np.uint8))
        self.reset()

    # ── 预处理 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(frame_rgb: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 -> (3,288,512) float32 归一化。"""
        x = cv2.resize(frame_rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
        x = x.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        return np.ascontiguousarray(x.transpose(2, 0, 1))

    def reset(self) -> None:
        self._buf = [self._blank, self._blank]  # 预填 2 帧

    # ── 推理 + 解码 ─────────────────────────────────────────────────────────
    def detect(self, frame_rgb: np.ndarray, orig_w: int, orig_h: int) -> dict | None:
        """返回 {"x","y","conf"}（原分辨率像素）或 None。当前帧结果对应 ch1（滞后 1 帧）。"""
        self._buf.append(self._normalize(frame_rgb))
        self._buf = self._buf[-3:]
        inp = np.concatenate(self._buf, axis=0)[None]  # (1,9,288,512)
        heat = self._infer(inp)[0, OUT_CHANNEL]        # (288,512) sigmoid 已含
        mask = heat > SCORE_TH
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        w = heat[ys, xs].astype(np.float64)
        total = w.sum()
        cx = float((xs * w).sum() / total)
        cy = float((ys * w).sum() / total)
        return {
            "x": cx * orig_w / IN_W,
            "y": cy * orig_h / IN_H,
            "conf": float(heat[mask].max()),
        }
