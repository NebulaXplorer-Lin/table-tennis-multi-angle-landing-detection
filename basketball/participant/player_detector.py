"""球员检测模型最小实现（官方示例，教学基线，不追精度）。

参赛者仿写本文件即可完成完整接入（含 TensorRT）：
  1. 权重只带 ONNX（weights/player.onnx，跨机器可移植的唯一交付格式）
  2. 模型加载 / TRT 转换由参赛者在 Solution.prepare 里自己写——见 solution.py
  3. 本文件只管预处理 + 推理 + 解码

预处理：letterbox 等比缩放到 (IN_H, IN_W) + pad 114 → /255 → RGB→CHW float32。
解码：自动识别三种常见 YOLO 导出布局——
  - YOLOv5/7 风格 [1, N, 5+nc]（cx,cy,w,h,obj_conf,cls...），得分 = obj * cls
  - YOLOv8/11 风格 [1, 4+nc, N]（cx,cy,w,h,cls...），得分 = cls
  - 端到端 NMS-free 风格 [1, K, 6]（x1,y1,x2,y2,score,class_id，YOLO26/v10 等，
    图内已完成 NMS 与 argmax，K 固定如 300，补齐行 score=0）
  首次推理时按输出 shape 自动判定（可用 layout 参数强制指定）。
后处理：类过滤（COCO 预训练 person=0；自训单类模型传 None）+ 阈值 + numpy NMS
  + top-K 截断（默认 12：场上 10 人 + 裁判 + 余量；评分侧每帧上限 15，
  超上限的框一律计 FP，demo 主动收敛在安全线内）。

已知简化（README 有声明）：
  - 不做 Test-Time Augmentation / 多尺度
  - cv2.resize 与训练端 resize 有微小数值差异
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IN_H, IN_W = 960, 960          # letterbox 目标尺寸（1920x1080 下 640 会漏远端球员）
CONF_TH = 0.35
NMS_IOU_TH = 0.5
TOP_K = 12                     # 每帧输出上限（评分硬上限 15 的安全线）
PAD_COLOR = (114, 114, 114)


class PlayerDetector:
    """球员检测器：依赖一个外部传入的推理函数 _infer（输入 (1,3,IN_H,IN_W) float32）。"""

    def __init__(self, infer_fn, input_size: tuple[int, int] = (IN_H, IN_W),
                 classes: list[int] | None = None,
                 layout: str | None = None,
                 conf_th: float = CONF_TH, top_k: int = TOP_K):
        """classes: 保留的类别索引（COCO 预训练 person 检测传 [0]；单类模型传 None）。
        layout: "v5" | "v8" | "e2e" | None（自动检测）。"""
        self._infer = infer_fn
        self.in_h, self.in_w = input_size
        self.classes = classes
        self.conf_th = conf_th
        self.top_k = top_k
        self._layout = layout          # 首次推理时自动判定
        self._nc = None

    # ── 预处理 ──────────────────────────────────────────────────────────────
    def _letterbox(self, frame_rgb: np.ndarray):
        """等比缩放 + pad 到 (in_h, in_w)。返回 (输入张量, 缩放参数)。"""
        h, w = frame_rgb.shape[:2]
        s = min(self.in_w / w, self.in_h / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.in_h, self.in_w, 3), PAD_COLOR, np.uint8)
        ox, oy = (self.in_w - nw) // 2, (self.in_h - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        x = canvas.astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]   # (1,3,H,W) RGB
        return x, (s, ox, oy)

    # ── 解码 ────────────────────────────────────────────────────────────────
    def _decode(self, out: np.ndarray, s: float, ox: int, oy: int,
                src_h: int, src_w: int) -> np.ndarray:
        """输出 (n,5) [x1,y1,x2,y2,score]（原图坐标，已 clip 到画面内）。"""
        out = out[0]
        if self._layout is None:                     # 自动判定布局
            if out.ndim != 2:
                raise ValueError(f"不认识的输出 shape {out.shape}（需要 2D），"
                                 f"请在构造时指定 layout='v5'|'v8'|'e2e'")
            if (out.shape[1] == 6 and out.shape[0] <= 1024
                    and np.all(out[:, 5] == np.round(out[:, 5]))):
                self._layout = "e2e"             # 末列整数类别 id → 端到端导出
            else:
                self._layout = "v5" if out.shape[0] > out.shape[1] else "v8"

        if self._layout == "e2e":                # [K,6] x1,y1,x2,y2,score,cls（图内已 NMS）
            self._nc = 1
            boxes_cxcywh = np.stack([(out[:, 0] + out[:, 2]) / 2,
                                     (out[:, 1] + out[:, 3]) / 2,
                                     out[:, 2] - out[:, 0], out[:, 3] - out[:, 1]], 1)
            scores_idx = out[:, 4:5]
            cls_ids = out[:, 5].astype(np.int64)     # 已是最终类别，不走 argmax
        elif self._layout == "v5":               # [N, 5+nc] cx,cy,w,h,obj,cls*
            self._nc = out.shape[1] - 5
            obj = out[:, 4:5]
            scores_idx = out[:, 5:] * obj
            boxes_cxcywh = out[:, :4]
            cls_ids = scores_idx.argmax(1)
        else:                                    # [4+nc, N] cx,cy,w,h,cls*
            self._nc = out.shape[0] - 4
            boxes_cxcywh, scores_idx = out[:4, :].T, out[4:, :].T
            cls_ids = scores_idx.argmax(1)
        scores = scores_idx.max(1)
        keep = scores > self.conf_th
        if self.classes is not None:
            keep &= np.isin(cls_ids, self.classes)
        if not keep.any():
            return np.zeros((0, 5), np.float32)
        boxes_cxcywh, scores, cls_ids = boxes_cxcywh[keep], scores[keep], cls_ids[keep]

        cx, cy = (boxes_cxcywh[:, 0] - ox) / s, (boxes_cxcywh[:, 1] - oy) / s
        bw, bh = boxes_cxcywh[:, 2] / s, boxes_cxcywh[:, 3] / s
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        # 贴 letterbox 灰边的框回映射后会微越界（如 x1=-0.35）→ clip 到画面；
        # 完全落在画面外的框整行丢弃（clip 后会退化成 x1==x2，过不了 x1<x2 校验）
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, src_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, src_h)
        ok = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, cls_ids = boxes[ok], scores[ok], cls_ids[ok]

        order = scores.argsort()[::-1][: self.top_k * 3]
        boxes, scores, cls_ids = boxes[order], scores[order], cls_ids[order]
        keep = _nms(boxes, scores, NMS_IOU_TH)
        boxes, scores = boxes[keep][: self.top_k], scores[keep][: self.top_k]
        return np.concatenate([boxes, scores[:, None]], 1)

    # ── 推理入口 ────────────────────────────────────────────────────────────
    def detect(self, frame_rgb: np.ndarray) -> np.ndarray:
        """返回 (n,5) [x1,y1,x2,y2,score]（原分辨率像素坐标，已 clip 画面内）。"""
        x, (s, ox, oy) = self._letterbox(frame_rgb)
        out = self._infer(x)
        return self._decode(np.asarray(out), s, ox, oy, *frame_rgb.shape[:2])


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> list[int]:
    """numpy NMS。boxes (n,4) xyxy。返回保留索引（按分数降序）。"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= iou_th]
    return keep


def weights_path() -> Path:
    """demo 权重路径；不存在时给出可操作的报错（权重由赛事方提供）。"""
    p = Path(__file__).resolve().parent / "weights" / "player.onnx"
    if not p.exists():
        raise FileNotFoundError(
            f"缺少检测模型权重: {p}\n"
            f"  请把球员检测 ONNX 命名为 player.onnx 放入 participant/weights/。\n"
            f"  约定：输入 (1,3,H,W) float32 RGB /255（H/W 建议 960 或 1280，静态尺寸），\n"
            f"  输出 YOLOv5 风格 [1,N,5+nc] 或 YOLOv8 风格 [1,4+nc,N]（自动识别），\n"
            f"  详见 weights/README.md。")
    return p
