"""Small fixed-shape YOLO ONNX decoder used by the submitted solution."""
from __future__ import annotations

import cv2
import numpy as np


class YoloBallDetector:
    """Letterbox preprocessing and single-class NMS for exported Ultralytics YOLO."""

    def __init__(self, infer, size: int = 1280, score_threshold: float = 0.05, nms_iou: float = 0.45):
        self._infer = infer
        self.size = size
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou

    def reset(self) -> None:
        """Single-frame detector has no video state to clear."""
        pass

    def detect(self, frame_rgb: np.ndarray, orig_w: int, orig_h: int) -> dict | None:
        scale = min(self.size / orig_w, self.size / orig_h)
        resized_w, resized_h = round(orig_w * scale), round(orig_h * scale)
        pad_x, pad_y = (self.size - resized_w) // 2, (self.size - resized_h) // 2
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = cv2.resize(frame_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        inp = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        raw = np.asarray(self._infer(inp))
        # Ultralytics detect export: (1, 4 + nc, anchors).  Be permissive about
        # a transposed output because TensorRT preserves the ONNX shape.
        pred = raw[0]
        if pred.shape[0] in (5, 84):
            pred = pred.T
        if pred.ndim != 2 or pred.shape[1] < 5:
            raise RuntimeError(f"unexpected YOLO output shape {raw.shape}")
        scores = pred[:, 4]
        keep = scores >= self.score_threshold
        if not np.any(keep):
            return None
        boxes, scores = pred[keep, :4], scores[keep]
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        # cv2.dnn.NMSBoxes takes x/y/w/h; use it only after score filtering so
        # tiny low-confidence anchors never dominate the detector.
        nms_boxes = [[float(a), float(b), float(c-a), float(d-b)] for a, b, c, d in xyxy]
        idx = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), self.score_threshold, self.nms_iou)
        if len(idx) == 0:
            return None
        idx = np.asarray(idx).reshape(-1)
        best = int(idx[np.argmax(scores[idx])])
        cx, cy = boxes[best, 0], boxes[best, 1]
        x, y = (cx - pad_x) / scale, (cy - pad_y) / scale
        if not (0 <= x < orig_w and 0 <= y < orig_h):
            return None
        return {"x": float(x), "y": float(y), "conf": float(scores[best])}
