"""IoU 跟踪器（ByteTrack 思路的精简教学版：两段关联，无 Kalman）。

流程（每帧 update）：
  1) 高置信检测 vs 存活轨迹：IoU Hungarian 关联（延续上一帧框做预测，无运动模型）
  2) 低置信检测 vs 剩余未匹配轨迹：再关联一次（ByteTrack 的核心思想——
     遮挡期检测分数骤降，低分框对"续命"轨迹非常有用，直接丢弃会碎轨迹）
  3) 未匹配的高置信检测 → 新轨迹（track_id 递增，每视频局部）
  4) 轨迹失联超 max_age → 删除；hit_streak < min_hits 的轨迹不上报（抑制闪烁）

已知简化（教学定位）：无 Kalman/ReID/相机运动补偿；关联只用 IoU。
GT 轨迹本身有遮挡断裂-同 id 回归，跟踪鲁棒性主要靠 2) 的低分续命。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_TH_HIGH = 0.5     # 关联的 IoU 门槛（与评分一致）
HIGH_TH = 0.6         # 高/低置信分界
LOW_TH = 0.1          # 低于此分数的检测直接丢
MAX_AGE = 30          # 失联多少帧后删除轨迹（30 帧=1.2s 遮挡容忍）
MIN_HITS = 2          # 连续命中多少帧才开始上报


class _Trk:
    __slots__ = ("tid", "box", "hits", "age", "time_since_update")

    def __init__(self, tid: int, box: np.ndarray):
        self.tid, self.box = tid, box
        self.hits, self.age, self.time_since_update = 1, 0, 0


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(n,4) b:(m,4) xyxy → (n,m)。"""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


class IoUTracker:
    """两段关联 IoU 跟踪器。update(dets) 返回 [(track_id, box(4,))]。"""

    def __init__(self, iou_th: float = IOU_TH_HIGH, high_th: float = HIGH_TH,
                 low_th: float = LOW_TH, max_age: int = MAX_AGE,
                 min_hits: int = MIN_HITS):
        self.iou_th, self.high_th, self.low_th = iou_th, high_th, low_th
        self.max_age, self.min_hits = max_age, min_hits
        self.tracks: list[_Trk] = []
        self._next_id = 1

    def _associate(self, dets: np.ndarray, det_idx: list[int],
                   candidates: list[int]) -> dict[int, int]:
        """Hungarian 关联 dets[det_idx] × tracks[candidates] → {det_pos_in_idx: trk_pos}。"""
        if not det_idx or not candidates:
            return {}
        iou = _iou(dets[det_idx][:, :4],
                   np.stack([self.tracks[t].box for t in candidates]))
        valid = iou >= self.iou_th
        if not valid.any():
            return {}
        cost = np.where(valid, 1.0 - iou, 1e6)
        r, c = linear_sum_assignment(cost)
        return {det_idx[ri]: candidates[ci] for ri, ci in zip(r, c)
                if valid[ri, ci]}

    def update(self, dets: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """dets: (n,5) [x1,y1,x2,y2,score]。返回本帧确认的轨迹。"""
        if dets.size == 0:
            dets = np.zeros((0, 5), np.float32)
        for t in self.tracks:
            t.age += 1

        high = [i for i in range(len(dets)) if dets[i, 4] >= self.high_th]
        low = [i for i in range(len(dets)) if self.low_th <= dets[i, 4] < self.high_th]

        matched: dict[int, int] = {}                  # det_idx -> track 下标
        matched_trk: set[int] = set()
        for d, t in self._associate(dets, high,
                                    [i for i, t in enumerate(self.tracks)
                                     if i not in matched_trk]).items():
            matched[d] = t
            matched_trk.add(t)
        for d, t in self._associate(dets, low,
                                    [i for i, t in enumerate(self.tracks)
                                     if i not in matched_trk]).items():
            matched[d] = t
            matched_trk.add(t)

        # 更新命中轨迹
        for d, t in matched.items():
            trk = self.tracks[t]
            trk.box = dets[d, :4].astype(np.float64)
            trk.hits += 1
            trk.time_since_update = 0

        # 未匹配高置信检测 → 新轨迹
        for d in high:
            if d not in matched:
                self.tracks.append(_Trk(self._next_id, dets[d, :4].astype(np.float64)))
                self._next_id += 1

        # 老化 & 上报（刚出生 1 帧的轨迹等 min_hits 确认后再报；失联帧不上报）
        out = []
        alive = []
        for trk in self.tracks:
            if trk.time_since_update == 0 and trk.hits >= self.min_hits:
                out.append((trk.tid, trk.box))
            trk.time_since_update += 1
            if trk.time_since_update <= self.max_age:
                alive.append(trk)
        self.tracks = alive
        return out
