"""YOLO 球定位 + 原有落点启发式。

**参赛者直接改本文件（participant/ 就是你的代码区）**——本文件实现 run.py 调用的
四个生命周期方法（prepare/reset/process_frame/finish_video），TRT 转换在 prepare
里直接写明，落点判定算法在本类内部。

落点启发式：图像 y 向下为正，"下落转上升"= 球心 y 的局部极大值。每帧重扫已积累
轨迹（短片 O(n) 可忽略），配下落/反弹最小幅度门控；MIN_GAP 帧内只出一个；
尾帧在 finish_video 降级判定。参数在 public 验证集上粗调过（见 README 基线）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.types import Landing, RunContext, VideoMeta

from .base_solution import BaseSolution
from .yolo_ball_detector import YoloBallDetector

MIN_GAP = 10         # 两次落点最小帧间隔（数据集回合节奏 ~10-12 帧/落点）
# The detector threshold is applied inside YoloBallDetector.  Keep this equal
# to it so that every selected YOLO centre reaches the unchanged heuristic.
CONF_TH = 0.05
FALL_AMP = 2.0       # 判定前的最小下落幅度（px，图像 y）
RISE_AMP = 1.0       # 判定后的最小反弹幅度（px，图像 y）


class Solution(BaseSolution):
    """落点启发式：球心 y 的局部极大值 + 下落/反弹门控。"""

    def prepare(self, context: RunContext) -> None:
        # 1. 权重：只带 ONNX，TRT engine 现场构建（首跑约 5 分钟，之后从缓存秒级加载）
        onnx_path = Path(__file__).resolve().parent / "weights" / "yolo_ball_full_v1.onnx"
        assert onnx_path.exists(), f"缺少权重: {onnx_path}"

        # 2. TRT 转换：core.trt.load_or_build 自动处理缓存（按版本/GPU/ONNX 哈希分目录）
        #    缓存目录由参赛者自己定——这里放到 <output>/trt_cache
        from core.trt import load_or_build
        trt_cache = Path(context.output_dir) / "trt_cache"
        model, how = load_or_build(
            onnx_path, trt_cache, input_name="images",
            shape=(1, 3, 1280, 1280), log=print)

        # 3. 检测器：只管预处理 + 推理 + 解码，推理函数由 prepare 注入
        self.det = YoloBallDetector(model.infer, size=1280, score_threshold=CONF_TH)

        # 4. warmup：TRT 首次推理会触发 kernel JIT 编译，跑一次 dummy 把 kernel
        #    都预热好，避免第一视频第一帧把这个时间算进 infer_sec
        self.det.detect(np.zeros((1080, 1920, 3), np.uint8), 1920, 1080)

        # 5. 回填 backend 给 run_status（让平台看到是 built 还是 loaded）
        context.info["backend"] = f"tensorrt_{how}"

        self.traj: list[tuple[int, float, float, float]] = []
        self.last_emit = -10_000

    def reset(self, video_meta: VideoMeta) -> None:
        self.det.reset()
        self.traj = []
        self.last_emit = -10_000

    def process_frame(self, frame_rgb, frame_meta) -> list[Landing]:
        fid = frame_meta["frame_id"]
        det = self.det.detect(frame_rgb, frame_meta["width"], frame_meta["height"])
        if det and det["conf"] >= CONF_TH:
            self.traj.append((fid, det["x"], det["y"], det["conf"]))
        # YOLO output is aligned to the input frame; wait for two future
        # trajectory points to confirm the same local-maximum heuristic.
        return self._emit(need_future=2)

    def finish_video(self, video_meta: VideoMeta) -> list[Landing]:
        # 尾帧降级：至少 1 个未来点也允许判定，避免丢视频末尾落点
        return self._emit(need_future=1)

    # ── 内部 ────────────────────────────────────────────────────────────────
    def _emit(self, need_future: int) -> list[Landing]:
        out: list[Landing] = []
        n = len(self.traj)
        for j in range(1, n - need_future):  # 保证 traj[j+1..j+need_future] 存在
            fid, x, y, conf = self.traj[j]
            if fid - self.last_emit < MIN_GAP:
                continue
            prev_y = self.traj[j - 1][2]
            next_y = self.traj[j + 1][2]
            if (y > prev_y and y >= next_y                       # 局部极大
                    and (y - prev_y) >= FALL_AMP                  # 之前在下落
                    and (y - next_y) >= RISE_AMP):                # 之后在反弹
                out.append(Landing(frame_id=fid, x=x, y=y))
                self.last_emit = fid
        return out
