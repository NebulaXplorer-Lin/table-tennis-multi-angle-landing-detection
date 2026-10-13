"""官方示例方案：球员检测 + IoU 两段关联跟踪（接入样例，非精度基线）。

**参赛者直接改 participant/ 后提交即可**——本文件实现 run.py 调用的
四个生命周期方法（prepare/reset/process_frame/finish_video），TRT 转换在 prepare
里直接写明，跟踪逻辑由 tracker.py 提供。

后端选择：优先 TensorRT（engine 现场构建，首跑约 5 分钟，之后缓存秒级加载）；
环境无 TRT（如本地 CPU 调试）时回退 onnxruntime（CUDA EP 优先）。backend 回填
run_status 的 solution_info，便于排查"评测跑的是哪个后端"。
"""

from __future__ import annotations

import numpy as np

from core.types import RunContext, Track, VideoMeta

from .base_solution import BaseSolution
from .player_detector import IN_H, IN_W, PlayerDetector, weights_path
from .tracker import IoUTracker

CLASSES = [0]         # player.onnx（YOLO26n COCO80 预训练导出）为 80 类 → 只留 person（自训单类模型改 None）


class Solution(BaseSolution):
    """YOLO 检测 + ByteTrack 式两段 IoU 关联。"""

    def prepare(self, context: RunContext) -> None:
        onnx_path = weights_path()          # 缺权重时这里给出可操作报错

        # 1) 首选 TensorRT：engine 缓存 <output>/trt_cache（core.trt 按版本/GPU/哈希分目录）
        model, backend = None, None
        try:
            from core.trt import load_or_build
            from pathlib import Path
            trt_cache = Path(context.output_dir) / "trt_cache"
            input_name, shape = self._onnx_input_meta(onnx_path)
            model, how = load_or_build(
                onnx_path, trt_cache, input_name=input_name, shape=shape, log=print)
            backend = f"tensorrt_{how}"
            infer = model.infer
        except Exception as e:              # TRT 不可用（本地调试机无 TRT/GPU 等）
            print(f"[demo] TRT 不可用（{type(e).__name__}: {e}），回退 onnxruntime")
            import onnxruntime as ort
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(str(onnx_path), providers=prov)
            iname = sess.get_inputs()[0].name
            infer = lambda x: sess.run(None, {iname: x})[0]
            backend = "onnxruntime_" + (sess.get_providers()[0]
                                        .removesuffix("ExecutionProvider").lower())

        # 2) 检测器 + 跟踪器；warmup 把首帧 kernel JIT 成本挡在计时外
        #    layout="e2e"：player.onnx 是 YOLO26 端到端导出（输出 [1,300,6] 图内已 NMS）
        self.det = PlayerDetector(infer, input_size=(IN_H, IN_W), classes=CLASSES,
                                  layout="e2e")
        self.det.detect(np.zeros((720, 1280, 3), np.uint8))
        self.tracker = IoUTracker()

        context.info["backend"] = backend

    @staticmethod
    def _onnx_input_meta(onnx_path) -> tuple[str, tuple[int, ...]]:
        """读 ONNX 图的输入名与 shape（动态维度落到 letterbox 目标尺寸）。"""
        import onnx
        m = onnx.load(str(onnx_path))
        inp = m.graph.input[0]
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            v = d.dim_value if d.dim_value > 0 else -1
            dims.append(v)
        dims = [1 if d == -1 and i == 0 else
                (IN_H if d == -1 and i == 2 else (IN_W if d == -1 and i == 3 else d))
                for i, d in enumerate(dims)]
        return inp.name, tuple(dims)

    def reset(self, video_meta: VideoMeta) -> None:
        self.tracker = IoUTracker()         # track_id 每视频局部：这里清零

    def process_frame(self, frame_rgb, frame_meta) -> list[Track]:
        fid = frame_meta["frame_id"]
        dets = self.det.detect(frame_rgb)
        return [Track(frame_id=fid, track_id=tid,
                      bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])))
                for tid, b in self.tracker.update(dets)]

    def finish_video(self, video_meta: VideoMeta) -> list[Track]:
        # demo 是逐帧即时输出，无延迟缓冲；延迟输出型算法在这里 flush
        return []
