"""主流程：prepare（一次）→ 逐视频 reset/process_frame/finish_video，失败隔离。"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from .decoder import DecoderError, decode_video
from .logging_utils import RunLogger
from .types import RunContext, Solution, VideoMeta
from .writer import PredictionWriter


class Runner:
    def __init__(
        self,
        solution: Solution,
        input_dir: str | Path,
        output_dir: str | Path,
        logger: RunLogger,
        decoder: str = "gpu",
        allow_cpu_fallback: bool = False,
        timeout_sec: int = 7200,
    ):
        self.solution = solution
        self.input_dir = str(input_dir)
        self.output_dir = str(output_dir)
        self.logger = logger
        self.decoder = decoder
        self.allow_cpu_fallback = allow_cpu_fallback
        self.timeout_sec = timeout_sec

    # ── 对外 ────────────────────────────────────────────────────────────────
    def _prepare(self) -> tuple[RunContext | None, float | None]:
        """跑 Solution.prepare；返回 (ctx, prepare_sec) 或 (None, None) 失败（已 logger.error）。"""
        ctx = RunContext(
            input_dir=self.input_dir, output_dir=self.output_dir,
            decoder=self.decoder,
        )
        try:
            t0 = time.monotonic()
            self.solution.prepare(ctx)
            prepare_sec = time.monotonic() - t0
            self.logger.info(f"prepare 完成（{prepare_sec:.1f}s）")
            return ctx, prepare_sec
        except Exception as e:  # prepare 失败 = 全局失败
            self.logger.error("prepare 阶段失败", e)
            return None, None

    def run(self, videos: list[VideoMeta]) -> dict:
        t_start = time.monotonic()
        ctx, prepare_sec = self._prepare()
        if ctx is None:
            return self.logger.finish(
                "failed", meta={"error": "prepare failed（见 run.log）"})

        from core import trt as _trt  # 模块级 import 不依赖 tensorrt，安全

        timing_meta = {
            "solution_info": dict(ctx.info),
            "timing": {
                # 总 wall time 含 engine build（公平性：转换成本计入正式评测）
                "prepare_sec": round(prepare_sec, 1),
                "engine_build_sec": round(_trt.BUILD_SECONDS_TOTAL, 1),
                "infer_sec": None,   # 结束时回填
                "total_sec": None,
            },
        }

        deadline = t_start + self.timeout_sec
        with PredictionWriter(self.output_dir) as writer:
            for meta in videos:
                if time.monotonic() > deadline:
                    self.logger.info(f"全局超时，跳过 {meta.video_id}")
                    self.logger.add_video({
                        "video_id": meta.video_id, "status": "skipped",
                        "frames": 0, "predictions": 0, "sec": 0.0,
                        "error": "global timeout"})
                    continue
                rec = self._run_one(meta, writer)
                self.logger.add_video(rec)
                self.logger.info(
                    f"{meta.video_id} [{meta.game}/{meta.cam}] {rec['status']}: "
                    f"{rec['frames']} 帧 / {rec['predictions']} 框 / {rec['sec']:.1f}s"
                    + ("（cpu_fallback）" if rec.get("cpu_fallback") else ""))

        wall = time.monotonic() - t_start
        infer_sec = wall - prepare_sec
        timing_meta["timing"]["infer_sec"] = round(infer_sec, 1)
        timing_meta["timing"]["total_sec"] = round(wall, 1)
        timing_meta["performance"] = self._performance(infer_sec)
        return self.logger.finish("ok", meta=timing_meta)

    def _performance(self, infer_sec: float) -> dict:
        """工程性能指标：engine build 不计入推理 FPS（单列），但计入 total_sec。"""
        vids = [v for v in self.logger.status["videos"]
                if v["status"] == "ok" and v.get("sec", 0) > 0]
        fps = sorted(v["frames"] / v["sec"] for v in vids)
        total_frames = sum(v["frames"] for v in vids)
        return {
            "total_frames": total_frames,
            "infer_sec": round(infer_sec, 1),
            "infer_fps": round(total_frames / infer_sec, 1) if infer_sec > 0 else 0.0,
            "avg_video_fps": round(sum(fps) / len(fps), 1) if fps else 0.0,
            "p50_video_fps": round(fps[len(fps) // 2], 1) if fps else 0.0,
            "p90_video_fps": round(fps[min(int(len(fps) * 0.9), len(fps) - 1)], 1)
            if fps else 0.0,
        }

    # ── 单视频 ──────────────────────────────────────────────────────────────
    def _run_one(self, meta: VideoMeta, writer: PredictionWriter) -> dict:
        t0 = time.monotonic()
        try:
            frames, preds, timing = self._infer(meta, writer, self.decoder)
            sec = round(time.monotonic() - t0, 2)
            rec = {"video_id": meta.video_id, "status": "ok",
                   "frames": frames, "predictions": preds, "sec": sec,
                   "fps": round(frames / sec, 1) if sec > 0 else 0.0}
            rec.update(timing)
            return rec
        except DecoderError as e:
            if self.decoder == "gpu" and self.allow_cpu_fallback:
                self.logger.info(f"{meta.video_id}: GPU 解码失败（{e}），回退 CPU 重试")
                try:
                    frames, preds, timing = self._infer(meta, writer, "cpu")
                    sec = round(time.monotonic() - t0, 2)
                    rec = {"video_id": meta.video_id, "status": "ok",
                           "frames": frames, "predictions": preds, "sec": sec,
                           "fps": round(frames / sec, 1) if sec > 0 else 0.0,
                           "cpu_fallback": True, "fallback_reason": str(e)}
                    rec.update(timing)
                    return rec
                except DecoderError as e2:
                    return self._failed(meta, e2, t0, "decode")
                except Exception as e2:
                    return self._failed(meta, e2, t0, "solution")
            return self._failed(meta, e, t0, "decode")
        except Exception as e:
            return self._failed(meta, e, t0, "solution")

    def _failed(self, meta: VideoMeta, e: BaseException, t0: float,
                stage: str) -> dict:
        self.logger.error(f"{meta.video_id}: 失败（阶段={stage}）", e)
        return {"video_id": meta.video_id, "status": "failed",
                "failure_stage": stage,  # decode=解码器问题 / solution=参赛算法问题
                "frames": 0, "predictions": 0,
                "sec": round(time.monotonic() - t0, 2),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}

    def _infer(self, meta: VideoMeta, writer: PredictionWriter, mode: str):
        """返回 (frames, preds, {decode_sec, solution_sec, write_sec})。"""
        self.solution.reset(meta)
        frames = 0
        preds = 0
        decode_sec = solution_sec = write_sec = 0.0
        gen = decode_video(meta, self.input_dir, mode)
        try:
            while True:
                t0 = time.monotonic()
                try:
                    fid, img = next(gen)
                except StopIteration:
                    break
                decode_sec += time.monotonic() - t0
                frame_meta = {
                    "video_id": meta.video_id, "frame_id": fid, "pts": None,
                    "width": meta.width, "height": meta.height,
                }
                t0 = time.monotonic()
                out = self.solution.process_frame(img, frame_meta)
                solution_sec += time.monotonic() - t0
                if out:
                    t0 = time.monotonic()
                    preds += writer.write(meta.video_id, out)
                    write_sec += time.monotonic() - t0
                frames += 1
            t0 = time.monotonic()
            tail = self.solution.finish_video(meta)
            solution_sec += time.monotonic() - t0
            if tail:
                t0 = time.monotonic()
                preds += writer.write(meta.video_id, tail)
                write_sec += time.monotonic() - t0
        finally:
            gen.close()
        return frames, preds, {
            "decode_sec": round(decode_sec, 2),
            "solution_sec": round(solution_sec, 2),
            "write_sec": round(write_sec, 2),
        }
