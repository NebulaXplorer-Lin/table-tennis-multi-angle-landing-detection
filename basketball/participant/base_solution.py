"""官方示例基类：参赛者继承本类即可获得通用流程兜底，只需实现 process_frame。

run.py 调用的四个生命周期方法：
- prepare / reset / process_frame：必须实现（core.types.Solution 已标 @abstractmethod）
- finish_video：可选，默认 return []（无尾帧 flush 需求时可不覆写）

参赛者直接继承 core.types.Solution 时前三个必须自己写；
继承本 BaseSolution 时 prepare/reset/finish_video 有默认实现兜底，最薄子类只需实现 process_frame。
"""

from __future__ import annotations

from abc import abstractmethod

from core.types import RunContext, Solution, Track, VideoMeta


class BaseSolution(Solution):
    """通用流程兜底：四个生命周期方法的默认实现，跟踪算法留给子类。"""

    def prepare(self, context: RunContext) -> None:
        """默认空实现：参赛者按需覆写（典型如加载模型、TRT 转换、warmup）。"""
        pass

    def reset(self, video_meta: VideoMeta) -> None:
        """每个视频开始前调用。参赛者按需覆写清状态（跟踪器、track_id 计数）。"""
        pass

    def finish_video(self, video_meta: VideoMeta) -> list[Track]:
        """当前视频所有帧 process_frame 完成后被调一次，flush 因"延迟输出"
        而积压在内部缓冲的尾部 Track。立即输出型算法可不覆写（默认返回 []）。"""
        return []

    @abstractmethod
    def process_frame(self, frame_rgb, frame_meta) -> list[Track]:
        """逐帧推理核心：接收 RGB 帧和元数据，返回本帧确认的 Track 列表。
        子类必须实现。允许内部延迟若干帧再输出（尾部在 finish_video flush）。"""
