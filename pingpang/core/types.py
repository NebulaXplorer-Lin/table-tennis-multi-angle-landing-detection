"""接口契约：数据类型 + 参赛者 Solution 基类。

参赛者只需要关心 Solution 与 Landing；Frame/VideoMeta/RunContext 由框架传入。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Landing:
    """一个落点预测。frame_id 是落点实际所属帧（允许比当前处理帧早若干帧）。"""

    frame_id: int
    x: float
    y: float

    def to_dict(self) -> dict:
        return {"frame_id": int(self.frame_id), "x": float(self.x),
                "y": float(self.y)}


@dataclass
class VideoMeta:
    """manifest 中一条视频的元数据。"""

    video_id: str
    path: str
    angle: str
    fps: int
    width: int
    height: int
    n_frames: int

    @classmethod
    def from_entry(cls, e: dict) -> "VideoMeta":
        return cls(video_id=e["video_id"], path=e["path"], angle=e["angle"],
                   fps=int(e["fps"]), width=int(e["width"]),
                   height=int(e["height"]), n_frames=int(e["n_frames"]))


@dataclass
class RunContext:
    """prepare 阶段传给 Solution 的运行环境。只暴露任何参赛者都可能用到的字段；
    框架不假设参赛者用 TRT/ORT——参赛者按需自取（如 engine 缓存目录可从 output_dir 拼）。"""

    input_dir: str          # 数据集根目录（视频相对它寻址）
    output_dir: str         # 输出目录（predictions.jsonl/run_status.json/run.log）
    decoder: str            # "gpu" | "cpu"
    info: dict = field(default_factory=dict)  # 参赛者回填（如 backend），进 run_status
    extra: dict = field(default_factory=dict)


class Solution(ABC):
    """参赛者接口。继承并实现 prepare/reset/process_frame；finish_video 可选。

    生命周期：prepare(一次) → [reset(视频) → process_frame*(帧) → finish_video(视频)]*。
    process_frame 返回"当前已确认"的落点列表（允许内部延迟若干帧再输出）；
    finish_video 返回尾部 flush 的落点。框架负责补 video_id、写文件、校验。
    """

    @abstractmethod
    def prepare(self, context: RunContext) -> None:
        """加载模型 / warmup。TRT engine 转换也应在这里做（不计入视频计时）。"""

    @abstractmethod
    def reset(self, video_meta: VideoMeta) -> None:
        """每个视频开始前调用。"""

    @abstractmethod
    def process_frame(self, frame_rgb: np.ndarray,
                      frame_meta: dict[str, Any]) -> list[Landing]:
        """处理一帧（RGB uint8 HxWx3）。frame_meta 含 video_id/frame_id/pts/width/height。"""

    def finish_video(self, video_meta: VideoMeta) -> list[Landing]:
        """视频结束，flush 尾部延迟结果。默认空实现，无尾帧 flush 可不覆写。"""
        return []
