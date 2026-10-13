"""接口契约：数据类型 + 参赛者 Solution 基类。

参赛者只需要关心 Solution 与 Track；Frame/VideoMeta/RunContext 由框架传入。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Track:
    """一个球员跟踪预测：某帧某条轨迹的 bbox。

    frame_id 是该预测实际所属帧（0 基），允许比当前处理帧早若干帧（延迟输出）。
    bbox 为画面像素坐标 [x1, y1, x2, y2]，图像坐标系原点左上、y 向下为正。
    track_id 每视频局部正整数即可（评分离线做全局最优 id 映射，命名无关）。
    """

    frame_id: int
    track_id: int
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {"frame_id": int(self.frame_id), "track_id": int(self.track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]}


@dataclass
class VideoMeta:
    """manifest 中一条视频的元数据。"""

    video_id: str
    path: str
    game: str        # 比赛场次（如 "骑士vs鹈鹕"）——划分与统计的分组单位
    cam: str         # 机位/画面类型："main" | "switch"
    fps: float
    width: int
    height: int
    n_frames: int

    @classmethod
    def from_entry(cls, e: dict) -> "VideoMeta":
        return cls(video_id=e["video_id"], path=e["path"], game=e["game"],
                   cam=e["cam"], fps=float(e["fps"]), width=int(e["width"]),
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
    process_frame 返回"当前已确认"的 Track 列表（允许内部延迟若干帧再输出，
    延迟上限 = 该视频结束）；finish_video 返回尾部 flush 的 Track。
    框架负责补 video_id、写文件、校验。
    """

    @abstractmethod
    def prepare(self, context: RunContext) -> None:
        """加载模型 / warmup。TRT engine 转换也应在这里做（不计入视频计时）。"""

    @abstractmethod
    def reset(self, video_meta: VideoMeta) -> None:
        """每个视频开始前调用。track_id 计数等每视频状态在这里清零。"""

    @abstractmethod
    def process_frame(self, frame_rgb: np.ndarray,
                      frame_meta: dict[str, Any]) -> list[Track]:
        """处理一帧（RGB uint8 HxWx3）。frame_meta 含 video_id/frame_id/pts/width/height。"""

    def finish_video(self, video_meta: VideoMeta) -> list[Track]:
        """视频结束，flush 尾部延迟结果。默认空实现，无尾帧 flush 可不覆写。"""
        return []
