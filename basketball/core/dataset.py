"""manifest 加载与视频遍历。"""

from __future__ import annotations

import json
from pathlib import Path

from .types import VideoMeta


def load_manifest(path: str | Path) -> list[VideoMeta]:
    """读 manifest.json → VideoMeta 列表（保持文件顺序）。"""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    videos = [VideoMeta.from_entry(e) for e in doc["videos"]]
    ids = [v.video_id for v in videos]
    assert len(ids) == len(set(ids)), "manifest 存在重复 video_id"
    return videos


def resolve_manifest(input_dir: str | Path, explicit: str | None = None) -> Path:
    """解析 manifest 路径：显式参数 > <input>/manifest.json > 仓库 manifest_public.json。

    候选必须是本比赛 schema（顶层含 "videos"）；数据集自带的打包 manifest.json
    （name/counts/entries，无 videos 键）会被跳过。
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        candidates.append(Path(input_dir) / "manifest.json")
        # 镜像自带公开验证集（15 条，含 manifest+GT），开箱即测
        candidates.append(Path(__file__).resolve().parent.parent / "public_data" / "manifest.json")
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            doc = json.loads(cand.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get("videos"), list):
            return cand
        if explicit:
            raise ValueError(f"manifest 缺少 videos 列表: {cand}")
    raise FileNotFoundError(
        f"未找到合法 manifest：请用 --manifest 指定，或把含 \"videos\" 的 manifest.json "
        f"放进 {input_dir}"
    )
