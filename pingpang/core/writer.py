"""predictions.jsonl 写入。每行 {"video_id", "frame_id", "x", "y"}。"""

from __future__ import annotations

import json
from pathlib import Path

from .types import Landing


class PredictionWriter:
    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / "predictions.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, video_id: str, landings: list[Landing]) -> int:
        n = 0
        for l in landings:
            self._f.write(json.dumps(
                {"video_id": video_id, **l.to_dict()}, ensure_ascii=False) + "\n")
            n += 1
        self.count += n
        return n

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "PredictionWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
