"""predictions.jsonl 写入。每行 {"video_id","frame_id","track_id","bbox"}。"""

from __future__ import annotations

import json
from pathlib import Path

from .types import Track


class PredictionWriter:
    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / "predictions.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, video_id: str, tracks: list[Track]) -> int:
        n = 0
        for t in tracks:
            self._f.write(json.dumps(
                {"video_id": video_id, **t.to_dict()}, ensure_ascii=False) + "\n")
            n += 1
        self.count += n
        return n

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "PredictionWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
