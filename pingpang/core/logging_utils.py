"""run.log 与 run_status.json。注意本模块故意不叫 logging（避免遮蔽 stdlib）。"""

from __future__ import annotations

import datetime as _dt
import json
import traceback
from pathlib import Path


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


class RunLogger:
    """单文件 run.log + 结构化 run_status。"""

    def __init__(self, output_dir: str | Path):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._f = (out / "run.log").open("a", encoding="utf-8")
        self.status: dict = {
            "status": "running",
            "started_at": _now(),
            "videos": [],
            "totals": {"videos": 0, "ok": 0, "failed": 0, "skipped": 0,
                       "cpu_fallback": 0, "frames": 0, "predictions": 0},
        }

    def info(self, msg: str) -> None:
        line = f"{_now()} [INFO] {msg}"
        print(line, flush=True)
        self._f.write(line + "\n")
        self._f.flush()

    def error(self, msg: str, exc: BaseException | None = None) -> None:
        line = f"{_now()} [ERROR] {msg}"
        if exc is not None:
            line += "\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(line, flush=True)
        self._f.write(line + "\n")
        self._f.write("".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__)) if exc is not None else "")
        self._f.flush()

    def add_video(self, record: dict) -> None:
        self.status["videos"].append(record)
        t = self.status["totals"]
        t["videos"] += 1
        t[record["status"]] = t.get(record["status"], 0) + 1
        if record.get("cpu_fallback"):
            t["cpu_fallback"] += 1
        t["frames"] += record.get("frames", 0)
        t["predictions"] += record.get("predictions", 0)

    def finish(self, final_status: str, meta: dict | None = None) -> dict:
        self.status["status"] = final_status
        self.status["finished_at"] = _now()
        if meta:
            self.status.update(meta)
        if self.status["totals"]["failed"] and final_status == "ok":
            self.status["status"] = "partial_failed"
        return self.status

    def dump_status(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "run_status.json").write_text(
            json.dumps(self.status, ensure_ascii=False, indent=1), encoding="utf-8")

    def close(self) -> None:
        self._f.close()
