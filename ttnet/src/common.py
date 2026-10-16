from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
CONFIGS = SRC / "configs"
PROJECT = ROOT.parent
DATA_ROOT = PROJECT / "data"
CACHE = DATA_ROOT / "cache"
OUTPUT = PROJECT / "out/results"
RUNS = PROJECT / "out/experiments"
DATA = DATA_ROOT / "datasets/pp_ball_v2"
EVENT_DATA = DATA_ROOT / "datasets/openttgames_event25_v3"


def output_directory(root=ROOT):
    return OUTPUT if Path(root).resolve() == ROOT.resolve() else Path(root)/'output'


def run_directory(root=ROOT):
    return RUNS if Path(root).resolve() == ROOT.resolve() else Path(root)/'runs'


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)
