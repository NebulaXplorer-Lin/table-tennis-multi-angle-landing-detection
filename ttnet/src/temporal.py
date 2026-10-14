"""Shared localization/event sampling contract; native fixed3 vs time-based masked9."""
from __future__ import annotations

import numpy as np

from ttnet.data_tools.geometry import window_indices

MODES = ("single", "fixed3", "fps_mask9")


def temporal_mode(config):
    mode = (config or {}).get("temporal_mode", "single")
    if mode not in MODES:
        raise ValueError(f"temporal_mode must be one of {MODES}")
    return mode


def slots_for_mode(mode):
    return {"single": 1, "fixed3": 3, "fps_mask9": 9}[mode]


def sample_window(timestamps, center, mode="fixed3", half_span=1/30):
    times = np.asarray(timestamps, dtype=np.float64)
    if len(times) == 0 or not 0 <= center < len(times):
        raise ValueError("invalid temporal center")
    if len(times)>1 and np.any(np.diff(times)<=0):
        raise ValueError("non-monotonic timestamps")
    if mode == "fps_mask9":
        return window_indices(times, center, half_span)
    if mode not in ("single", "fixed3"):
        raise ValueError("unknown temporal mode")
    ids = [center] if mode == "single" else [center-1, center, center+1]
    ids = [i if 0 <= i < len(times) else None for i in ids]
    real = [i for i in ids if i is not None]
    return dict(frame_ids=ids, frame_valid=[i is not None for i in ids],
                time_offsets_sec=[0. if i is None else float(times[i]-times[center]) for i in ids],
                center_frame_id=center, actual_span_sec=float(times[real[-1]]-times[real[0]]))


def move_inputs(batch, device):
    if isinstance(batch, dict):
        return tuple(batch[k].to(device, non_blocking=True) for k in ["images", "frame_valid", "time_offsets_sec"])
    return (batch.to(device, non_blocking=True),)


def batch_size(batch):
    return len(batch["images"]) if isinstance(batch, dict) else len(batch)
