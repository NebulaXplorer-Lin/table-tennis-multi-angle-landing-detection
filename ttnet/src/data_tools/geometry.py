from __future__ import annotations

import cv2
import numpy as np


def letterbox_transform(w, h, width=512, height=288):
    ratio = min(width / w, height / h)
    rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (width - rw) // 2, (height - rh) // 2
    return dict(scale_x=rw/w, scale_y=rh/h, pad_left=left, pad_top=top,
                     resized_width=rw, resized_height=rh, width=width, height=height,
                     original_width=w, original_height=h)


def letterbox(image, width=512, height=288):
    h, w = image.shape[:2]
    t = letterbox_transform(w, h, width, height)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    left, top, rw, rh = t["pad_left"], t["pad_top"], t["resized_width"], t["resized_height"]
    out[top:top+rh, left:left+rw] = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return out, t


def map_point(x, y, transform, inverse=False):
    if inverse:
        return ((x-transform["pad_left"])/transform["scale_x"],
                (y-transform["pad_top"])/transform["scale_y"])
    return (x*transform["scale_x"]+transform["pad_left"],
            y*transform["scale_y"]+transform["pad_top"])


def crop_transform(x, y, width, height, output_size=256, side_at_1080=256):
    # Keep a fixed field of view even at image boundaries: pad, never shift.
    side = max(2, round(side_at_1080 * height / 1080))
    left, top = round(x-side/2), round(y-side/2)
    return dict(left=left, top=top, side=side, output_size=output_size,
                scale=output_size/side, original_width=width, original_height=height)


def crop_image(image, transform):
    t = transform
    side, left, top = t["side"], t["left"], t["top"]
    out = np.zeros((side, side, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, left), max(0, top), min(w, left+side), min(h, top+side)
    if x2 > x1 and y2 > y1:
        out[y1-top:y2-top, x1-left:x2-left] = image[y1:y2, x1:x2]
    return cv2.resize(out, (t["output_size"], t["output_size"]), interpolation=cv2.INTER_LINEAR)


def crop_point(x, y, transform, inverse=False):
    if inverse:
        return x/transform["scale"]+transform["left"], y/transform["scale"]+transform["top"]
    return (x-transform["left"])*transform["scale"], (y-transform["top"])*transform["scale"]


def window_indices(timestamps, center, half_span=1/30):
    """Nearest real boundaries, centered slots; edges have explicit missing slots."""
    times = np.asarray(timestamps, dtype=np.float64)
    if not len(times) or not 0 <= center < len(times):
        raise ValueError("invalid center")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must strictly increase")
    tc = times[center]
    left = int(np.argmin(np.abs(times-(tc-half_span))))
    right = int(np.argmin(np.abs(times-(tc+half_span))))
    ids = list(range(left, right+1))
    # Supports faster-than-120fps sources without exceeding the fixed interface.
    before, after = [i for i in ids if i < center], [i for i in ids if i > center]
    if len(before) > 4:
        before = [before[i] for i in np.linspace(0, len(before)-1, 4).round().astype(int)]
    if len(after) > 4:
        after = [after[i] for i in np.linspace(0, len(after)-1, 4).round().astype(int)]
    slots = [None] * 9
    for slot, i in zip(range(4-len(before), 4), before):
        slots[slot] = i
    slots[4] = center
    for slot, i in zip(range(5, 5+len(after)), after):
        slots[slot] = i
    return dict(frame_ids=slots, frame_valid=[i is not None for i in slots],
                time_offsets_sec=[0.0 if i is None else float(times[i]-tc) for i in slots],
                center_frame_id=center, actual_span_sec=float(times[right]-times[left]))
