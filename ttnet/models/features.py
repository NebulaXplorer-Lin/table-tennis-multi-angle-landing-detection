from __future__ import annotations

import numpy as np

from ttnet.data_tools.geometry import window_indices


class FeatureWindowCache:
    """Video-local frame descriptors with explicit data/checkpoint identity."""
    def __init__(self, dataset_id, global_sha256, local_sha256):
        self.identity = dict(dataset_id=dataset_id, global_sha256=global_sha256, local_sha256=local_sha256)
        self.reset(None)

    def reset(self, video_id):
        self.video_id = video_id
        self.frames = []

    def append(self, frame_id, timestamp_sec, global_features, local_features, local_valid,
               localization, crop_transform=None):
        if frame_id != len(self.frames):
            raise ValueError("append complete, ordered frames; reset between videos")
        g = np.asarray(global_features, dtype=np.float32)
        l = np.asarray(local_features, dtype=np.float32) if local_valid else np.zeros(128, np.float32)
        if g.shape != (128,) or l.shape != (128,):
            raise ValueError("each branch descriptor must have 128 dimensions")
        if not np.isfinite(g).all() or not np.isfinite(l).all():
            raise ValueError("nonfinite descriptor")
        if self.frames and timestamp_sec <= self.frames[-1]["timestamp_sec"]:
            raise ValueError("timestamps must strictly increase")
        self.frames.append(dict(frame_id=frame_id, timestamp_sec=timestamp_sec, features=np.concatenate([g,l]),
                                local_valid=bool(local_valid), localization=localization, crop_transform=crop_transform))

    def window(self, center):
        index = window_indices([f["timestamp_sec"] for f in self.frames], center)
        features = np.zeros((1,9,256), np.float32)
        local_valid = np.zeros((1,9), bool)
        crops = [None]*9
        for slot, fid in enumerate(index["frame_ids"]):
            if fid is not None:
                features[0,slot] = self.frames[fid]["features"]
                local_valid[0,slot] = self.frames[fid]["local_valid"]
                crops[slot] = self.frames[fid]["crop_transform"]
        return dict(features=features, frame_valid=np.array([index["frame_valid"]],bool),
                    local_valid=local_valid, time_offsets_sec=np.array([index["time_offsets_sec"]],np.float32),
                    video_id=self.video_id, center_frame_id=center,
                    localization=self.frames[center]["localization"], crop_transforms=crops,
                    identity=self.identity)
