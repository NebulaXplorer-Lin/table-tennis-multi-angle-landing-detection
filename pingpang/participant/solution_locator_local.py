"""Local ORT adapter: learned Global/Local centres into the existing heuristic.

Run through the unmodified competition runner with PINGPONG_LOCATOR_CONFIG.
The one-frame lookahead belongs to the fixed3 locator, and retains centre IDs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

# The research helpers are part of this checkout, rather than a pip package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ttnet.data_tools.geometry import letterbox, map_point, crop_transform, crop_image, crop_point
from ttnet.evaluation.metrics import accept, decode_heatmap
from ttnet.training.dataset import normalize
from .solution import Solution as HeuristicSolution
from .solution import CONF_TH, MIN_GAP, FALL_AMP, RISE_AMP


class ReadyDetection:
    """Pass one already aligned centre to the inherited process_frame method."""
    def reset(self):
        self.value = None

    def detect(self, *_):
        return self.value


class Locator:
    def __init__(self, config):
        import torch
        import onnxruntime as ort
        if not torch.cuda.is_available():
            raise RuntimeError('Local locator entry requires CUDA')
        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.config = config
        self.sessions = {}
        for stage in ('global', 'local'):
            path = Path(config[stage + '_onnx'])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != config[stage + '_onnx_sha256']:
                raise ValueError(f'{stage} ONNX identity changed')
            session = ort.InferenceSession(str(path), sess_options=options,
                                          providers=[('CUDAExecutionProvider', {'use_tf32': 0}), 'CPUExecutionProvider'])
            if session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError(f'{stage} CUDA provider unavailable')
            session.disable_fallback()
            self.sessions[stage] = session
        for stage in self.sessions:
            w, h = config[stage + '_input']
            payload = dict(images=np.zeros((1, 3, 3, h, w), np.float32),
                           frame_valid=np.ones((1, 3), bool),
                           time_offsets_sec=np.array([[-.04, 0., .04]], np.float32))
            for _ in range(50):
                self.sessions[stage].run(['heatmap'], payload)

    def heatmap(self, stage, images, valid, offsets):
        return self.sessions[stage].run(['heatmap'], dict(
            images=np.stack(images)[None], frame_valid=np.array([valid], bool),
            time_offsets_sec=np.array([offsets], np.float32)))[0]

    def infer(self, images, valid, offsets, width, height):
        cfg = self.config
        gw, gh = cfg['global_input']
        lw, lh = cfg['local_input']
        batch = []
        transform = None
        for image in images:
            if image is None:
                batch.append(np.zeros((3, gh, gw), np.float32))
            else:
                scaled, transform = letterbox(image, gw, gh)
                batch.append(normalize(scaled))
        x, y, conf = decode_heatmap(self.heatmap('global', batch, valid, offsets))
        x, y = map_point(x, y, transform, inverse=True)
        resolution = dict(width=width, height=height)
        g = accept(dict(x=x, y=y, confidence=conf, source='global'), cfg['global_threshold'], resolution)
        local = None
        if g['valid']:
            crop = crop_transform(x, y, width, height, lw, cfg['local_crop_side_1080'])
            batch = [normalize(crop_image(image, crop)) if image is not None
                     else np.zeros((3, lh, lw), np.float32) for image in images]
            xx, yy, cc = decode_heatmap(self.heatmap('local', batch, valid, offsets))
            xx, yy = crop_point(xx, yy, crop, inverse=True)
            local = accept(dict(x=xx, y=yy, confidence=cc, source='local'), cfg['local_threshold'], resolution)
        selected = local if local and local['valid'] else g
        return dict(global_prediction=g, local_prediction=local,
                    prediction=selected if selected and selected['valid'] else None)


class Solution(HeuristicSolution):
    def prepare(self, context):
        config_path = Path(os.environ['PINGPONG_LOCATOR_CONFIG'])
        self.locator_config = json.loads(config_path.read_text())
        timestamps = Path(self.locator_config['timestamps_path'])
        if hashlib.sha256(timestamps.read_bytes()).hexdigest() != self.locator_config['timestamps_sha256']:
            raise ValueError('source timestamps identity changed')
        self.timestamps = json.loads(timestamps.read_text())
        self.locator = Locator(self.locator_config)
        self.det = ReadyDetection()
        self.output = Path(context.output_dir)
        self.traj = []
        self.last_emit = -10_000
        context.info.update(backend='onnxruntime_cuda_fp32_locator_local',
                            locator=self.locator_config,
                            heuristic=dict(MIN_GAP=MIN_GAP, FALL_AMP=FALL_AMP,
                                           RISE_AMP=RISE_AMP, CONF_TH=CONF_TH),
                            locator_lookahead_frames=1, tracker='independent',
                            timestamp_source='decoded source PTS indexed by original frame ID')

    def reset(self, video_meta):
        super().reset(video_meta)
        self.meta = video_meta
        self.video_timestamps = self.timestamps[video_meta.video_id]
        if len(self.video_timestamps) != video_meta.n_frames:
            raise ValueError('source timestamps and manifest frame count differ')
        self.frames = {}
        self.last_input = -1
        self.last_center = -1
        self.records = []

    def _consume(self, center):
        slots = [self.frames.get(i) for i in (center - 1, center, center + 1)]
        current = self.frames[center]
        valid = [s is not None for s in slots]
        offsets = [s[2] - current[2] if s is not None else 0. for s in slots]
        record = self.locator.infer([s[0] if s is not None else None for s in slots],
                                    valid, offsets, self.meta.width, self.meta.height)
        self.records.append(dict(record, frame_id=center, frame_valid=valid, time_offsets_sec=offsets))
        prediction = record['prediction']
        self.det.value = dict(x=prediction['x'], y=prediction['y'], conf=prediction['confidence']) if prediction else None
        # The original method and _emit are used without changing any rule.
        result = super().process_frame(current[0], current[1])
        self.last_center = center
        for old in list(self.frames):
            if old < center:
                self.frames.pop(old)
        return result

    def process_frame(self, frame_rgb, frame_meta):
        fid = frame_meta['frame_id']
        if fid != self.last_input + 1:
            raise ValueError('locator adapter requires consecutive source frames')
        timestamp = self.video_timestamps[fid]
        self.frames[fid] = (frame_rgb, dict(frame_meta), timestamp)
        self.last_input = fid
        return self._consume(fid - 1) if fid else []

    def finish_video(self, video_meta):
        result = []
        if self.last_center < self.last_input:
            result.extend(self._consume(self.last_input))
        result.extend(super().finish_video(video_meta))
        if len(self.records) != video_meta.n_frames:
            raise ValueError('not every source centre was processed')
        destination = self.output / 'locator_trace' / f'{video_meta.video_id}.jsonl'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(''.join(json.dumps(r) + '\n' for r in self.records))
        return result
