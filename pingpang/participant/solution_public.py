"""Public gated EventSystem through the official streaming participant API.

Set TTNET_CHECKPOINT to the trusted full best.pt, or place it at
participant/weights/public_best.pt. No video files or labels are read here.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

from core.types import Landing, Solution as SolutionInterface

# run.py adds pingpang/, while the research package lives one directory above it.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ttnet.events.data import assemble_window
from ttnet.events.engine import EventSystem
from ttnet.events.inference import StreamingCorpus
from ttnet.scene.visual_cache import batch_windows


class StreamingPeaks:
    """Exact bounce-only equivalent of metrics.peaks on chronological rows.

    Greedy NMS can depend on an arbitrarily long chain of nearby candidates.
    Resolve a whole connected component once a gap exceeds the NMS radius;
    imposing a fixed timeout would change the offline algorithm on such chains.
    Invalid ball positions still participate in NMS, as in the research run.
    """

    def __init__(self, threshold, nms_sec):
        self.threshold = float(threshold)
        self.radius = float(nms_sec) + 1e-9
        self.previous = None
        self.left_score = -math.inf
        self.pending = []

    def _resolve(self):
        chosen = []
        for row in sorted(self.pending,
                          key=lambda r: (-r['probabilities'][0], r['frame_id'])):
            if all(abs(row['timestamp_sec'] - c['timestamp_sec']) > self.radius
                   for c in chosen):
                chosen.append(row)
        self.pending = []
        return sorted(chosen, key=lambda r: r['frame_id'])

    def _decide(self, right_score):
        row = self.previous
        score = row['probabilities'][0]
        output = []
        if score >= self.threshold and score >= self.left_score and score >= right_score:
            if self.pending and row['timestamp_sec'] - self.pending[-1]['timestamp_sec'] > self.radius:
                output.extend(self._resolve())
            self.pending.append(row)
        # All potential peaks through this time now have their right neighbor.
        if self.pending and row['timestamp_sec'] - self.pending[-1]['timestamp_sec'] > self.radius:
            output.extend(self._resolve())
        return output

    def push(self, row):
        output = []
        if self.previous is not None:
            if row['frame_id'] != self.previous['frame_id'] + 1:
                raise ValueError('event scores must be contiguous')
            output = self._decide(row['probabilities'][0])
            self.left_score = self.previous['probabilities'][0]
        self.previous = row
        return output

    def finish(self):
        output = self._decide(-math.inf) if self.previous is not None else []
        output.extend(self._resolve())
        self.previous = None
        return output


class Solution(SolutionInterface):
    def prepare(self, context):
        checkpoint = Path(os.environ.get(
            'TTNET_CHECKPOINT', Path(__file__).parent / 'weights' / 'public_best.pt'))
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Set TTNET_CHECKPOINT to the complete public checkpoint best.pt: {checkpoint}')
        # Full trusted training checkpoint includes optimizer/RNG Python objects.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.config = dict(state['config'])
        cfg = self.config
        if cfg.get('event_window_mode', 'frames') != 'frames' or cfg.get('window', 13) != 13:
            raise ValueError('This adapter requires the public checkpoint 13-frame window')
        if cfg.get('event_head_type') != 'tcn' or cfg.get('table_source') == 'oracle':
            raise ValueError('Expected TCN with predicted table geometry')
        torch.set_num_threads(int(cfg.get('torch_threads', 4)))
        device = torch.device(os.environ.get('TTNET_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.system = EventSystem(cfg, device, initialize=False)
        self.system.load_state_dict(state['system'], strict=True)
        self.system.configure(cfg.get('mode', 'frozen'), False)
        self.threshold = float(state['thresholds'][0])
        self.nms_sec = float(cfg.get('nms_sec', .08))
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('Invalid checkpoint event threshold')
        if not math.isfinite(self.nms_sec) or self.nms_sec < 0:
            raise ValueError('Invalid NMS radius')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        context.info.update(
            backend='PyTorch', device=str(device), seed=cfg.get('seed'), epoch=state.get('epoch'),
            checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            threshold=self.threshold, nms_sec=self.nms_sec,
            precision='AMP' if cfg.get('amp') and device.type == 'cuda' else 'FP32',
            localizer_future_frames=1, event_future_frames=6,
            nms_delay='one score neighbor + connected-component closure',
            timestamp_source='frame_id / manifest.fps (official pts is unavailable)',
        )

    def reset(self, video_meta):
        if video_meta.n_frames <= 0 or video_meta.fps <= 0:
            raise ValueError('Video must contain frames and have a positive frame rate')
        self.meta = video_meta
        self.vid = video_meta.video_id
        self.corpus = StreamingCorpus(self.vid, dict(
            n_frames=video_meta.n_frames, width=video_meta.width,
            height=video_meta.height, fps=video_meta.fps))
        self.features = {}
        self.records = {}
        self.received = 0
        self.localized = -1
        self.next_center = 0
        self.finished = False
        self.peaks = StreamingPeaks(self.threshold, self.nms_sec)
        self.stats = dict(max_images=0, max_features=0, max_records=0,
                          max_metadata=0, max_nms_candidates=0, scores=0,
                          emitted_during_stream=0, emitted_at_finish=0)
        self.system.reset_tracking(self.vid)

    def _amp(self):
        return torch.autocast(device_type=self.system.device.type,
                              enabled=bool(self.config.get('amp')) and self.system.device.type == 'cuda')

    def _localize(self, fid):
        descriptors, records, _ = self.system.forward_frames(self.corpus, self.vid, [fid])
        self.features[fid] = descriptors[fid].float().cpu()
        self.records.update(records)
        self.localized = fid

    def _landings(self, selected):
        result = []
        for row in selected:
            p = row.get('prediction')
            if not p or not p['valid']:
                continue
            x, y = float(p['x']), float(p['y'])
            if not (math.isfinite(x) and math.isfinite(y) and
                    0 <= x < self.meta.width and 0 <= y < self.meta.height):
                raise ValueError('Localizer returned an invalid original-image coordinate')
            result.append(Landing(frame_id=int(row['frame_id']), x=x, y=y))
        return result

    def _drain(self, final=False):
        output = []
        frames = self.corpus.videos[self.vid]['frames']
        while self.next_center < self.meta.n_frames and (final or self.next_center + 6 <= self.localized):
            center = self.next_center
            inputs = assemble_window(self.corpus, self.vid, center, self.features,
                                     self.records, torch.device('cpu'), self.config)
            tensors = batch_windows([inputs], self.system.device)
            scores = self.system.head(*tensors).float().sigmoid().cpu().numpy()[0].tolist()
            row = dict(frame_id=center, timestamp_sec=frames[center]['timestamp_sec'],
                       probabilities=scores, prediction=self.records[center]['prediction'])
            output.extend(self._landings(self.peaks.push(row)))
            self.stats['scores'] += 1
            self.next_center += 1
            # The first frame of the next window also needs its predecessor for velocity.
            keep = max(0, self.next_center - 7)
            for cache in (self.features, self.records, frames):
                for fid in list(cache):
                    if fid < keep:
                        del cache[fid]
        return output

    def _measure(self):
        for key, cache in (('images', self.corpus.images), ('features', self.features),
                           ('records', self.records), ('metadata', self.corpus.videos[self.vid]['frames']),
                           ('nms_candidates', self.peaks.pending)):
            self.stats[f'max_{key}'] = max(self.stats[f'max_{key}'], len(cache))

    @torch.inference_mode()
    def process_frame(self, frame_rgb, frame_meta):
        if self.finished:
            raise RuntimeError('reset must precede a new video')
        fid = frame_meta['frame_id']
        if frame_meta['video_id'] != self.vid or fid != self.received or fid >= self.meta.n_frames:
            raise ValueError('Expected complete chronological frames for the current video')
        if frame_rgb.shape != (self.meta.height, self.meta.width, 3) or frame_rgb.dtype != np.uint8:
            raise ValueError('Expected RGB uint8 at the original video resolution')
        frames = self.corpus.videos[self.vid]['frames']
        # pts has no defined unit in the official interface and is currently always None.
        frames[fid] = dict(timestamp_sec=fid / self.meta.fps, source_frame_id=fid,
                           label_status='unknown', xy=None)
        self.corpus.images[fid] = frame_rgb.copy()
        self.received += 1
        self._measure()
        with self._amp():
            if fid >= 1:
                self._localize(fid - 1)
                self.corpus.images.pop(fid - 2, None)
            self._measure()
            output = self._drain()
        self._measure()
        self.stats['emitted_during_stream'] += len(output)
        return output

    @torch.inference_mode()
    def finish_video(self, video_meta):
        if video_meta.video_id != self.vid:
            raise ValueError('finish_video metadata does not match reset')
        if self.finished:
            return []
        if self.received != self.meta.n_frames:
            raise ValueError(f'Incomplete video: {self.received}/{self.meta.n_frames}')
        with self._amp():
            self._localize(self.meta.n_frames - 1)
            self.corpus.images.clear()
            self._measure()
            output = self._drain(final=True)
            output.extend(self._landings(self.peaks.finish()))
        self.stats['emitted_at_finish'] += len(output)
        self.features.clear()
        self.records.clear()
        self.corpus.videos[self.vid]['frames'].clear()
        self.finished = True
        if self.system.device.type == 'cuda':
            torch.cuda.synchronize(self.system.device)
        return output
