"""Causal beam association. Search predictions are never returned as observations."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

DEFAULTS = dict(top_k=5, beam_size=8, history=6, reset_after=3,
                min_confidence=.1, motion_scale_px=80., motion_cap=2.,
                motion_weight=.6, appearance_weight=.2, missing_cost=2.5,
                roi_merge_fraction=.5)


@dataclass
class Hypothesis:
    history: tuple = ()
    costs: tuple = ()
    missing: int = 0

    @property
    def score(self):
        return sum(self.costs)


class BeamTracker:
    def __init__(self, config=None):
        self.config = dict(DEFAULTS, **(config or {}))
        unknown = set(self.config)-set(DEFAULTS)
        if unknown:
            raise ValueError(f'unknown association parameters: {sorted(unknown)}')
        c = self.config
        if any(type(c[k]) is not int or c[k] < 1 for k in ('top_k','beam_size','history','reset_after')):
            raise ValueError('association counts must be positive integers')
        if not 0 < c['min_confidence'] <= 1 or c['motion_scale_px'] <= 0:
            raise ValueError('invalid association confidence or motion scale')
        if any(c[k] < 0 for k in ('motion_cap','motion_weight','appearance_weight','missing_cost','roi_merge_fraction')):
            raise ValueError('association costs must be nonnegative')
        self.reset()

    def reset(self):
        self.beam = [Hypothesis()]
        self.last_frame = self.last_time = None

    def _prediction(self, h, timestamp):
        if not h.history or h.missing > self.config['reset_after']:
            return None
        t, p = h.history[-1]
        xy = np.array([p['x'], p['y']], dtype=float)
        if len(h.history) > 1:
            pt, prev = h.history[-2]
            xy += (xy-np.array([prev['x'],prev['y']])) * ((timestamp-t)/(t-pt))
        return dict(x=float(xy[0]), y=float(xy[1]), valid=False, predicted=True,
                    missing=h.missing, timestamp_sec=timestamp)

    def predict(self, timestamp):
        return self._prediction(self.beam[0], timestamp)

    def update(self, candidates, frame_id, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('timestamp must be finite')
        if self.last_frame is not None and (frame_id <= self.last_frame or timestamp <= self.last_time):
            raise ValueError('association requires strictly increasing video frames and timestamps')
        gap = 0 if self.last_frame is None else frame_id-self.last_frame-1
        c = self.config
        observations = [dict(p) for p in candidates if p.get('valid', True)
                        and np.isfinite([p['x'],p['y'],p['confidence']]).all()
                        and p['confidence'] >= c['min_confidence']]
        extensions = []
        for h in self.beam:
            missing = h.missing + gap
            history = h.history if missing <= c['reset_after'] else ()
            base = Hypothesis(history, h.costs, missing)
            predicted = self._prediction(base, timestamp)
            for p in observations:
                cost = -math.log(max(1e-8, min(1., p['confidence'])))
                if predicted is not None:
                    distance = math.hypot(p['x']-predicted['x'], p['y']-predicted['y'])
                    cost += c['motion_weight'] * min(c['motion_cap'], distance/(c['motion_scale_px']*(1+missing)))
                    a, b = history[-1][1].get('appearance'), p.get('appearance')
                    if a is not None and b is not None:
                        cost += c['appearance_weight'] * min(1., float(np.linalg.norm(np.array(a)-b)))
                extensions.append((Hypothesis((history+((timestamp,p),))[-c['history']:],
                                               (h.costs+(cost,))[-c['history']:],0), p))
            miss = missing+1
            extensions.append((Hypothesis(history if miss <= c['reset_after'] else (),
                                           (h.costs+(c['missing_cost'],))[-c['history']:],miss), None))
        extensions.sort(key=lambda item:item[0].score)
        # Retain distinct recent paths, including a missing branch. Every step
        # still evaluates all Global candidates, even after a long loss.
        self.beam = [h for h,_ in extensions[:c['beam_size']]]
        selected = extensions[0][1]
        self.last_frame, self.last_time = frame_id, timestamp
        return selected

    def state(self, timestamp):
        return dict(prediction=self.predict(timestamp), missing=self.beam[0].missing,
                    hypotheses=len(self.beam), observation_valid=self.beam[0].missing == 0)


def appearance(rgb, candidate, radius=5):
    """Small normalized RGB patch descriptor; no annotation dependency."""
    x,y = int(round(candidate['x'])),int(round(candidate['y']))
    patch = rgb[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]
    if not patch.size:
        return None
    return (patch.mean(axis=(0,1))/255.).tolist()
