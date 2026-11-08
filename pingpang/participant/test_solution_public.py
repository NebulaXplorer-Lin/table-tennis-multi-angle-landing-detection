"""Streaming/offline equivalence tests; no checkpoint or video data needed."""
import unittest

import numpy as np
import torch

from core.types import VideoMeta
from participant.solution_public import Solution, StreamingPeaks
from ttnet.events.data import assemble_window
from ttnet.events.inference import StreamingCorpus
from ttnet.events.metrics import peaks


class PeakTests(unittest.TestCase):
    def test_equivalent_to_offline_including_ties_and_long_chains(self):
        rng = np.random.default_rng(44)
        sequences = [[], [0.8], [0.9] * 100, [0.8, 0, 0.9, 0, 1, 0, 0.7]]
        sequences += [rng.integers(0, 11, size=300).astype(float) / 10 for _ in range(100)]
        for fps in (25, 30, 60):
            for scores in sequences:
                rows = [dict(frame_id=i, timestamp_sec=i/fps, probabilities=[float(p), 0.])
                        for i, p in enumerate(scores)]
                online = StreamingPeaks(.3, .08)
                result = []
                for row in rows:
                    result += online.push(row)
                result += online.finish()
                self.assertEqual([r['frame_id'] for r in result],
                                 [r['frame_id'] for r in peaks(rows, [.3, 2.], .08)])
                self.assertEqual(online.finish(), [])


def record(fid):
    return dict(prediction=dict(x=float(fid % 4), y=2., valid=True, confidence=.9),
                local_valid=True, table=dict(visual=dict(roi=torch.full((32,), float(fid)), context=torch.tensor([float(fid), -float(fid)]), roi_valid=fid % 3 != 0, context_valid=True)))


class FakeHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, *inputs):
        self.inputs.append(tuple({k:v.clone() for k,v in x.items()} if isinstance(x,dict) else x.clone() for x in inputs))
        fid = int(inputs[0][0, 6, 0])
        return torch.tensor([[3. if fid % 5 == 0 else -3., -10.]])


class FakeSystem:
    device = torch.device('cpu')

    def __init__(self):
        self.head = FakeHead()

    def reset_tracking(self, vid):
        pass


class FakeSolution(Solution):
    def __init__(self):
        self.system = FakeSystem()
        self.config = dict(event_window_mode='frames', amp=False, table_fusion=dict(mode="gated", channels=2))
        self.threshold = .3
        self.nms_sec = .08

    def _localize(self, fid):
        frames = self.corpus.videos[self.vid]['frames']
        for i in (fid-1, fid, fid+1):
            if 0 <= i < self.meta.n_frames:
                assert i in self.corpus.images
                assert i in frames
        self.features[fid] = torch.full((256,), float(fid))
        self.records[fid] = record(fid)
        self.localized = fid


class WindowTests(unittest.TestCase):
    def test_all_windows_short_videos_tail_and_reset(self):
        solution = FakeSolution()
        for n in (1, 2, 6, 7, 8, 13, 14, 20, 200):
            meta = VideoMeta('test', '', 'test', 25, 8, 6, n)
            solution.system.head.inputs.clear()
            solution.reset(meta)
            result = []
            for fid in range(n):
                result += solution.process_frame(np.zeros((6, 8, 3), dtype=np.uint8),
                                                 dict(video_id='test', frame_id=fid))
            result += solution.finish_video(meta)
            self.assertEqual(solution.finish_video(meta), [])
            self.assertEqual(solution.stats['scores'], n)
            self.assertLessEqual(solution.stats['max_images'], 3)
            self.assertLessEqual(solution.stats['max_features'], 14)
            self.assertLessEqual(solution.stats['max_records'], 14)
            self.assertLessEqual(solution.stats['max_metadata'], 15)
            if n == 200:
                self.assertGreater(solution.stats['emitted_during_stream'], 0)
            corpus = StreamingCorpus('test', dict(n_frames=n, width=8, height=6, fps=25))
            corpus.videos['test']['frames'] = {i: dict(timestamp_sec=i/25) for i in range(n)}
            features = {i: torch.full((256,), float(i)) for i in range(n)}
            records = {i: record(i) for i in range(n)}
            for fid, actual in enumerate(solution.system.head.inputs):
                expected = assemble_window(corpus, 'test', fid, features, records,
                                           torch.device('cpu'), solution.config)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual([r.frame_id for r in result], list(range(0, n, 5)))

    def test_incomplete_or_out_of_order_frames_fail(self):
        solution = FakeSolution()
        meta = VideoMeta('test', '', '', 25, 8, 6, 3)
        solution.reset(meta)
        with self.assertRaises(ValueError):
            solution.finish_video(meta)
        with self.assertRaises(ValueError):
            solution.process_frame(np.zeros((6, 8, 3), np.uint8),
                                   dict(video_id='test', frame_id=1))


if __name__ == '__main__':
    unittest.main()
