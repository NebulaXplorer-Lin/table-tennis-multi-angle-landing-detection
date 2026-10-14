"""Ball-only temporal/coordinate and evaluation regression checks.

    .venv/bin/python scripts/test_baseline_ball.py
"""
import unittest

import numpy as np

from evaluate_baseline_ball import BallDetector, aggregate, score


class BaselineBallTests(unittest.TestCase):
    def test_middle_frame_alignment_scaling_and_reset(self):
        # An injected model reads the marker from each normalized RGB frame.
        # Unlike a fixed heatmap fixture, its output depends on the actual stack.
        def infer(x):
            self.assertEqual(x.shape, (1, 9, 288, 512))
            heat = np.zeros((1, 3, 288, 512), np.float32)
            for channel in range(3):
                red = x[0, channel * 3]
                y, px = np.unravel_index(red.argmax(), red.shape)
                if red[y, px] > 0:
                    heat[0, channel, y, px] = .9
            return heat

        det = BallDetector(infer, 'unused.onnx')
        frames = []
        for px in [50, 100, 150]:
            rgb = np.zeros((288, 512, 3), np.uint8)
            rgb[80, px, 0] = 255
            frames.append(rgb)
        self.assertIsNone(det.detect(frames[0], 1024, 576))
        pred = det.detect(frames[1], 1024, 576)
        self.assertEqual((pred['x'], pred['y']), (100, 160))
        pred = det.detect(frames[2], 1024, 576)
        self.assertEqual((pred['x'], pred['y']), (200, 160))
        det.reset()
        self.assertIsNone(det.detect(frames[2], 1024, 576))

    def test_strict_heat_threshold_and_all_blob_centroid(self):
        heat = np.zeros((1, 3, 288, 512), np.float32)
        heat[0, 1, 40, 20] = .8
        heat[0, 1, 40, 100] = 1.
        heat[0, 1, 40, 200] = .7  # exactly threshold: must be excluded
        heat[0, 0, 200, 400] = 1.  # other frame channel: must be excluded
        det = BallDetector(lambda _: heat, 'unused.onnx')
        p = det.detect(np.zeros((288, 512, 3), np.uint8), 512, 288)
        weight = float(heat[0, 1, 40, 20])
        self.assertAlmostEqual(p['x'], (20 * weight + 100) / (weight + 1))
        self.assertEqual(p['y'], 40)
        self.assertEqual(p['conf'], 1.)

    def test_wrong_location_counts_false_positive_and_false_negative(self):
        m = score([[0, 0, 2, 2]], dict(x=40, y=40, conf=.9), 10, .7)
        self.assertEqual((m['tp'], m['fp'], m['fn']), (0, 1, 1))
        m = score([[0, 0, 2, 2]], None, 10, .7)
        self.assertEqual((m['tp'], m['fp'], m['fn']), (0, 0, 1))

    def test_one_to_one_match_and_inclusive_distance(self):
        m = score([[0, 0, 2, 2], [10, 0, 12, 2]], dict(x=1, y=1, conf=.9), 10, .7)
        self.assertEqual((m['tp'], m['fp'], m['fn']), (1, 0, 1))
        m = score([[0, 0, 2, 2]], dict(x=11, y=1, conf=.9), 10, .7)
        self.assertEqual(m['tp'], 1)

    def test_unknown_frame_and_video_boundary_break_miss_runs(self):
        rows = [dict(video='a', frame=f, boxes=[[0, 0, 2, 2]] if f != 2 else [],
                     label_status='positive' if f != 2 else 'unknown', prediction=None)
                for f in range(5)]
        rows.append(dict(video='b', frame=0, boxes=[[0, 0, 2, 2]],
                         label_status='positive', prediction=None))
        m = aggregate(rows)
        self.assertEqual(m['max_consecutive_missed'], 2)
        self.assertEqual(m['fn'], 5)
        self.assertEqual(m['fp'], 0)


if __name__ == '__main__':
    unittest.main()
