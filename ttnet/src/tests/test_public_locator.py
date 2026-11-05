"""The three-frame adapter must leave the competition heuristic unchanged."""
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from ttnet.common import PROJECT

sys.path.insert(0, str(PROJECT / 'pingpang'))
from participant.solution import Solution as Original
from participant.solution_locator_local import Solution as Adapted, ReadyDetection


class PublicAdapterTests(unittest.TestCase):
    def test_center_ids_boundaries_and_inherited_heuristic(self):
        with tempfile.TemporaryDirectory() as directory:
            for count in (1, 2, 3, 35):
                meta = SimpleNamespace(video_id=str(count), fps=25, width=64, height=48, n_frames=count)
                predictions = [None if i in (5, 13) else dict(x=30., y=float(10 + (i % 12 if i % 12 < 6 else 12-i % 12)), confidence=.99)
                               for i in range(count)]
                class FakeLocator:
                    def __init__(self): self.calls=[]
                    def infer(self, images, valid, offsets, width, height):
                        fid = int(images[1][0, 0, 0]);self.calls.append((fid, valid, offsets))
                        return dict(prediction=predictions[fid])
                reference = Original();reference.det = ReadyDetection();reference.reset(meta)
                expected = []
                times=[fid*.04+fid*fid*.00001 for fid in range(count)]
                adapter = Adapted();adapter.det = ReadyDetection();adapter.locator = FakeLocator();adapter.output = Path(directory)
                adapter.timestamps={str(count):times};adapter.reset(meta)
                actual = []
                for fid, pred in enumerate(predictions):
                    image=np.full((48,64,3), fid, np.uint8)
                    frame_meta=dict(frame_id=fid, width=64, height=48)
                    reference.det.value = dict(x=pred['x'], y=pred['y'], conf=pred['confidence']) if pred else None
                    expected.extend(reference.process_frame(image, frame_meta))
                    actual.extend(adapter.process_frame(image, frame_meta))
                expected.extend(reference.finish_video(meta));actual.extend(adapter.finish_video(meta))
                self.assertEqual([p.to_dict() for p in actual], [p.to_dict() for p in expected])
                self.assertEqual([x[0] for x in adapter.locator.calls], list(range(count)))
                self.assertFalse(adapter.locator.calls[0][1][0]);self.assertFalse(adapter.locator.calls[-1][1][-1])
                self.assertEqual(adapter.traj, reference.traj)
                self.assertIs(Adapted._emit, Original._emit)
                for fid, valid, offsets in adapter.locator.calls:
                    for slot, present in enumerate(valid):
                        self.assertAlmostEqual(offsets[slot], times[fid+slot-1]-times[fid] if present else 0.)


if __name__ == '__main__': unittest.main()
