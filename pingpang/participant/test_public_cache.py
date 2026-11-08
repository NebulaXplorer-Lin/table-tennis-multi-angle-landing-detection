import unittest
import numpy as np
from participant.solution_public_cached import GlobalFrameCache


class CacheTests(unittest.TestCase):
    def test_exact_values_boundaries_eviction_and_reset(self):
        rng = np.random.default_rng(44)
        cache = GlobalFrameCache((16, 8))
        for n in (1, 2, 3, 8, 30):
            cache.clear(reset_stats=True)
            frames = {i:rng.integers(0,256,(17,29,3),dtype=np.uint8) for i in range(n)}
            for center in range(n):
                images = {i:frames[i] for i in range(max(0,center-1),min(n,center+2))}
                cache.activate(images, center)
                for fid in range(center-1,center+2):
                    if fid not in images:
                        padding=np.zeros((8,16,3),np.uint8)
                        np.testing.assert_array_equal(cache.normalize(padding),cache.original_normalize(padding))
                        continue
                    expected, transform = cache.original_letterbox(images[fid],16,8)
                    actual, actual_transform = cache.letterbox(images[fid],16,8)
                    np.testing.assert_array_equal(actual,expected)
                    self.assertEqual(actual_transform,transform)
                    np.testing.assert_array_equal(cache.normalize(actual),cache.original_normalize(expected))
                self.assertLessEqual(len(cache.entries),3)
            self.assertEqual(cache.stats['resize_calls'],n)
            self.assertEqual(cache.stats['normalize_calls'],n)
            self.assertLessEqual(cache.stats['max_entries'],3)
            cache.clear()
            self.assertEqual(cache.entries,{})

    def test_local_crops_and_instances_are_independent(self):
        first,second=GlobalFrameCache((16,8)),GlobalFrameCache((16,8))
        for cache,value in ((first,20),(second,240)):
            raw=np.full((17,29,3),value,np.uint8)
            cache.activate({0:raw},0)
            boxed,_=cache.letterbox(raw,16,8)
            normalized=cache.normalize(boxed)
            np.testing.assert_array_equal(normalized,cache.original_normalize(boxed))
            local=np.full((4,4,3),value,np.uint8)
            np.testing.assert_array_equal(cache.normalize(local),cache.original_normalize(local))
        self.assertFalse(np.array_equal(first.entries[0][2],second.entries[0][2]))


if __name__=='__main__':unittest.main()
