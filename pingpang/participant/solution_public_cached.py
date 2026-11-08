"""Reuse global letterbox/normalization; retain all three-frame model calls."""
from types import FunctionType, MethodType

from participant.solution_public import Solution as Base
from ttnet.tracking import runtime


class GlobalFrameCache:
    def __init__(self, size):
        self.size = tuple(size)
        self.original_letterbox = runtime.letterbox
        self.original_normalize = runtime.normalize
        self.clear(reset_stats=True)

    def clear(self, reset_stats=False):
        self.entries = {}
        self.sources = {}
        self.boxed = {}
        if reset_stats:
            self.stats = dict(resize_calls=0, normalize_calls=0, resize_hits=0,
                              normalize_hits=0, max_entries=0, max_bytes=0)

    def activate(self, images, center):
        wanted = set(range(center-1, center+2))
        for fid in list(self.entries):
            if fid not in wanted:
                del self.entries[fid]
        self.sources = {id(image): fid for fid, image in images.items() if fid in wanted}
        self.boxed = {id(entry[0]): entry for entry in self.entries.values()}

    def letterbox(self, image, width, height):
        fid = self.sources.get(id(image))
        if fid is None or (width, height) != self.size:
            return self.original_letterbox(image, width, height)
        if fid not in self.entries:
            boxed, transform = self.original_letterbox(image, width, height)
            self.entries[fid] = [boxed, transform, None]
            self.boxed[id(boxed)] = self.entries[fid]
            self.stats['resize_calls'] += 1
        else:
            self.stats['resize_hits'] += 1
        self._measure()
        return self.entries[fid][0], self.entries[fid][1]

    def normalize(self, image):
        entry = self.boxed.get(id(image))
        # Local crops and boundary padding retain the original implementation.
        if entry is None:
            return self.original_normalize(image)
        if entry[2] is None:
            entry[2] = self.original_normalize(image)
            self.stats['normalize_calls'] += 1
        else:
            self.stats['normalize_hits'] += 1
        self._measure()
        return entry[2]

    def _measure(self):
        self.stats['max_entries'] = max(self.stats['max_entries'], len(self.entries))
        size = sum(entry[0].nbytes+(entry[2].nbytes if entry[2] is not None else 0)
                   for entry in self.entries.values())
        self.stats['max_bytes'] = max(self.stats['max_bytes'], size)


class Solution(Base):
    def prepare(self, context):
        super().prepare(context)
        if self.config.get('tracker_type') != 'association':
            raise ValueError('Global cache requires chronological association')
        self.global_cache = GlobalFrameCache(self.config['global_input'])
        self.cache_video_stats = []
        context.info['global_cache_video_stats'] = self.cache_video_stats
        # Use unchanged association code with instance-local preprocessing bindings.
        # No module globals or other Solution instances are monkeypatched.
        original = runtime.associated_frames
        bindings = dict(original.__globals__)
        bindings.update(letterbox=self.global_cache.letterbox, normalize=self.global_cache.normalize)
        associated = FunctionType(original.__code__, bindings, original.__name__,
                                  original.__defaults__, original.__closure__)
        self.system._forward_frames = MethodType(associated, self.system)
        context.info['global_preprocess_cache'] = dict(max_frames=3, scope='letterbox + normalize only',
                                                       padding='unchanged', model_windows='unchanged')

    def reset(self, meta):
        super().reset(meta)
        self.global_cache.clear(reset_stats=True)

    def _localize(self, fid):
        self.global_cache.activate(self.corpus.images, fid)
        return super()._localize(fid)

    def finish_video(self, meta):
        try:
            already_finished = self.finished
            result = super().finish_video(meta)
            if not already_finished:
                self.cache_video_stats.append(dict(video_id=meta.video_id, frames=meta.n_frames, **self.global_cache.stats))
            return result
        finally:
            self.global_cache.clear()
