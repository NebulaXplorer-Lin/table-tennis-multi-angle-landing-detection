"""Read-only RGB frame stores, independent of datasets, models and input sizes."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from ttnet.common import identity, save, sha


def image_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise OSError(f"unreadable image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class FrameStore:
    """Sparse frame IDs map to dense chunk rows; never fabricate a missing frame.

    Dataset descriptors can reference an external store through frame_store.json.
    Legacy image datasets remain readable when no reference exists. Mappings are
    worker-local and bounded; fork/spawn never shares an open writable buffer.
    """
    def __init__(self, dataset, max_open_chunks=8, legacy=False):
        self.dataset = Path(dataset)
        self.max_open_chunks = max_open_chunks
        self.root = None
        reference = self.dataset/'frame_store.json'
        if reference.is_file() and not legacy:
            ref = json.loads(reference.read_text())
            self.root = (self.dataset/ref['path']).resolve()
            manifest = json.loads((self.root/'manifest.json').read_text())
            if manifest.get('status') != 'completed' or manifest['storage_id'] != ref['storage_id']:
                raise ValueError('incomplete or mismatched frame store')
        self._chunks = OrderedDict()
        self._videos = {}
        self._pid = os.getpid()
        self._lock = threading.RLock()

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_chunks'] = OrderedDict()
        state['_videos'] = {}
        state.pop('_lock',None)
        return state

    def __setstate__(self,state):
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def _video(self, video_id):
        if self._pid != os.getpid():
            self._chunks.clear()
            self._pid = os.getpid()
        if video_id not in self._videos:
            manifest = json.loads((self.root/video_id/'manifest.json').read_text())
            if manifest.get('status') != 'completed':
                raise ValueError(f'incomplete video store: {video_id}')
            ids = np.load(self.root/video_id/'frame_ids.npy', allow_pickle=False)
            if len(ids) != manifest['frames'] or np.any(np.diff(ids) <= 0):
                raise ValueError(f'invalid frame index: {video_id}')
            self._videos[video_id] = (manifest, ids)
        return self._videos[video_id]

    def get_rgb(self, video_id, frame_id, *, path=None):
        if self.root is None:
            return self._get_rgb(video_id,frame_id,path=path)
        with self._lock:
            return self._get_rgb(video_id,frame_id,path=path)

    def _get_rgb(self, video_id, frame_id, *, path=None):
        if self.root is None:
            if path is None:
                raise KeyError(f'legacy frame requires its image path: {video_id}/{frame_id}')
            image = image_rgb(self.dataset/path)
            image.setflags(write=False)
            return image
        manifest, ids = self._video(video_id)
        pos = int(np.searchsorted(ids, frame_id))
        if pos >= len(ids) or ids[pos] != frame_id:
            raise KeyError(f'missing frame: {video_id}/{frame_id}')
        chunk_id, offset = divmod(pos, manifest['chunk_frames'])
        record = manifest['chunks'][chunk_id]
        key = (video_id, chunk_id)
        if key not in self._chunks:
            chunk_path = self.root/video_id/record['path']
            if chunk_path.stat().st_size != record['bytes']:
                raise ValueError(f'damaged chunk: {chunk_path}')
            chunk = np.load(chunk_path, mmap_mode='r', allow_pickle=False)
            if chunk.dtype != np.uint8 or list(chunk.shape) != record['shape']:
                raise ValueError(f'invalid chunk shape or dtype: {chunk_path}')
            self._chunks[key] = chunk
            while len(self._chunks) > self.max_open_chunks:
                self._chunks.popitem(last=False)
        self._chunks.move_to_end(key)
        return self._chunks[key][offset]

    def row_rgb(self, row):
        return self.get_rgb(row['video_id'], row['frame_id'], path=row.get('frame_path', row.get('path')))


def pack_video(root, video_id, rows, reader, *, chunk_frames=32, source_identity=None):
    """Atomically publish checked chunks. Reusing a completed chunk verifies it.

    reader(row) must return the exact previous RGB pixels. The persisted digest
    covers the dense RGB byte stream; every newly written frame is compared with
    its source through a read-only mapping before that chunk is committed.
    """
    if chunk_frames < 1:
        raise ValueError('chunk_frames must be positive')
    rows = sorted(rows, key=lambda r:r['frame_id'])
    ids = np.asarray([r['frame_id'] for r in rows], np.int64)
    if not len(ids) or np.any(np.diff(ids) <= 0):
        raise ValueError('frame IDs must be unique and increasing')
    dest = Path(root)/video_id
    dest.mkdir(parents=True, exist_ok=True)
    first = reader(rows[0])
    shape = first.shape
    if first.dtype != np.uint8 or len(shape) != 3 or shape[-1] != 3:
        raise ValueError('expected HWC RGB uint8')
    key = identity(dict(source=source_identity, frame_ids=ids.tolist(), shape=shape,
                        chunk_frames=chunk_frames, format='npy_rgb_u8_v1'))
    chunks = []
    for start in range(0, len(rows), chunk_frames):
        group = rows[start:start+chunk_frames]
        index = start//chunk_frames
        path = dest/f'{index:06d}.npy'
        marker = dest/f'{index:06d}.json'
        if path.exists() and marker.exists():
            record = json.loads(marker.read_text())
            if record.get('source_key') == key and sha(path) == record['sha256']:
                chunks.append(record)
                continue
            raise ValueError(f'chunk identity or checksum mismatch: {path}')
        temporary = path.with_suffix('.partial.npy')
        mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8,
                                          shape=(len(group), *shape))
        expected = hashlib.sha256()
        for i, row in enumerate(group):
            image = reader(row)
            if image.shape != shape or image.dtype != np.uint8:
                raise ValueError(f'frame dimensions changed within video {video_id}')
            mapped[i] = image
            expected.update(image.tobytes())
        mapped.flush()
        del mapped
        checked = np.load(temporary, mmap_mode='r', allow_pickle=False)
        actual = hashlib.sha256()
        for i, row in enumerate(group):
            if not np.array_equal(checked[i], reader(row)):
                raise ValueError(f'RGB pixel mismatch: {video_id}/{row["frame_id"]}')
            actual.update(checked[i].tobytes())
        if actual.digest() != expected.digest():
            raise ValueError('RGB stream checksum mismatch')
        del checked
        temporary.replace(path)
        record = dict(path=path.name, bytes=path.stat().st_size,
                      shape=[len(group), *shape], sha256=sha(path),
                      rgb_sha256=actual.hexdigest(), source_key=key)
        save(marker, record)
        chunks.append(record)
    for name, array in [('frame_ids', ids),
                        ('source_frame_ids', np.asarray([r.get('source_frame_id',r['frame_id']) for r in rows],np.int64)),
                        ('timestamps', np.asarray([r['timestamp_sec'] for r in rows],np.float64))]:
        temporary = dest/f'{name}.partial.npy'
        np.save(temporary, array, allow_pickle=False)
        temporary.replace(dest/f'{name}.npy')
    result = dict(schema=1, status='completed', video_id=video_id, frames=len(rows),
                  shape=list(shape), dtype='uint8', color='RGB', chunk_frames=chunk_frames,
                  source_key=key, chunks=chunks, pixel_verification='all frames array_equal to legacy decoder')
    save(dest/'manifest.json', result)
    return result
