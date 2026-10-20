"""Audit a published frame store without changing any dataset metadata."""
import json
from pathlib import Path

import numpy as np

from ttnet.common import sha
from .frame_store import FrameStore


def verify_store(dataset,full=False):
    store=FrameStore(dataset)
    if store.root is None:raise ValueError('dataset has no array store')
    manifest=json.loads((store.root/'manifest.json').read_text())
    count=chunks=0
    for video_id in manifest['videos']:
        video,ids=store._video(video_id)
        timestamps=np.load(store.root/video_id/'timestamps.npy',allow_pickle=False)
        sources=np.load(store.root/video_id/'source_frame_ids.npy',allow_pickle=False)
        if len(timestamps)!=len(ids) or len(sources)!=len(ids):raise ValueError('metadata array length mismatch')
        if np.any(np.diff(timestamps)<=0):raise ValueError('timestamps must be strictly increasing')
        if store.is_video:
            source=store.source_path(video_id)
            if source.stat().st_size!=video['source_bytes']:
                raise ValueError(f'source size mismatch: {source}')
            if full and sha(source)!=video['source_sha256']:
                raise ValueError(f'source checksum mismatch: {source}')
            # Seek to real beginning, middle and end indices; never fabricate gaps.
            for pos in sorted({0,len(ids)//2,len(ids)-1}):
                image=store.get_rgb(video_id,int(ids[pos]))
                if list(image.shape)!=video['shape']:
                    raise ValueError(f'source frame dimensions changed: {video_id}')
        for record in video['chunks']:
            path=store.root/video_id/record['path'];mapped=np.load(path,mmap_mode='r',allow_pickle=False)
            if list(mapped.shape)!=record['shape'] or mapped.dtype!=np.uint8 or path.stat().st_size!=record['bytes']:
                raise ValueError(f'invalid chunk: {path}')
            if full and sha(path)!=record['sha256']:raise ValueError(f'checksum mismatch: {path}')
            chunks+=1
        count+=len(ids)
    store.close()
    result=dict(passed=True,dataset=str(dataset),storage_id=manifest['storage_id'],frames=count,
                videos=len(manifest['videos']),chunks=chunks,full_checksums=full)
    if store.is_video:result['pixel_backend']=store.backend_identity()
    return result
