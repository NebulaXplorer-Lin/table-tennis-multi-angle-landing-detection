"""Fill sparse event context from the original video into the common NPY store."""
from collections import OrderedDict
from fractions import Fraction
import json
from pathlib import Path

import av
from ttnet.common import CACHE, identity, save, sha
from ttnet.data_tools.frame_store import FrameStore, pack_video
from ttnet.events.prepare import nearest_index


def ensure_continuous_context(corpus, vid):
    video = corpus.videos[vid];meta = video['meta'];frames = video['frames']
    if sorted(frames) == list(range(meta['n_frames'])):
        return
    source = Path(meta['source'])
    key = identity(dict(source_sha256=sha(source),dataset=corpus.identity,video=vid,
                        fps=meta['fps'],version=1))
    root = CACHE/'continuous_context'/key[:20]
    descriptor = root/'descriptor'
    existing = dict(frames)
    rows = []
    if not all(p.exists() for p in (root/vid/'manifest.json',root/'manifest.json',root/'rows.json',descriptor/'frame_store.json')):
        with av.open(str(source)) as container:
            stream=container.streams.video[0];rate=Fraction(stream.average_rate)
            mapping={nearest_index(Fraction(i)*rate/Fraction(str(meta['fps']))):i for i in range(meta['n_frames'])}
            if len(mapping)!=meta['n_frames']:
                raise ValueError('continuous context cannot duplicate source frames')
            first=None
            for sid,frame in enumerate(container.decode(stream)):
                if frame.pts is None:raise ValueError('missing PTS')
                ts=float(frame.pts*frame.time_base)
                if first is None:first=ts
                if sid in mapping:
                    fid=mapping[sid]
                    rows.append(dict(existing.get(fid,dict(frame_id=fid,source_frame_id=sid,label_status='unknown',xy=None,path='')),timestamp_sec=ts-first))
        if len(rows)!=meta['n_frames']:raise ValueError('incomplete continuous context')
        # Bounded sequential reader. pack_video re-reads one chunk for verification.
        with av.open(str(source)) as container:
            stream=container.streams.video[0];stream.thread_type='AUTO'
            decoder=enumerate(container.decode(stream));recent=OrderedDict()
            def reader(row):
                fid=row['frame_id']
                if fid in recent:return recent[fid]
                for sid,frame in decoder:
                    if sid==row['source_frame_id']:break
                else:raise ValueError('incomplete source decode')
                rgb=corpus.image(vid,fid) if fid in existing else frame.to_ndarray(format='rgb24')
                recent[fid]=rgb
                while len(recent)>32:recent.popitem(last=False)
                return rgb
            manifest=pack_video(root,vid,rows,reader,source_identity=key)
        save(root/'rows.json',rows)
        save(root/'manifest.json',dict(status='completed',storage_id=key,videos={vid:manifest}))
        save(descriptor/'frame_store.json',dict(path='..',storage_id=key))
    else:
        rows=json.loads((root/'rows.json').read_text())
    corpus.context_stores[vid]=FrameStore(descriptor)
    video['frames']={r['frame_id']:r for r in rows}
    corpus.image.cache_clear()


def stream_context_features(system,corpus,vid,needed_ids):
    """Run association from frame zero, retaining only requested descriptors.

    Fill gaps by decoding source video with a one-frame lookahead. Existing
    sampled NPY pixels take precedence, so there is no second raw-frame cache.
    """
    from fractions import Fraction
    import numpy as np
    original=corpus.videos[vid];meta=original['meta'];existing=original['frames']
    wanted=set(needed_ids)
    if not wanted:return
    if min(wanted)<0 or max(wanted)>=meta['n_frames']:raise ValueError('context feature ID out of range')
    class Window:
        def __init__(self):self.videos={vid:dict(meta=meta,frames={})};self.images={}
        def image(self,v,f):return self.images[f]
    window=Window();frames=window.videos[vid]['frames'];last=max(wanted);seen=set();finished=False
    system.reset_tracking(vid)
    with av.open(str(meta['source'])) as container:
        stream=container.streams.video[0];rate=Fraction(stream.average_rate);stream.thread_type='AUTO'
        stop=min(last+1,meta['n_frames']-1)
        mapping={nearest_index(Fraction(i)*rate/Fraction(str(meta['fps']))):i for i in range(stop+1)}
        if len(mapping)!=stop+1:raise ValueError('duplicate source frames in context sampling')
        first=None;previous=None
        for sid,frame in enumerate(container.decode(stream)):
            if frame.pts is None:raise ValueError('missing PTS')
            ts=float(frame.pts*frame.time_base)
            if first is None:first=ts
            if sid not in mapping:continue
            fid=mapping[sid];actual=ts-first
            if previous is not None and actual<=previous:raise ValueError('non-monotonic context timestamps')
            previous=actual
            if fid in existing:
                row=existing[fid]
                if row['source_frame_id']!=sid or abs(row['timestamp_sec']-actual)>1e-7:raise ValueError('source context mapping changed')
                rgb=corpus.image(vid,fid)
            else:
                row=dict(frame_id=fid,source_frame_id=sid,timestamp_sec=actual,label_status='unknown',xy=None)
                rgb=frame.to_ndarray(format='rgb24')
            frames[fid]=row;window.images[fid]=rgb
            if fid:
                center=fid-1
                d,r,_=system.forward_frames(window,vid,[center]);seen.add(center)
                if center in wanted:yield center,d[center],r[center]
                for old in list(window.images):
                    if old<center:window.images.pop(old);frames.pop(old,None)
                if center==last:finished=True;break
            if fid%2000==0:print(f'context {vid}: {fid}/{last+1}',flush=True)
        if not finished and stop==last and last in frames:
            d,r,_=system.forward_frames(window,vid,[last]);seen.add(last)
            if last in wanted:yield last,d[last],r[last]
        if not wanted<=seen:raise ValueError('incomplete chronological context decode')
