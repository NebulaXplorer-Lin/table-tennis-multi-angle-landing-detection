"""Bounded, exact-PTS source-video access for disks that cannot hold raw RGB.

PP reproduces the lossless PNG preparation path. TTGames reproduces its original
JPEG95 encode/decode step instead of silently replacing those training pixels.
Decoder versions are part of the backend identity; compare pixels when moving
to a different image. No video is resized or transcoded on disk.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import json
import os
from pathlib import Path

import cv2
import numpy as np

from ttnet.common import PROJECT, identity, jsonl, save, sha

FORMAT = 'source_video_rgb_v1'


def source_path(source, project=PROJECT):
    """Resolve retained provenance paths against the current project first."""
    path = Path(source)
    if 'data' in path.parts:
        candidate = Path(project).joinpath(*path.parts[path.parts.index('data'):])
        if candidate.is_file():
            return candidate.resolve()
    if not path.is_absolute():
        candidate = Path(project)/path
        if candidate.is_file():
            return candidate.resolve()
    if path.is_file():
        return path.resolve()
    raise FileNotFoundError(f'source video not transferred: {source}')


def decoder_identity():
    import av
    return dict(format=FORMAT, pyav=av.__version__, opencv=cv2.__version__,
                libraries={k:list(v) for k,v in av.library_versions.items()},
                code_sha256=sha(Path(__file__)))


def frame_rgb(frame, policy):
    bgr = frame.to_ndarray(format='bgr24')
    if policy == 'jpeg95_roundtrip':
        ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise OSError('cannot reproduce JPEG95 training pixels')
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError('cannot decode reproduced JPEG95 training pixels')
    elif policy != 'lossless_bgr_to_rgb':
        raise ValueError(f'unknown source pixel policy: {policy}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.setflags(write=False)
    return rgb


class VideoReader:
    """One seekable decoder with eight decoded frames, never an entire video."""
    def __init__(self, path, policy, max_frames=8):
        import av
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = 2
        self.decoder = iter(self.container.decode(self.stream))
        first = next(self.decoder)
        if first.pts is None:
            self.close()
            raise ValueError(f'source has no initial PTS: {path}')
        self.first_sec = float(first.pts * first.time_base)
        self.time_base = self.stream.time_base
        self.last_sec = self.first_sec
        self.policy, self.max_frames = policy, max_frames
        self.frames = OrderedDict()
        self.frames[int(round(self.first_sec / self.time_base))] = frame_rgb(first, policy)

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None

    def get(self, timestamp_sec):
        wanted = float(timestamp_sec) + self.first_sec
        tick = round(wanted / self.time_base)
        if abs(float(tick * self.time_base) - wanted) > max(1e-7, float(self.time_base)*.25):
            raise ValueError('stored timestamp does not map to a source PTS tick')
        if tick in self.frames:
            self.frames.move_to_end(tick)
            return self.frames[tick]
        if wanted <= self.last_sec or wanted-self.last_sec > 2.:
            self.container.seek(tick, stream=self.stream, backward=True, any_frame=False)
            self.decoder = iter(self.container.decode(self.stream))
        for frame in self.decoder:
            if frame.pts is None:
                raise ValueError('missing source PTS')
            actual = float(frame.pts * frame.time_base)
            self.last_sec = actual
            current = round(actual / self.time_base)
            if current < tick:
                continue
            if current != tick:
                raise ValueError(f'source PTS missing: expected {wanted}, found {actual}')
            rgb = frame_rgb(frame, self.policy)
            self.frames[tick] = rgb
            while len(self.frames) > self.max_frames:
                self.frames.popitem(last=False)
            return rgb
        raise ValueError(f'incomplete source video at relative time {timestamp_sec}')


def prepare(dataset, output=None, publish=False, project=PROJECT):
    """Build only manifests and indices. Existing labels and split bytes stay put."""
    dataset, project = Path(dataset).resolve(), Path(project).resolve()
    groups = defaultdict(list)
    sources = {}
    if (dataset/'labels/train.jsonl').is_file():
        metadata = json.loads((dataset/'metadata/dataset.json').read_text())
        for split in ('train','val'):
            for row in jsonl(dataset/'labels'/f'{split}.jsonl'):
                groups[row['video_id']].append(dict(frame_id=row['frame_id'],
                    source_frame_id=row['frame_id'], timestamp_sec=row['timestamp_sec']))
        for entry in metadata['source_videos']:
            video = json.loads((dataset/'metadata/videos'/f'{entry["video_id"]}.json').read_text())
            sources[entry['video_id']] = dict(source=entry['video'],
                source_sha256=entry['video_sha256'], pixel_policy='lossless_bgr_to_rgb',
                shape=[video['resolution']['height'],video['resolution']['width'],3])
    else:
        metadata = json.loads((dataset/'dataset.json').read_text())
        for path in sorted(dataset.glob('*/frames.jsonl')):
            meta = json.loads((path.parent/'metadata.json').read_text())
            groups[path.parent.name] = jsonl(path)
            sources[path.parent.name] = dict(source=meta['source'],
                source_sha256=meta['source_identity']['video_sha256'],
                pixel_policy='jpeg95_roundtrip', shape=[meta['height'],meta['width'],3])
        split=json.loads((dataset/'split.json').read_text())
        declared={vid for part in ('train','val','test') for vid in split[part]}
        if set(groups)!=declared:
            raise ValueError('event descriptor is incomplete; transfer all declared match metadata')
    if not groups:
        raise ValueError('no prepared frame indices')
    source_records = {}
    for vid,rows in groups.items():
        rows.sort(key=lambda r:r['frame_id'])
        ids = [r['frame_id'] for r in rows]
        times = [r['timestamp_sec'] for r in rows]
        if len(set(ids)) != len(ids) or any(b<=a for a,b in zip(times,times[1:])):
            raise ValueError(f'invalid frame IDs or timestamps: {vid}')
        source = source_path(sources[vid]['source'], project)
        source_records[vid] = dict(sources[vid], source=source.relative_to(project).as_posix(),
            source_bytes=source.stat().st_size,
            index_identity=identity([(r['frame_id'],r['source_frame_id'],r['timestamp_sec']) for r in rows]))
    storage_id = identity(dict(format=FORMAT,dataset_id=metadata['dataset_id'],videos=source_records))
    output = Path(output).resolve() if output else project/'data/cache/video_sources'/storage_id[:20]
    if (output/'manifest.json').exists():
        previous = json.loads((output/'manifest.json').read_text())
        if previous['storage_id'] != storage_id:
            raise ValueError('video backend destination identity differs')
    for vid,rows in groups.items():
        dest = output/vid
        dest.mkdir(parents=True,exist_ok=True)
        for name,values,dtype in [('frame_ids',[r['frame_id'] for r in rows],np.int64),
                                  ('source_frame_ids',[r['source_frame_id'] for r in rows],np.int64),
                                  ('timestamps',[r['timestamp_sec'] for r in rows],np.float64)]:
            temporary = dest/f'{name}.partial.npy'
            np.save(temporary,np.asarray(values,dtype=dtype),allow_pickle=False)
            temporary.replace(dest/f'{name}.npy')
        record = dict(source_records[vid],schema=1,status='completed',frames=len(rows),
                      dtype='uint8',color='RGB',chunks=[],
                      source=os.path.relpath(project/source_records[vid]['source'],output))
        save(dest/'manifest.json',record)
    save(output/'manifest.json',dict(schema=1,status='completed',format=FORMAT,
         storage_id=storage_id,dataset_id=metadata['dataset_id'],
         videos={vid:dict(frames=len(rows)) for vid,rows in groups.items()},
         policy='decode by recorded PTS; retain original frame IDs; bounded RAM; no RGB disk cache'))
    reference = dict(schema=1,path=os.path.relpath(output,dataset),storage_id=storage_id,
                     logical_dataset_id=metadata['dataset_id'],format=FORMAT)
    if publish:
        old = dataset/'frame_store.json'
        backup = dataset/'frame_store.before_video.json'
        if old.is_file() and not backup.exists():
            backup.write_bytes(old.read_bytes())
        save(old,reference)
    return dict(reference=reference,store=str(output),videos=len(groups),
                frames=sum(map(len,groups.values())),published=publish,
                runtime_identity=decoder_identity())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--publish',action='store_true',help='switch this descriptor to source-video access')
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset,args.output,args.publish),indent=2))
