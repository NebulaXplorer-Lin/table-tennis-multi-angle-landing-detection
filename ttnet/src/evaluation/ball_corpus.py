"""Native full-frame PP corpus adapter for the current three-frame runtime."""
from collections import defaultdict
import json
from pathlib import Path
from ttnet.common import jsonl
from ttnet.data_tools.frame_store import FrameStore


class BallCorpus:
    def __init__(self, root, split='val'):
        self.root=Path(root);self.frame_store=FrameStore(root)
        self.metadata=json.loads((self.root/'metadata/dataset.json').read_text())
        self.identity=self.metadata['dataset_id'];self.rows=defaultdict(list);self.videos={}
        self.split={split:[r['video_id'] for r in jsonl(self.root/f'splits/{split}.jsonl')]}
        for r in jsonl(self.root/f'labels/{split}.jsonl'):self.rows[r['video_id']].append(r)
        for vid,rows in self.rows.items():
            rows.sort(key=lambda r:r['frame_id'])
            if [r['frame_id'] for r in rows]!=list(range(len(rows))):raise ValueError('full video frames required')
            self.videos[vid]=dict(meta=dict(n_frames=len(rows),fps=rows[0]['fps'],**rows[0]['resolution']),
                frames={r['frame_id']:dict(timestamp_sec=r['timestamp_sec'],label_status=r['label_status'],
                    xy=r['ball_centers'][0] if r['ball_centers'] else None) for r in rows})

    def image(self,vid,fid):return self.frame_store.row_rgb(self.rows[vid][fid])

    def window_ids(self,vid,center):
        return [i if 0<=i<len(self.rows[vid]) else None for i in range(center-6,center+7)]
