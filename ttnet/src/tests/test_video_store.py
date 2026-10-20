import json
import pickle
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from ttnet.common import save, save_jsonl, sha
from ttnet.data_tools.frame_store import FrameStore
from ttnet.data_tools.storage_check import verify_store
from ttnet.data_tools.video_store import prepare


class VideoStoreTests(unittest.TestCase):
    def fixture(self,root,kind):
        video=root/'data/ttnet/training/videos/v.mp4'
        video.parent.mkdir(parents=True)
        # B-frames, multiple GOPs and a time gap exercise seek/decode alignment.
        with av.open(str(video),'w') as container:
            stream=container.add_stream('libx264',rate=25)
            stream.width=64;stream.height=48;stream.pix_fmt='yuv420p'
            stream.options={'crf':'18','g':'4','bf':'2'}
            for i in range(10):
                image=np.random.default_rng(i).integers(0,256,(48,64,3),np.uint8)
                frame=av.VideoFrame.from_ndarray(image,format='rgb24')
                frame.pts=i+(i>=4);frame.time_base=Fraction(1,25)
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        expected={};times=[]
        with av.open(str(video)) as container:
            first=None
            for i,frame in enumerate(container.decode(video=0)):
                if first is None:first=float(frame.pts*frame.time_base)
                times.append(float(frame.pts*frame.time_base)-first)
                bgr=frame.to_ndarray(format='bgr24')
                # Compare against files written by the original preparation path.
                extension='png' if kind=='ball' else 'jpg'
                image_path=root/f'gold-{i}.{extension}'
                parameters=[] if kind=='ball' else [cv2.IMWRITE_JPEG_QUALITY,95]
                self.assertTrue(cv2.imwrite(str(image_path),bgr,parameters))
                expected[i]=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
        dataset=root/'data/datasets'/kind
        if kind=='ball':
            rows=[dict(video_id='v',frame_id=i,timestamp_sec=times[i],fps=25,
                       frame_path=f'frames/v/{i:06d}.png',resolution=dict(width=64,height=48),
                       label_status='positive',ball_centers=[[32.,24.]]) for i in range(10)]
            save_jsonl(dataset/'labels/train.jsonl',rows);save_jsonl(dataset/'labels/val.jsonl',[])
            save(dataset/'metadata/dataset.json',dict(status='completed',dataset_id='fixture-ball',
                 source_videos=[dict(video_id='v',video='data/ttnet/training/videos/v.mp4',video_sha256=sha(video))]))
            save(dataset/'metadata/videos/v.json',dict(resolution=dict(width=64,height=48)))
            save_jsonl(dataset/'splits/train.jsonl',[dict(video_id='v')]);save_jsonl(dataset/'splits/val.jsonl',[])
        else:
            rows=[dict(frame_id=i,source_frame_id=i,timestamp_sec=times[i],path=f'frames/{i:06d}.jpg',
                       label_status='unknown',xy=None) for i in (0,2,5,9)]
            save(dataset/'dataset.json',dict(dataset_id='fixture-event',complete=True,pilot=False))
            save(dataset/'split.json',dict(split_id='fixture',train=['v'],val=[],test=[]))
            save(dataset/'v/metadata.json',dict(source='/old/project/data/ttnet/training/videos/v.mp4',
                 source_identity=dict(video_sha256=sha(video)),width=64,height=48,n_frames=10,fps=25))
            save_jsonl(dataset/'v/frames.jsonl',rows)
            save_jsonl(dataset/'v/samples.jsonl',[]);save_jsonl(dataset/'v/events.jsonl',[])
        result=prepare(dataset,publish=True,project=root)
        return dataset,result,expected,rows

    def test_original_pixels_random_seeks_and_pickle(self):
        for kind in ('ball','event'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                dataset,result,expected,rows=self.fixture(Path(temp),kind)
                store=FrameStore(dataset)
                for fid in [r['frame_id'] for r in rows][::-1]+[r['frame_id'] for r in rows]:
                    image=store.get_rgb('v',fid)
                    np.testing.assert_array_equal(image,expected[fid])
                    self.assertFalse(image.flags.writeable)
                self.assertLessEqual(len(store._readers['v'].frames),8)
                with self.assertRaises(KeyError):store.get_rgb('v',99)
                restored=pickle.loads(pickle.dumps(store))
                np.testing.assert_array_equal(restored.get_rgb('v',0),expected[0])
                self.assertIn('pyav',store.backend_identity())
                store.close();restored.close()
                audit=verify_store(dataset,full=True)
                self.assertTrue(audit['passed']);self.assertEqual(audit['frames'],len(rows))

    def test_existing_labels_unchanged_and_prepare_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset,result,_,_=self.fixture(root,'ball')
            path=dataset/'labels/train.jsonl';before=path.read_bytes()
            again=prepare(dataset,publish=True,project=root)
            self.assertEqual(result['reference'],again['reference'])
            self.assertEqual(path.read_bytes(),before)
            from ttnet.training.dataset import BallDataset
            batch=BallDataset(dataset,'train',stage='global',temporal_mode='fixed3',input_size=(128,96))
            data,heat,valid,_=batch[4]
            self.assertEqual(tuple(data['images'].shape),(3,3,96,128))
            self.assertTrue(valid.item());self.assertGreater(float(heat.max()),0)

    def test_event_corpus_source_relocation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset,_,expected,_=self.fixture(root,'event')
            from ttnet.events.data import EventCorpus
            corpus=EventCorpus(dataset)
            self.assertTrue(corpus.videos['v']['meta']['source'].startswith(str(root)))
            np.testing.assert_array_equal(corpus.image('v',5),expected[5])
            with self.assertRaises(KeyError):corpus.image('v',1)
            from ttnet.events.context import ensure_continuous_context
            with self.assertRaisesRegex(ValueError,'forbids dense RGB'):
                ensure_continuous_context(corpus,'v')

    def test_source_identity_and_missing_pts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset,result,_,_=self.fixture(Path(temp),'event')
            manifest_path=Path(result['store'])/'v/manifest.json'
            manifest=json.loads(manifest_path.read_text());manifest['source_bytes']+=1
            save(manifest_path,manifest)
            with self.assertRaises(ValueError):FrameStore(dataset).get_rgb('v',0)
            manifest['source_bytes']-=1;save(manifest_path,manifest)
            timestamp_path=Path(result['store'])/'v/timestamps.npy'
            times=np.load(timestamp_path);times[1]=.14  # No such video PTS.
            np.save(timestamp_path,times)
            with self.assertRaises(ValueError):FrameStore(dataset).get_rgb('v',2)


if __name__=='__main__':unittest.main()
