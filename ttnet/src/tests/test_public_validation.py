"""Semantic checks for the user-authorized public-validation partition."""
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from ttnet.common import save, save_jsonl, sha
from ttnet.events.public_validation import (
    attach_public_validation, score_public_rows, validate_public_cached,
)


class PublicValidationTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'videos/public.mp4'
        source.parent.mkdir(parents=True)
        with av.open(str(source), 'w') as container:
            stream = container.add_stream('libx264', rate=25)
            stream.width = 64; stream.height = 48; stream.pix_fmt = 'yuv420p'
            stream.options = dict(crf='18', g='4', bf='2')
            for i in range(10):
                rgb = np.random.default_rng(i).integers(0, 256, (48,64,3), np.uint8)
                frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                frame.pts = i + (i>=4); frame.time_base = Fraction(1,25)
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
        with av.open(str(source)) as container:
            actual_rate = float(container.streams.video[0].average_rate)
        save(root/'manifest.json', dict(name='fixture', videos=[dict(video_id='v',
            path='videos/public.mp4', angle='0度', width=64, height=48, n_frames=10,
            fps=actual_rate, n_valid_landings=1)]))
        save_jsonl(root/'gt_public.jsonl', [dict(video_id='v', frame_id=2, x=10., y=12., valid=True),
                                           dict(video_id='v', frame_id=5, x=20., y=22., valid=False)])
        def image(vid, fid): return np.zeros((48,64,3), np.uint8)
        image.cache_clear = lambda:None
        train = dict(video_id='pp_a', frame_id=1, target=[1.,0.], supervised=[True,False])
        old_val = dict(video_id='pp_b', frame_id=2, target=[1.,0.], supervised=[True,False])
        sparse = dict(video_id='game_5', frame_id=3, target=[0.,1.], supervised=[True,True])
        corpus = SimpleNamespace(identity='original', metadata=dict(complete=True, pilot=False),
            split=dict(split_id='old', train=['game_1','pp_a'], val=['game_5','pp_b'], test=['test_1']),
            videos={vid:dict(meta={},frames={}) for vid in ['game_1','game_5','test_1','pp_a','pp_b']},
            pp_annotations={'pp_a':dict(split='train',source_sha256='pp-a'),
                            'pp_b':dict(split='val',source_sha256='pp-b')},
            samples=dict(train=[train], val=[sparse,old_val], test=[]), image=image)
        return corpus, source

    def test_old_val_is_train_and_public_is_full_unknown_evaluation_only(self):
        with tempfile.TemporaryDirectory() as temp:
            corpus, source = self.fixture(Path(temp))
            audit = attach_public_validation(corpus, temp)
            self.assertEqual(corpus.split['train'], ['game_1','pp_a','pp_b'])
            self.assertEqual(corpus.split['val'], ['game_5','public_v'])
            self.assertEqual([s['video_id'] for s in corpus.samples['train']], ['pp_a','pp_b'])
            self.assertEqual([s['video_id'] for s in corpus.samples['val']], ['game_5'])
            self.assertEqual(corpus.public_id_mapping, {'v':'public_v'})
            video = corpus.videos['public_v']
            self.assertEqual(sorted(video['frames']), list(range(10)))
            self.assertTrue(all(r['label_status']=='unknown' and r['xy'] is None for r in video['frames'].values()))
            self.assertEqual(video['events'], [])
            self.assertEqual(corpus.pp_annotations['pp_b']['split'], 'train')
            self.assertAlmostEqual(video['frames'][4]['timestamp_sec'], .2)
            self.assertEqual(audit['public_training_samples'], 0)
            self.assertEqual(audit['public_sources'][0]['source_sha256'], sha(source))
            self.assertEqual(json.loads(json.dumps(audit)), audit)
            with self.assertRaises(ValueError): attach_public_validation(corpus, temp)
            corpus.image.cache_clear()

    def test_bounded_reader_matches_direct_decode_during_backward_batch_access(self):
        with tempfile.TemporaryDirectory() as temp:
            corpus, source = self.fixture(Path(temp))
            attach_public_validation(corpus, temp)
            with av.open(str(source)) as container:
                expected = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
            for fid in [0,1,2,3,4,5,6,7,8,9,2,1,0,9,4,5]:
                actual = corpus.image('public_v',fid)
                np.testing.assert_array_equal(actual,expected[fid])
                self.assertFalse(actual.flags.writeable)
                self.assertLessEqual(len(corpus.public_frame_store.reader.frames),3)
            corpus.image.cache_clear()

    def test_source_duplicate_rejected_before_changing_partition(self):
        with tempfile.TemporaryDirectory() as temp:
            corpus, source = self.fixture(Path(temp))
            corpus.pp_annotations['pp_b']['source_sha256'] = sha(source)
            original = list(corpus.split['val'])
            with self.assertRaisesRegex(ValueError,'duplicates'):
                attach_public_validation(corpus,temp)
            self.assertEqual(corpus.split['val'],original)
            self.assertEqual(corpus.identity,'original')

    def test_gt_identity_is_bound_and_coordinates_never_enter_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, _ = self.fixture(root)
            first = attach_public_validation(corpus,root)['dataset_id']
            corpus.image.cache_clear()
            # Rebuild just the original corpus, preserving identical source bytes.
            corpus2 = SimpleNamespace(identity='original', metadata=dict(complete=True,pilot=False),
                split=dict(split_id='old',train=['game_1','pp_a'],val=['game_5','pp_b'],test=['test_1']),
                videos={vid:dict(meta={},frames={}) for vid in ['game_1','game_5','test_1','pp_a','pp_b']},
                pp_annotations={'pp_a':dict(split='train',source_sha256='pp-a'),
                                'pp_b':dict(split='val',source_sha256='pp-b')},
                samples=dict(train=[dict(video_id='pp_a',frame_id=1,target=[1.,0.],supervised=[True,False])],
                    val=[dict(video_id='game_5',frame_id=3,target=[0.,1.],supervised=[True,True]),
                         dict(video_id='pp_b',frame_id=2,target=[1.,0.],supervised=[True,False])],test=[]),
                image=corpus.image)
            gt = [json.loads(line) for line in (root/'gt_public.jsonl').read_text().splitlines()]
            gt[0]['x'] += 1
            save_jsonl(root/'gt_public.jsonl',gt)
            second = attach_public_validation(corpus2,root)['dataset_id']
            self.assertNotEqual(first,second)
            self.assertFalse(any('x' in s or 'xy' in s for part in corpus2.samples.values() for s in part))
            corpus2.image.cache_clear()

    def scoring_fixture(self):
        # A valid event sharing an ignored GT must still match the valid event;
        # a prediction far from an ignored GT at its exact time remains an FP.
        truth = [dict(frame_id=1,x=0.,y=0.,valid=True),
                 dict(frame_id=1,x=0.,y=0.,valid=False),
                 dict(frame_id=4,x=0.,y=0.,valid=False),
                 dict(frame_id=7,x=0.,y=0.,valid=False)]
        corpus = SimpleNamespace(public_video_ids={'public_v':'v'},
            public_ground_truth={'public_v':truth},
            videos={'public_v':dict(meta=dict(n_frames=10,angle='0度'))},
            split=dict(train=['pp_a'],val=['public_v']),samples=dict(train=[],val=[],test=[]))
        rows = [dict(frame_id=i,timestamp_sec=i*.04,probabilities=[.9 if i in (1,4,7) else 0.,0.],
            prediction=dict(x=100. if i==4 else 0.,y=0.,valid=True)) for i in range(10)]
        return corpus, rows

    def test_ignored_gt_requires_space_and_does_not_steal_valid_match(self):
        corpus, rows = self.scoring_fixture()
        result = score_public_rows(corpus,{'public_v':rows},.5)
        self.assertEqual({k:result['landing'][k] for k in ('tp','fp','fn')},dict(tp=1,fp=1,fn=0))
        self.assertEqual(result['per_video']['public_v']['landing']['ignored'],1)
        self.assertAlmostEqual(result['macro_f1'],2/3)
        with self.assertRaisesRegex(ValueError,'every original frame'):
            score_public_rows(corpus,{'public_v':rows[:-1]},.5)

    def test_angle_macro_is_not_pooled_f1_and_threshold_ties_are_explicit(self):
        corpus, rows = self.scoring_fixture()
        corpus.public_video_ids['public_w'] = 'w'
        corpus.videos['public_w'] = dict(meta=dict(n_frames=10,angle='90度'))
        corpus.public_ground_truth['public_w'] = [dict(frame_id=1,x=0.,y=0.,valid=True)]
        corpus.split['val'].append('public_w')
        missed = [dict(r,probabilities=[0.,0.]) for r in rows]
        score = score_public_rows(corpus,{'public_v':rows,'public_w':missed},.5)
        self.assertAlmostEqual(score['macro_f1'],1/3)
        self.assertAlmostEqual(score['landing']['f1'],.5)
        system = SimpleNamespace(config=dict(nms_sec=.08))
        with patch('ttnet.events.inference.predict_windows',side_effect=[rows,missed]):
            best,sweep,pred = validate_public_cached(system,corpus,
                {'public_v':(None,None),'public_w':(None,None)},[.3,.5])
        self.assertEqual(best['threshold'],.5)
        self.assertEqual(len(sweep),2)
        self.assertEqual(set(pred),{'public_v','public_w'})
        corpus.samples['train'].append(dict(video_id='public_v'))
        with self.assertRaisesRegex(ValueError,'supervised samples'):
            validate_public_cached(system,corpus,{},[.5])


if __name__=='__main__':
    unittest.main()
