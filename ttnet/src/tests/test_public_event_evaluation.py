"""Public validation acceptance: matching, complete coverage and decision parity."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('evaluate_public_event', PROJECT / 'scripts/evaluate_public_event.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def frames(n=5):
    return [dict(frame_id=i, source_frame_id=i, timestamp_sec=i / 25,
                 probabilities=[.01, .01], prediction=dict(valid=True, x=130., y=145., confidence=.8)) for i in range(n)]


def records(rows):
    return {r['frame_id']: dict(frame_id=r['frame_id'], prediction=r['prediction'], local_valid=True) for r in rows}


def tiny_video(path, n=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25., (320, 180))
    if not writer.isOpened():
        raise RuntimeError('test video codec unavailable')
    for i in range(n):
        image = np.zeros((180, 320, 3), np.uint8)
        cv2.circle(image, (130 + i, 145), 4, (255, 255, 255), -1)
        writer.write(image)
    writer.release()


class PublicEventEvaluationTests(unittest.TestCase):
    def test_thresholds_default_to_checkpoint_and_overrides_are_explicit(self):
        self.assertEqual(evaluation.thresholds_from(dict(thresholds=[.31, .62])), [.31, .62])
        self.assertEqual(evaluation.thresholds_from({}, [.2, .7]), [.2, .7])
        for bad in (None, [.5], [.5, float('nan')], [-.1, .5]):
            with self.assertRaises(ValueError):
                evaluation.thresholds_from(dict(thresholds=bad))

    def test_valid_match_precedes_ignored_and_missing_coordinates_remain_errors(self):
        rows = frames(10)
        rows[1]['probabilities'][0] = .9
        rows[5]['probabilities'][0] = .9
        rows[9]['probabilities'][0] = .9
        rows[9]['prediction'] = None
        truth = [dict(frame_id=1, x=130., y=145., valid=False),
                 dict(frame_id=1, x=130., y=145., valid=True),
                 dict(frame_id=5, x=130., y=145., valid=False),
                 dict(frame_id=9, x=130., y=145., valid=True)]
        metric, predictions, missing, _, errors = evaluation.score_video(rows, truth, [.5, .5])
        self.assertEqual([metric[k] for k in ('tp', 'fp', 'fn', 'ignored')], [1, 0, 1, 1])
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(missing), 1)
        self.assertCountEqual([e['kind'] for e in errors], ['FN', 'event_without_coordinate'])

    def test_macro_is_over_angles_and_totals_are_summed(self):
        manifest = dict(videos=[dict(video_id='a', angle='0'), dict(video_id='b', angle='0'), dict(video_id='c', angle='90')])
        per_video = dict(a=dict(tp=9, fp=0, fn=0, ignored=0, events_without_coordinates=0),
                         b=dict(tp=1, fp=0, fn=0, ignored=0, events_without_coordinates=0),
                         c=dict(tp=0, fp=0, fn=1, ignored=0, events_without_coordinates=2))
        result = evaluation.summarize(per_video, manifest)
        self.assertEqual(result['macro_f1'], .5)
        self.assertEqual(result['overall']['tp'], 10)
        self.assertEqual(result['overall']['fn'], 1)
        self.assertEqual(result['events_without_coordinates'], 2)

    def test_incomplete_frames_source_mapping_and_pts_are_rejected(self):
        rows = frames()
        video = dict(video_id='v', n_frames=5)
        evaluation.validate_rows(rows, records(rows), video, [i / 25 for i in range(5)])
        with self.assertRaises(ValueError):
            evaluation.validate_rows(rows[:-1], records(rows), video)
        drift = deepcopy(rows)
        drift[2]['source_frame_id'] = 1
        with self.assertRaises(ValueError):
            evaluation.validate_rows(drift, records(drift), video)
        drift = deepcopy(rows)
        drift[3]['timestamp_sec'] += .001
        with self.assertRaises(ValueError):
            evaluation.validate_rows(drift, records(drift), video, [i / 25 for i in range(5)])

    def test_sub_tolerance_probability_flip_fails_decision_parity(self):
        cached = frames()
        online = deepcopy(cached)
        cached[2]['probabilities'][0] = .5 - 1e-7
        online[2]['probabilities'][0] = .5 + 1e-7
        result, differences = evaluation.compare_frames(cached, online, [.5, .5])
        self.assertTrue(result['within_tolerance'])
        self.assertFalse(result['decisions_equal'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['counts']['threshold_decision_differences'], 1)
        self.assertEqual(differences[0]['frame_id'], 2)

    def test_same_event_count_does_not_hide_peak_frame_shift(self):
        cached = frames()
        online = deepcopy(cached)
        cached[1]['probabilities'][0], cached[2]['probabilities'][0] = .7, .70001
        online[1]['probabilities'][0], online[2]['probabilities'][0] = .70001, .7
        result, _ = evaluation.compare_frames(cached, online, [.5, .5])
        self.assertEqual(len(result['cache_peak_decisions']), len(result['online_peak_decisions']))
        self.assertFalse(result['peak_decisions_equal'])
        self.assertFalse(result['passed'])

    def test_coordinate_validity_and_out_of_tolerance_drift_fail_parity(self):
        cached = frames()
        online = deepcopy(cached)
        online[1]['prediction']['x'] += .2
        online[2]['prediction'] = None
        result, _ = evaluation.compare_frames(cached, online, [.5, .5])
        self.assertFalse(result['passed'])
        self.assertEqual(result['counts']['coordinate_validity_differences'], 1)
        self.assertEqual(result['counts']['coordinates_outside_atol'], 1)
        exact, difference = evaluation.compare_frames(cached, cached, [.5, .5])
        self.assertTrue(exact['exact'])
        self.assertEqual(difference, [])

    def test_full_source_decode_and_all_error_overlay_artifacts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'source.mp4'
            tiny_video(source)
            video = dict(video_id='v', n_frames=5, width=320, height=180, fps=25)
            scan = evaluation.scan_video(source, video)
            self.assertEqual(scan['frames'], 5)
            with self.assertRaises(ValueError):
                evaluation.scan_video(source, dict(video, n_frames=6))
            rows = frames()
            rows[1]['probabilities'][0] = .9
            truth = [dict(frame_id=4, x=130., y=145., valid=True)]
            match, predictions, _, _, errors = evaluation.score_video(rows, truth, [.5, .5])
            info = evaluation.overlays(source, video, rows, records(rows), truth, predictions, match, errors, root, [.5, .5])
            self.assertEqual(info['decoded_verification_frames'], 5)
            self.assertCountEqual([e['kind'] for e in errors], ['FP', 'FN'])
            for error in errors:
                self.assertTrue((root / error['visualization']).is_file())
                self.assertTrue((root / error['overlay_video']).is_file())

    def test_cache_requires_complete_loaded_system_identity_before_loading(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            expected = dict(dataset_id='dataset', global_sha256='correct', extraction_batch=16)
            args = SimpleNamespace(cache=root, corpus=None, pp_annotations=None,
                                   cache_mapping=None, public=root)
            fake_corpus = SimpleNamespace(identity='dataset', public_id_mapping={'v':'public_v'},
                 split=dict(train=['train_v'], val=['public_v'], test=[]))
            system = SimpleNamespace(config=dict(dataset=str(root), pp_annotations=str(root / 'pp.json')))
            (root / 'identity.json').write_text(json.dumps(dict(expected, global_sha256='wrong')))
            with patch.object(evaluation, 'EventCorpus', return_value=fake_corpus), \
                 patch('ttnet.events.continuous.attach_pp'), patch('ttnet.events.public_validation.attach_public_validation'), \
                 patch.object(evaluation, 'cache_identity', return_value=expected) as recompute, \
                 patch.object(evaluation, 'load_cache', return_value={'public_v':({}, {})}) as loader:
                with self.assertRaisesRegex(ValueError, 'global_sha256'):
                    evaluation.cache_inputs(args, system)
                recompute.assert_called_once_with(system, fake_corpus)
                loader.assert_not_called()
                (root / 'identity.json').write_text(json.dumps(expected))
                result = evaluation.cache_inputs(args, system)
                self.assertEqual(result[2], {'v':'public_v'})
                loader.assert_called_once_with(fake_corpus, root, parts=('train', 'val', 'test'), expected=expected)

    def test_evaluation_writes_evaluated_receipt_without_gt_in_inference(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'source.mp4'
            tiny_video(source)
            manifest = dict(videos=[dict(video_id='v', path='source.mp4', n_frames=5, width=320, height=180, fps=25, angle='0')])
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'gt_public.jsonl').write_text(json.dumps(dict(video_id='v', frame_id=1, x=130., y=145., valid=True)) + '\n')
            checkpoint = root / 'best.pt'
            torch.save(dict(config=dict(mode='frozen', amp=False), system={}, epoch=3, thresholds=[.5, .5]), checkpoint)
            rows = frames()
            rows[1]['probabilities'][0] = .9
            fake_system = SimpleNamespace(device=torch.device('cpu'), load_state_dict=lambda _: None, configure=lambda *_: None)
            def predict(system, source_path, video_id, **kwargs):
                self.assertIs(system, fake_system)
                self.assertEqual(source_path, source)
                self.assertEqual(video_id, 'v')
                self.assertEqual(kwargs, dict(target_fps=None))
                return deepcopy(rows), records(rows), dict(frames=5, elapsed_sec=.1)
            args = SimpleNamespace(output=root / 'eval', checkpoint=checkpoint, public=root, thresholds=None,
                 nms_sec=.08, device='cpu', cache=None, probability_atol=2e-4, coordinate_atol=.1)
            expected = dict(videos=1, frames=5, valid_gt=1, ignored_gt=0)
            with patch.object(evaluation, 'PUBLIC_EXPECTED', expected), patch.object(evaluation, 'EventSystem', return_value=fake_system), \
                 patch.object(evaluation, 'predict_video', side_effect=predict), patch.object(evaluation, 'code_identity', return_value={}):
                result = evaluation.evaluate(args)
            self.assertEqual(result['status'], 'evaluated')
            self.assertEqual(result['overall']['tp'], 1)
            self.assertFalse(result['ground_truth_input_to_inference'])
            state = json.loads((args.output / 'state.json').read_text())
            self.assertEqual(state['status'], 'evaluated')
            self.assertTrue((args.output / 'trajectory/v.jsonl').is_file())
            self.assertTrue((args.output / 'overlays/v.mp4').is_file())
            with self.assertRaises(FileExistsError):
                evaluation.evaluate(args)


if __name__ == '__main__':
    unittest.main()
