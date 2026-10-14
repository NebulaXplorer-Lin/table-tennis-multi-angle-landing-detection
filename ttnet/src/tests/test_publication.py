"""Verify result snapshots survive archive changes and report incomplete evidence."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from ttnet.common import save, sha
from ttnet.publication import publish_run, read, write_index


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'runs/20261014-123000_global_ball'
        self.run.mkdir(parents=True)
        save(self.run/'config.yaml', dict(stage='global', smoke=False, dataset_id='data', split_id='fixed'))
        save(self.run/'run.json', dict(status='completed', best_epoch=3, epochs_completed=5))
        for name in ['checkpoints/best.pt', 'checkpoints/last.pt', 'exports/global.onnx',
                     'predictions/best_val.jsonl', 'visualizations/best_val/video/0.jpg']:
            path = self.run/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        checkpoint_sha = sha(self.run/'checkpoints/best.pt')
        save(self.run/'evaluations/best_val/metrics.json', dict(center={'10': dict(precision=.8, recall=.7, f1=.75, tp=7, fp=2, fn=3)},
             global_checkpoint_sha256=checkpoint_sha, local_checkpoint_sha256=None))
        save(self.run/'evaluations/best_val/per_video.json', {})
        save(self.run/'evaluations/best_val/errors.json', [dict(video_id='video', frame_id=0, kind='miss', visualization='visualizations/best_val/video/0.jpg')])
        (self.run/'evaluations/best_val/index.html').write_text('<img src="../../visualizations/best_val/video/0.jpg">')
        save(self.run/'evaluations/benchmark/metrics.json', dict(e2e_fps=100, global_sha256=checkpoint_sha, local_sha256=None))
        save(self.run/'exports/manifest.json', dict(models=[dict(stage='global', checkpoint_sha256=checkpoint_sha,
             onnx_sha256=sha(self.run/'exports/global.onnx'), parity=dict(passed=True))]))
        save(self.run/'feature_cache/best_val/identity.json', {})
        path = self.run/'logs/attempt_001/epochs.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text('{"epoch":1,"train_loss":0.1}\n')

    def test_snapshot_is_self_contained_and_keeps_hashes(self):
        dest = publish_run(self.run, self.root)
        self.assertEqual(read(dest/'manifest.json')['status'], 'completed')
        self.assertIn('../../error_samples/', (dest/'eval/best_val/index.html').read_text())
        self.assertEqual(read(dest/'stats/summary.json')['center']['10']['recall'], .7)
        self.assertTrue((dest/'eval/best_val/errors.html').is_file())
        self.assertEqual(sha(dest/'weights/best.pt'), sha(self.run/'checkpoints/best.pt'))
        shutil.rmtree(self.run)
        errors = read(dest/'eval/best_val/errors.json')
        self.assertTrue((dest/errors[0]['visualization']).is_file())
        self.assertTrue((dest/'weights/best.pt').is_file())
        index = write_index(self.root)
        self.assertIn('70.00%', index.read_text())

    def test_missing_evidence_and_mismatched_benchmark_are_partial(self):
        (self.run/'visualizations/best_val/video/0.jpg').unlink()
        save(self.run/'evaluations/benchmark/metrics.json', dict(e2e_fps=100, global_sha256='wrong'))
        dest = publish_run(self.run, self.root)
        manifest = read(dest/'manifest.json')
        self.assertEqual(manifest['status'], 'partial')
        self.assertTrue(any('测速 checkpoint' in item for item in manifest['missing']))
        self.assertTrue(any('错误图片' in item for item in manifest['missing']))

    def test_refresh_is_idempotent_and_archives_changed_results(self):
        dest = publish_run(self.run, self.root)
        old = read(dest/'manifest.json')
        publish_run(self.run, self.root)
        self.assertEqual(read(dest/'manifest.json'), old)
        self.assertFalse((self.root/'output/.history').exists())
        save(self.run/'evaluations/best_val/extra_statistics.json', dict(measured=True))
        publish_run(self.run, self.root)
        self.assertEqual(len(list((self.root/'output/.history').iterdir())), 1)
        self.assertTrue((dest/'eval/best_val/extra_statistics.json').is_file())

    def test_local_bundle_includes_global_dependency(self):
        global_path = self.root/'runs/global/checkpoints/best.pt'
        global_path.parent.mkdir(parents=True)
        global_path.write_bytes(b'global checkpoint')
        save(self.run/'config.yaml', dict(stage='local', global_checkpoint=str(global_path)))
        metric = read(self.run/'evaluations/best_val/metrics.json')
        metric.update(local_checkpoint_sha256=metric['global_checkpoint_sha256'], global_checkpoint_sha256=sha(global_path))
        save(self.run/'evaluations/best_val/metrics.json', metric)
        dest = publish_run(self.run, self.root)
        self.assertEqual(sha(dest/'weights/global_best.pt'), sha(global_path))
        global_path.write_bytes(b'changed global checkpoint')
        publish_run(self.run, self.root)
        self.assertTrue(any('Global 权重' in item for item in read(dest/'manifest.json')['missing']))

    def test_running_and_smoke_runs_do_not_publish(self):
        save(self.run/'run.json', dict(status='training'))
        self.assertIsNone(publish_run(self.run, self.root))
        save(self.run/'run.json', dict(status='completed'))
        save(self.run/'config.yaml', dict(stage='global', smoke=True))
        self.assertIsNone(publish_run(self.run, self.root))


if __name__=='__main__':
    unittest.main()
