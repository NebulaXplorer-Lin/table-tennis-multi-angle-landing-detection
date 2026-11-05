"""The completed study hash registry has exactly one mutable source exception."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ttnet.common import sha

PROJECT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('package_public_event_best', PROJECT / 'scripts/package_public_event_best.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PublicPackageSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root/'study'
        self.run = self.root/'runs/selected'
        self.evaluation = self.run/'evaluations/public'
        critical = ['config.yaml', 'run.json', 'best_selection.json', 'checkpoints/best.pt',
                    'public_split_audit.json', 'evaluations/public/metrics.json',
                    'evaluations/public/performance.json', 'evaluations/public/identity.json',
                    'evaluations/public/cache_parity.json', 'evaluations/public/overlay_verification.json',
                    'evaluations/public/per_video.json', 'evaluations/public/source_identity.json',
                    'evaluations/benchmark/benchmark.json', 'evaluations/benchmark/benchmark_samples_ms.json']
        self.outside = [self.root/'configs/experiment.json', self.root/'public/gt_public.jsonl',
                        self.root/'cache/features.npz', self.root/'runs/other_seed/checkpoints/best.pt',
                        self.root/'source/video.mp4', self.root/'implementation.py']
        all_sources = [self.run/name for name in critical] + self.outside + [self.task/'task.json']
        for path in all_sources:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('original '+str(path))
        self.registry = {str(path):sha(path) for path in all_sources}
        self.selected = dict(row=dict(checkpoint=str(self.run/'checkpoints/best.pt'),
                            experiment_id=self.run.name, metrics=str(self.evaluation/'metrics.json')),
                             evaluation=str(self.evaluation))
        self.analysis = dict(status='completed', validation_set='pp_public', task=str(self.task),
                  checks=[dict(passed=True)], runs={'selected':self.selected}, sources_sha256=self.registry)
        (self.task/'analysis.json').write_text(json.dumps(self.analysis))
        for name in ('comparison.csv', 'summary.md', 'training_curves.png'):
            (self.task/name).write_text('fixed report '+name)
        self.completion = dict(schema='public_event_completion_v1', status='completed',
              validation_set='pp_public', all_checks_passed=True, task=str(self.task),
              machine_audit_checks=1, runs=1, sources_sha256=self.registry,
              reports_sha256={name:sha(self.task/name) for name in ('analysis.json', 'comparison.csv', 'summary.md', 'training_curves.png')})
        (self.task/'completion.json').write_text(json.dumps(self.completion))

    def verify(self):
        return package.verify_sources(self.task, self.analysis, self.completion, self.selected)

    def test_every_registry_source_except_exact_task_snapshot_is_verified(self):
        _, _, verified = self.verify()
        self.assertNotIn(str(self.task/'task.json'), verified)
        for source in self.registry:
            if source != str(self.task/'task.json'):
                self.assertEqual(verified[source], self.registry[source])
        (self.task/'task.json').write_text('appended orchestration completion links')
        self.verify()

    def test_mutated_config_gt_cache_other_seed_video_or_code_is_rejected(self):
        for path in self.outside:
            with self.subTest(source=str(path)):
                original = path.read_bytes()
                path.write_text('changed after completed study audit')
                with self.assertRaisesRegex(ValueError, 'changed audited source'):
                    self.verify()
                path.write_bytes(original)

    def test_non_task_missing_source_is_rejected(self):
        self.outside[0].unlink()
        with self.assertRaisesRegex(ValueError, 'changed audited source'):
            self.verify()

    def test_another_task_json_has_no_mutable_exception(self):
        path = self.root/'another/task.json'
        path.parent.mkdir()
        path.write_text('immutable other task snapshot')
        registry = dict(self.registry, **{str(path):sha(path)})
        analysis, completion = deepcopy(self.analysis), deepcopy(self.completion)
        analysis['sources_sha256'] = completion['sources_sha256'] = registry
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed audited source'):
            package.verify_sources(self.task, analysis, completion, self.selected)


if __name__ == '__main__':
    unittest.main()
