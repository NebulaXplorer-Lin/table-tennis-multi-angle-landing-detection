import unittest

from ttnet.training.stopping import JointEarlyStopping, validate_resume_config


POLICY = dict(mode='loss_and_f1', patience=3, loss_relative_min_delta=.01, f1_min_delta=.002)


class StoppingTests(unittest.TestCase):
    def test_requires_both_metrics_to_plateau(self):
        monitor = JointEarlyStopping(POLICY)
        for loss in [1, .9, .8, .7, .6]:
            self.assertFalse(monitor.update(loss, .8))
        self.assertFalse(monitor.update(.6, .8))
        self.assertFalse(monitor.update(.6, .8))
        self.assertTrue(monitor.update(.6, .8))

    def test_f1_gain_prevents_stop_with_flat_loss(self):
        monitor = JointEarlyStopping(POLICY)
        for f1 in [.7, .71, .72, .73, .74]:
            self.assertFalse(monitor.update(.01, f1))
        self.assertFalse(monitor.update(.01, .74))
        self.assertFalse(monitor.update(.01, .74))
        self.assertTrue(monitor.update(.01, .74))

    def test_small_fluctuations_do_not_reset_patience(self):
        monitor = JointEarlyStopping(POLICY)
        self.assertFalse(monitor.update(1, .8))
        self.assertFalse(monitor.update(.999, .8001))
        self.assertFalse(monitor.update(1.001, .8002))
        self.assertTrue(monitor.update(.999, .8003))

    def test_resume_matches_uninterrupted_monitor(self):
        monitor = JointEarlyStopping(POLICY)
        records = [dict(train_loss=loss, val=dict(f1=f1)) for loss, f1 in [(1, .8), (1, .8), (1, .8)]]
        for record in records:
            monitor.update(record['train_loss'], record['val']['f1'])
        restored = JointEarlyStopping(POLICY)
        restored.load_state_dict(monitor.state_dict())
        replayed = JointEarlyStopping(POLICY)
        replayed.replay(records, 'global')
        self.assertEqual(monitor.state_dict(), replayed.state_dict())
        self.assertTrue(monitor.update(1, .8))
        self.assertTrue(restored.update(1, .8))
        self.assertTrue(replayed.update(1, .8))

    def test_local_crop_switch_resets_comparisons(self):
        monitor = JointEarlyStopping(POLICY)
        records = [dict(train_loss=.01, val=dict(f1=.9), crop_mode='gt_jitter') for _ in range(4)]
        records.append(dict(train_loss=.2, val=dict(f1=.7), crop_mode='predicted_global'))
        monitor.replay(records, 'local')
        self.assertEqual(monitor.loss_wait, 0)
        self.assertEqual(monitor.f1_wait, 0)
        self.assertEqual(monitor.best_loss, .2)

    def test_resume_override_only_allows_stopping_controls(self):
        stored = dict(epochs=80, patience=12, lr=.0003)
        current = dict(epochs=20, patience=5, lr=.0003, early_stopping=POLICY)
        with self.assertRaises(RuntimeError):
            validate_resume_config(stored, current)
        self.assertEqual(set(validate_resume_config(stored, current, True)), {'epochs', 'patience', 'early_stopping'})
        current['lr'] = .001
        with self.assertRaises(RuntimeError):
            validate_resume_config(stored, current, True)
