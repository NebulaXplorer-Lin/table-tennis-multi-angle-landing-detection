"""Joint training-loss / validation-F1 plateau detection, separate from best selection."""
from __future__ import annotations

import math


class JointEarlyStopping:
    def __init__(self, policy):
        self.policy = dict(policy)
        self.patience = int(policy['patience'])
        self.loss_delta = float(policy['loss_relative_min_delta'])
        self.f1_delta = float(policy['f1_min_delta'])
        if self.patience < 1 or not 0 <= self.loss_delta < 1 or self.f1_delta < 0:
            raise ValueError('invalid joint early-stopping policy')
        self.reset()

    def reset(self):
        self.best_loss = None
        self.best_f1 = None
        self.loss_wait = 0
        self.f1_wait = 0

    def update(self, loss, f1):
        if not math.isfinite(loss) or not math.isfinite(f1):
            raise ValueError('early stopping requires finite loss and F1')
        if self.best_loss is None or loss < self.best_loss * (1 - self.loss_delta):
            self.best_loss = loss
            self.loss_wait = 0
        else:
            self.loss_wait += 1
        if self.best_f1 is None or f1 > self.best_f1 + self.f1_delta:
            self.best_f1 = f1
            self.f1_wait = 0
        else:
            self.f1_wait += 1
        return self.loss_wait >= self.patience and self.f1_wait >= self.patience

    def state_dict(self):
        return dict(policy=self.policy, best_loss=self.best_loss, best_f1=self.best_f1,
                    loss_wait=self.loss_wait, f1_wait=self.f1_wait)

    def load_state_dict(self, state):
        if state['policy'] != self.policy:
            raise ValueError('early-stopping policy differs')
        for key in ['best_loss', 'best_f1', 'loss_wait', 'f1_wait']:
            setattr(self, key, state[key])

    def replay(self, records, stage):
        """Rebuild state for old checkpoints or a user-authorized policy change."""
        self.reset()
        previous_crop = None
        for record in records:
            crop = record.get('crop_mode')
            if stage == 'local' and previous_crop == 'gt_jitter' and crop == 'predicted_global':
                self.reset()
            self.update(record['train_loss'], record['val']['f1'])
            previous_crop = crop


def validate_resume_config(stored, current, allow_stopping_change=False):
    observations = {'dataset_id', 'split_id', 'split_hashes', 'train_videos', 'val_videos',
                    'train_frames', 'val_frames', 'data_policy', 'evaluated_best_checkpoint_sha256', 'best_epoch'}
    changed = {key: {'before': stored.get(key), 'after': current.get(key)}
               for key in stored.keys() | current.keys()
               if key not in observations and stored.get(key) != current.get(key)}
    allowed = {'epochs', 'patience', 'early_stopping'} if allow_stopping_change else set()
    if changed.keys() - allowed:
        raise RuntimeError(f'resume configuration differs: {sorted(changed.keys() - allowed)}')
    return changed
