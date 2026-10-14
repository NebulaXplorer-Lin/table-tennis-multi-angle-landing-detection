"""Input sizes are run settings, never immutable source-dataset metadata."""
from __future__ import annotations

import math
import torch
from torch.utils.data import Sampler

DEFAULT_SIZES = {"global": (512, 288), "local": (256, 256)}


def input_size(config, stage):
    return validate_size((config or {}).get(stage + "_input", DEFAULT_SIZES[stage]), stage)


def validate_size(size, stage):
    if len(size) != 2 or any(not isinstance(v, int) or v < 32 or v % 4 for v in size):
        raise ValueError("input size is [width,height], dimensions >=32 and divisible by 4")
    if stage == "local" and size[0] != size[1]:
        raise ValueError("Local inputs must be square")
    return tuple(size)


def training_scales(config, stage):
    choices = config.get(stage + "_train_scales", [input_size(config, stage)])
    if not choices:
        raise ValueError("empty training scale list")
    return [validate_size(s, stage) for s in choices]


class MultiScaleBatchSampler(Sampler):
    """Every micro-batch uses one size, so normal DataLoader collation remains valid."""
    def __init__(self, length, batch_size, scales):
        self.length, self.batch_size, self.scales = length, batch_size, tuple(scales)

    def __len__(self):
        return math.ceil(self.length / self.batch_size)

    def __iter__(self):
        indices = torch.randperm(self.length).tolist()
        for start in range(0, self.length, self.batch_size):
            size = self.scales[int(torch.randint(len(self.scales), (1,)).item())]
            yield [(i, size) for i in indices[start:start+self.batch_size]]
