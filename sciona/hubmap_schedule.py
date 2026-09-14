"""Pinned source CosineLR, including its stateful get_lr restart semantics.

Copyright (c) 2021 Tom, MIT; see docs/licenses/HuBMAP-MIT.txt.
Source commit615444e86f4fb5ab916c0b66d311a307f4e4dd27.
"""
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CosineLR(_LRScheduler):
    def __init__(self, optimizer, step_size_min=1e-5, t0=100, tmult=2, curr_epoch=-1, last_epoch=-1):
        if not np.isfinite(step_size_min) or step_size_min < 0:
            raise ValueError('Finite nonnegative minimum learning rate required')
        if isinstance(t0,bool) or not isinstance(t0,int) or t0 < 1 or isinstance(tmult,bool) or not isinstance(tmult,int) or tmult < 1:
            raise ValueError('Positive integer period and multiplier required')
        self.step_size_min = step_size_min
        self.t0 = t0
        self.tmult = tmult
        self.epochs_since_restart = curr_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        self.epochs_since_restart += 1
        if self.epochs_since_restart > self.t0:
            self.t0 *= self.tmult
            self.epochs_since_restart = 0
        return [self.step_size_min + 0.5 * (base_lr - self.step_size_min) *
                (1 + np.cos(self.epochs_since_restart * np.pi / self.t0))
                for base_lr in self.base_lrs]
