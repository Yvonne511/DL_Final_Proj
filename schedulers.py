# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from enum import auto, Enum
import math


class LRSchedule(Enum):
    Constant = auto()
    Cosine = auto()


class Scheduler:
    def __init__(
        self,
        schedule: str,
        base_lr: float,
        data_loader,
        epochs: int,
        optimizer,
        batch_steps=None,
        batch_size=None,
    ):
        self.schedule = schedule
        self.base_lr = base_lr
        self.data_loader = data_loader
        self.epochs = epochs
        self.optimizer = optimizer

        if batch_size is None:
            self.batch_size = data_loader.config.batch_size
        else:
            self.batch_size = batch_size

        if batch_steps is None:
            self.batch_steps = len(data_loader)
        else:
            self.batch_steps = batch_steps

    def adjust_learning_rate(self, step: int):
        if self.schedule == LRSchedule.Constant:
            return self.base_lr
        else:
            max_steps = self.epochs * self.batch_steps
            warmup_steps = int(0.10 * max_steps)
            for param_group in self.optimizer.param_groups:
                base_lr = (
                    param_group["base_lr"] if "base_lr" in param_group else self.base_lr
                )
                base_lr = base_lr * self.batch_size / 256
                if step < warmup_steps:
                    lr = base_lr * step / warmup_steps
                else:
                    step -= warmup_steps
                    max_steps -= warmup_steps
                    q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
                    end_lr = base_lr * 0.001
                    lr = base_lr * q + end_lr * (1 - q)
                param_group["lr"] = lr
            return lr

class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd