"""Winning EfficientNet epoch schedule, including the 20-epoch refit clock."""
import math


def learning_rate(epoch):
    """Use the first 14 values unchanged for the reported full-data refit.

The source refit callback still invokes lrfn(epoch) with its default 20-epoch
schedule, rather than rescaling cosine decay to the shorter training run.
"""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < 20:
        raise ValueError('Epoch must be an integer from 0 through 19')
    if epoch < 4:
        return 1e-6 + (2e-4 + 1e-6) * (epoch / 4) ** 2.5
    return 1e-6 + (2e-4 - 1e-6) * (1 + math.cos((epoch - 4) * math.pi / 15)) / 2
