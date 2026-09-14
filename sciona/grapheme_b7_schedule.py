"""Winner-post B7 learning-rate multiplier with explicit unrecovered warmup.

The post specifies the formula, 200 epochs and batch 32, but does not assign
WARM_UP_STEP for these branches. Callers must supply and qualify that setting.
"""
import math


def warmup_linear_multiplier(step, *, total_steps, warmup_steps):
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError('positive integer total steps required')
    if type(step) is not int or not 0 <= step <= total_steps:
        raise ValueError('integer step within declared training lifecycle required')
    if (isinstance(warmup_steps, bool) or not isinstance(warmup_steps, (int, float))
            or not math.isfinite(warmup_steps) or not 0 < warmup_steps < total_steps):
        raise ValueError('explicit positive finite warmup shorter than training required')
    if step < warmup_steps:
        return step / warmup_steps
    return (total_steps - step) / (total_steps - warmup_steps)
