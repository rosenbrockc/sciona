"""Cornell learning rate at iteration start, preserving the historical horizon."""
import math


def learning_rate(iteration, *, steps_per_epoch, epochs, peak):
    """One-epoch linear warmup followed by source cosine with full-run period.

    The cosine period includes the warmup epoch in its configured length.
    Consequently a normal run ends before reaching the cosine minimum.
    A model-only continuation starts a new schedule at iteration zero.
    """
    if any(type(v) is not int for v in (iteration, steps_per_epoch, epochs)):
        raise ValueError('Expected integer schedule coordinates')
    if steps_per_epoch < 1 or epochs < 1 or steps_per_epoch*epochs < 2 or not 0 <= iteration < steps_per_epoch*epochs:
        raise ValueError('Iteration outside training schedule')
    if isinstance(peak,bool) or not isinstance(peak,(float,int)) or not math.isfinite(peak) or peak <= 0:
        raise ValueError('Expected positive finite peak learning rate')
    minimum=peak*.01
    if iteration < steps_per_epoch:
        return minimum+(peak-minimum)*iteration/steps_per_epoch
    progress=(iteration-steps_per_epoch)/(epochs*steps_per_epoch)
    return peak+((minimum-peak)/2)*(1-math.cos(math.pi*progress))
