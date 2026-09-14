"""Source DFDC polynomial scheduler, including its mutating get_lr behavior.

MIT 2020 Selim Seferbekov; docs/licenses/DFDC-MIT.txt. Source commit
89c6290490bac96b29193a4061b3db9dd3933e36. Logging get_lr is computationally
significant. This class must not be replaced with a pure polynomial schedule.
"""

from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to poly learning rate policy
    """
    def __init__(self, optimizer, max_iter=90000, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.last_epoch = (self.last_epoch + 1) % self.max_iter
        return [base_lr * ((1 - float(self.last_epoch) / self.max_iter) ** (self.power)) for base_lr in self.base_lrs]
