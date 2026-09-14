"""In-memory DFDC epoch checkpoint ordering, without pickle or prediction files.

Derived from MIT 2020 Selim Seferbekov, commit
89c6290490bac96b29193a4061b3db9dd3933e36; docs/licenses/DFDC-MIT.txt.
Source numbered snapshots precede validation and carry the previous best loss.
State mappings use the unwrapped model namespace, an explicit CPU adaptation.
"""
import math

import torch


def finish_epoch(model, *, epoch, best_loss, validation_loss=None):
    """Return ordered snapshot events and new best; never retain live tensors.

The caller persists selected events outside this function. There is no implied
historical checkpoint identity or optimizer/scheduler resume guarantee.
"""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError('epoch must be a nonnegative integer')
    for loss in (best_loss, validation_loss):
        if loss is not None and (isinstance(loss, bool) or not isinstance(loss, (int, float))
                                 or not math.isfinite(loss) or loss < 0):
            raise ValueError('checkpoint losses must be finite and nonnegative')
    if best_loss is None:
        raise ValueError('previous best loss is required')
    if model.training:
        raise ValueError('checkpoint model must be in evaluation mode')
    state = model.state_dict()
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('checkpoint requires finite tensor state')

    def snapshot(kind, loss):
        return {'kind': kind, 'epoch': epoch + 1, 'bce_best': float(loss),
                'state_dict': {key: value.detach().cpu().clone() for key, value in state.items()}}

    events = [snapshot('last', best_loss), snapshot(str(epoch), best_loss)]
    improved = validation_loss is not None and validation_loss < best_loss
    if validation_loss is not None:
        if improved:
            best_loss = validation_loss
            events.append(snapshot('best_dice', best_loss))
        events.append(snapshot('last', best_loss))
    return {'events': events, 'best_loss': float(best_loss), 'improved': improved}
