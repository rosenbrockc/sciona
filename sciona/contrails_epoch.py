"""In-memory extraction of the winning Contrails training epoch.

Copyright (c) 2023 Jun Koda; MIT, docs/licenses/Contrails-MIT.txt.
Source 08a15beb36f9cbed4c3990e74c625c1332b61fe8, final evaluate.py loops.
Validation and persistence are supplied by the outer lifecycle.
"""
import torch


def train_epoch(model, batches, criterion, optimizer, scheduler, *, epoch,
                accumulate, validations_per_epoch, device='cpu', validate=None):
    """Preserve source accumulation, value clipping and validation-break timing.

    An unmatched zero/duplicate validation checkpoint is left unmatched, as in
    source. Partial accumulation groups are not flushed at epoch end.
    """
    nbatch = len(batches)
    if nbatch < 1 or type(accumulate) is not int or accumulate < 1:
        raise ValueError('Nonempty batches and positive integer accumulation required')
    if type(validations_per_epoch) is not int or validations_per_epoch < 1:
        raise ValueError('Positive integer validation count required')
    nsteps = nbatch // accumulate
    icheck = [nsteps * (i + 1) // validations_per_epoch - 1
              for i in range(validations_per_epoch)]
    model.train()
    optimizer.zero_grad()
    istep = 0
    result = {'batches_processed': 0, 'optimizer_steps': 0, 'learning_rates': [],
              'validation_epochs': [], 'scheduler_epochs': [], 'losses': []}
    for ibatch, batch in enumerate(batches):
        x, y, y_sym, w = (batch[k].to(device) for k in ['x', 'y', 'y_sym', 'w'])
        y_sym_pred, y_pred = model(x)
        loss = criterion(y_sym_pred, y_sym, y_pred, y, w)
        result['losses'].append(float(loss.detach()))
        if accumulate > 1:
            loss = loss / accumulate
        loss.backward()
        result['batches_processed'] += 1
        ep = epoch + (ibatch + 1) / nbatch
        if (ibatch + 1) % accumulate == 0:
            istep += 1
            torch.nn.utils.clip_grad_value_(model.parameters(), 1000.)
            optimizer.step()
            optimizer.zero_grad()
            result['optimizer_steps'] += 1
            result['learning_rates'].append((ep, optimizer.param_groups[0]['lr']))
            if istep == icheck[0]:
                icheck.pop(0)
                result['validation_epochs'].append(ep)
                if validate is not None:
                    validate(model, ep)
                if not icheck:
                    break
            scheduler.step(ep)
            result['scheduler_epochs'].append(ep)
    return result
