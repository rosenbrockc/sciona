"""Connected DFDC CPU training lifecycle over explicitly prepared records.

MIT source: Selim Seferbekov, 89c6290490bac96b29193a4061b3db9dd3933e36;
docs/licenses/DFDC-MIT.txt. Single-process, no frozen epochs or resume mode.
Python and Torch seeding/isolation are explicit reproducibility adaptations;
the source seed itself controls dataset resets, not model initialization.
"""
import random

import numpy as np
import torch

from sciona.dfdc_dataset import PreparedDataset, make_loader
from sciona.dfdc_evaluation import ValidationDataset, evaluate
from sciona.dfdc_epoch import train_epoch
from sciona.dfdc_checkpoints import finish_epoch


def train(model, optimizer, scheduler, records, *, detector, predictor, seed,
          checkpoint_sink, fold=0, epochs=40, batch_size=12,
          batches_per_epoch=2500, test_every=1):
    """Run source reset/train/save/validate order; emit snapshots to a caller sink.

    The supplied model and optimizer are updated in place. The sink receives
    tensor states and must manage their lifetime; only aggregate metrics return.
    Initialization, record construction and historical checkpoint selection are
    separate explicit stages. Caller RNG streams are restored even on failure.
    """
    for name, value in [('epochs', epochs), ('batch_size', batch_size),
                        ('batches_per_epoch', batches_per_epoch), ('test_every', test_every)]:
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if type(seed) is not int or seed < 0 or (max(epochs, 2)*seed) >= 2**32:
        raise ValueError('seed must support all source population resets')
    if not callable(checkpoint_sink):
        raise ValueError('checkpoint sink must be callable')
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            torch.manual_seed(seed)
            data_train = PreparedDataset(records, mode='train', detector=detector,
                                         predictor=predictor, fold=fold)
            data_val = ValidationDataset(records, fold=fold)
            data_val.reset(1, seed)
            val_loader = make_loader(data_val, batch_size=batch_size)
            best_loss = 100.
            history = []
            for epoch in range(epochs):
                data_train.reset(epoch, seed)
                model.encoder.train()
                for parameter in model.encoder.parameters():
                    parameter.requires_grad = True
                train_loader = make_loader(data_train, batch_size=batch_size)
                metrics = train_epoch(model, optimizer, scheduler, train_loader,
                                      epoch=epoch, batches_per_epoch=batches_per_epoch)
                model.eval()
                snapshots = finish_epoch(model, epoch=epoch, best_loss=best_loss)
                for event in snapshots['events']:
                    checkpoint_sink(event)
                del snapshots
                validation = None
                if (epoch + 1) % test_every == 0:
                    validation = evaluate(model, val_loader)
                    snapshots = finish_epoch(model, epoch=epoch, best_loss=best_loss,
                                             validation_loss=validation['loss'])
                    for event in snapshots['events'][2:]:
                        checkpoint_sink(event)
                    best_loss = snapshots['best_loss']
                    del snapshots
                history.append({'epoch': epoch, 'training': metrics,
                                'validation': validation, 'best_loss': best_loss})
            return {'epochs': history, 'best_loss': best_loss}
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
