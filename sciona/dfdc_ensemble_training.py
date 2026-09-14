"""Sequential DFDC training runs with explicit ordered checkpoint collection.

Uses the source-selected B7/SGD/PolyLR configuration. CPU and deterministic
initialization adaptations are documented in the component validators.
"""
from collections.abc import Mapping
import random

import numpy as np
import torch

from sciona.dfdc_classifier import build_classifier
from sciona.dfdc_scheduler import PolyLR
from sciona.dfdc_selection import selection_plan, select_snapshots
from sciona.dfdc_training import train


def _train_runs(records, plan, factory, *, detector, predictor, batch_size,
                batches_per_epoch, test_every):
    if not isinstance(plan, Mapping):
        raise ValueError('an explicit executable selection plan is required')
    checked = selection_plan(plan.get('runs', ()), plan.get('requests', ()))
    if dict(plan) != checked:
        raise ValueError('selection plan metadata is inconsistent')
    requested = set(map(tuple, checked['requests']))
    snapshots = {}; summaries = []
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            for seed, fold, epochs in checked['runs']:
                model, optimizer, scheduler = factory(seed, fold)
                def collect(event):
                    kind = event['kind']
                    if not kind.isdecimal():
                        return
                    key = (seed, fold, int(kind))
                    if key in requested:
                        if key in snapshots:
                            raise ValueError('duplicate numbered checkpoint emission')
                        snapshots[key] = event
                metrics = train(model, optimizer, scheduler, records, detector=detector,
                                predictor=predictor, seed=seed, fold=fold, epochs=epochs,
                                batch_size=batch_size, batches_per_epoch=batches_per_epoch,
                                test_every=test_every, checkpoint_sink=collect)
                summaries.append({'seed': seed, 'fold': fold, 'training': metrics})
                del model, optimizer, scheduler
            return {'plan': checked, 'runs': summaries,
                    'snapshots': select_snapshots(checked, snapshots)}
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def train_ensemble(records, plan, *, detector, predictor, initialization,
                   states=None, batch_size=12, batches_per_epoch=2500, test_every=1):
    """Train fresh full B7 runs and return only requested numbered snapshots.

    Initialization is explicit: random, encoder or full state. Supplied states
    are keyed by (seed, fold); full state initializes weights, not optimizer resume.
    Seven B7 state dictionaries require substantial caller-owned memory. No files
    or historical weights are loaded automatically.
    """
    if initialization not in ('random', 'encoder', 'state'):
        raise ValueError('initialization must be random, encoder or state')
    if initialization == 'random' and states is not None:
        raise ValueError('random initialization cannot receive states')
    if initialization != 'random':
        if not isinstance(plan, Mapping) or not isinstance(states, Mapping):
            raise ValueError('explicit per-run initialization states are required')
        checked = selection_plan(plan.get('runs', ()), plan.get('requests', ()))
        if set(states) != {tuple(row[:2]) for row in checked['runs']}:
            raise ValueError('initialization states must exactly match configured runs')
    def factory(seed, fold):
        model = build_classifier(initialization=initialization, seed=seed,
                                 state=None if states is None else states[(seed, fold)])
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9,
                                    weight_decay=.0001, nesterov=True)
        scheduler = PolyLR(optimizer, max_iter=100500)
        return model, optimizer, scheduler
    return _train_runs(records, plan, factory, detector=detector, predictor=predictor,
                       batch_size=batch_size, batches_per_epoch=batches_per_epoch,
                       test_every=test_every)
