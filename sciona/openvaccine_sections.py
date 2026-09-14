"""Validation-gated pseudo-label sections with explicit rollback semantics.

Model weights include BatchNorm statistics. Neither policy rewinds random
streams or the caller's batch iterator. This is section rejection, not full
stochastic checkpoint replay. Exceptions restore model and optimizer state.
"""
from collections.abc import Callable, Iterable

import numpy as np

from sciona.openvaccine_targets import validation_rollback_required


def run_pseudo_label_section(runtime, batches: Iterable, validation_score: Callable,
                             *, absolute_tolerance: float, rollback_policy: str):
    if runtime.family == 'autoencoder':
        raise ValueError('Pseudo-label sections require a prediction model')
    if rollback_policy not in ('weights_only', 'model_and_optimizer'):
        raise ValueError('Explicit supported rollback policy required')
    validation_rollback_required(0., 0., absolute_tolerance=absolute_tolerance)
    before_score = validation_score(runtime)
    validation_rollback_required(before_score, before_score, absolute_tolerance=absolute_tolerance)
    optimizer = runtime.model.optimizer
    optimizer.build(runtime.model.trainable_variables)
    model_state = runtime.model.get_weights()
    optimizer_state = [v.numpy().copy() for v in optimizer.variables()]

    def restore_optimizer():
        variables = optimizer.variables()
        if len(variables) != len(optimizer_state):
            raise RuntimeError('Optimizer topology changed during section')
        for variable, value in zip(variables, optimizer_state):
            variable.assign(value)

    losses = []
    try:
        for batch in batches:
            losses.append(runtime.train_step(*batch))
        if not losses:
            raise ValueError('Pseudo-label section must contain training batches')
        after_score = validation_score(runtime)
        rollback = validation_rollback_required(before_score, after_score,
                                                absolute_tolerance=absolute_tolerance)
        if rollback:
            runtime.model.set_weights(model_state)
            if rollback_policy == 'model_and_optimizer':
                restore_optimizer()
    except BaseException:
        runtime.model.set_weights(model_state)
        restore_optimizer()
        raise
    return dict(steps=len(losses), mean_training_loss=float(np.mean(losses)),
                validation_before=float(before_score), validation_after_attempt=float(after_score),
                rolled_back=rollback, rollback_policy=rollback_policy,
                optimizer_iteration=int(optimizer.iterations.numpy()))
