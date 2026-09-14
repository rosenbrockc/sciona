"""Contrails fold training with terminal checkpoint capture.

Combines source-validated components. All loaders and initial state are supplied
explicitly; no filesystem access or implicit weight downloads.
"""
import torch

from sciona.contrails_checkpoints import initialize_model, terminal_checkpoint
from sciona.contrails_epoch import train_epoch
from sciona.contrails_evaluation import evaluate
from sciona.contrails_losses import BCELoss
from sciona.contrails_scheduler import Scheduler
from sciona.contrails_score import compute


def train_fold(model, cfg, training_loader, validation_loader, scoring_loader, *,
               initialization, initial_state=None, device='cpu', periodic_sink=None, epoch_limit=None):
    """Run the configured schedule, observe validation, capture the terminal state.

    Diagnostic threshold search at the end does not alter the fixed final ensemble
    threshold. A periodic sink receives snapshots only when configured.
    """
    initialize_model(model, policy=initialization, state=initial_state)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                  weight_decay=float(cfg['train']['weight_decay']))
    scheduler = Scheduler(optimizer, cfg['scheduler'])
    if epoch_limit is not None and (type(epoch_limit) is not int or not 1 <= epoch_limit <= len(scheduler)):
        raise ValueError('Epoch limit must lie within the source schedule')
    reports = []
    observations = []

    def observe(current, epoch):
        arguments = dict(augment_fraction=cfg['data']['augment_prob'], device=device)
        validation = evaluate(current, validation_loader, threshold=cfg['val']['th'], **arguments)
        scoring = evaluate(current, scoring_loader, threshold=cfg['test']['th'], **arguments)
        observations.append(dict(epoch=epoch, validation=validation, scoring=scoring))

    for epoch in range(len(scheduler) if epoch_limit is None else epoch_limit):
        reports.append(train_epoch(model, training_loader, BCELoss(), optimizer, scheduler,
                                   epoch=epoch, accumulate=cfg['train']['accumulate'],
                                   validations_per_epoch=cfg['val']['per_epoch'],
                                   device=device, validate=observe))
        if cfg['train'].get('checkpoint', False) and (epoch + 1) % 10 == 0:
            if periodic_sink is not None:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                periodic_sink(epoch + 1, state)
    if not sum(r['optimizer_steps'] for r in reports):
        raise ValueError('Fold completed without an optimizer update')
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Nonfinite terminal model state')
    diagnostic = compute(model, scoring_loader, device)
    return {'checkpoint': terminal_checkpoint(model), 'epochs': reports,
            'observations': observations, 'diagnostic': diagnostic,
            'initialization': initialization, 'checkpoint_selection': 'terminal'}
