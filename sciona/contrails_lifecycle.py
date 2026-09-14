"""Complete selected-fold training and final Contrails inference orchestration."""
import gc

import numpy as np
import torch

from sciona.contrails_checkpoints import checked_load
from sciona.contrails_defaults import defaults
from sciona.contrails_fold import train_fold
from sciona.contrails_population import make_loader, select_folds
from sciona.contrails_prediction import prepare_inference, ensemble_probabilities, encode_mask
from sciona.contrails_single_model import Model as SingleTraining
from sciona.contrails_temporal_model import Model as TemporalTraining
from sciona.contrails_single_inference import Model as SingleInference
from sciona.contrails_temporal_inference import Model as TemporalInference


def run_lifecycle(training, scoring, prediction, *, initialization, initial_states=None,
                  variant='v47', device='cpu', batch_size=None, num_workers=None,
                  epoch_limit=None):
    """Train all final selected folds and predict in caller-provided example order.

    Explicit epoch_limit truncates the source schedule without rescaling it and is
    recorded as a reduced training run. Initial states map branch to shared encoder
    state or fold to full-model state, according to the explicit policy.
    """
    if initialization not in ('random', 'encoder', 'full_model'):
        raise ValueError('Explicit initialization policy required')
    if initialization == 'random' and initial_states is not None:
        raise ValueError('Random initialization cannot supply states')
    if not scoring or not prediction:
        raise ValueError('Scoring and prediction populations must be nonempty')
    if epoch_limit is not None and (type(epoch_limit) is not int or epoch_limit < 1):
        raise ValueError('Epoch limit must be a positive integer')
    configs = defaults()
    selections = {b: select_folds(len(training), branch=b, variant=variant) for b in configs}
    if initialization != 'random':
        if initial_states is None or set(initial_states) != set(configs):
            raise ValueError('Both branch initial states required')
        if initialization == 'full_model':
            for branch, folds in selections.items():
                if set(initial_states[branch]) != {f for f, _, _ in folds}:
                    raise ValueError('Exact selected fold checkpoints required')
    probabilities = {}
    fold_reports = []
    for branch, training_type, inference_type in [
        ('temporal', TemporalTraining, TemporalInference), ('single', SingleTraining, SingleInference)
    ]:
        cfg = configs[branch]
        size = cfg['train']['batch_size'] if batch_size is None else batch_size
        workers = cfg['train']['num_workers'] if num_workers is None else num_workers
        branch_probs = []
        for fold, train_indices, val_indices in selections[branch]:
            train_loader = make_loader(training, train_indices, branch=branch, training=True,
                                       batch_size=size, num_workers=workers)
            val_loader = make_loader(training, val_indices, branch=branch, training=False,
                                     batch_size=size, num_workers=workers)
            scoring_loader = make_loader(scoring, range(len(scoring)), branch=branch, training=False,
                                         batch_size=size, num_workers=workers)
            state = None if initialization == 'random' else initial_states[branch]
            if initialization == 'full_model':
                state = state[fold]
            model = training_type(cfg, pretrained=False)
            trained = train_fold(model, cfg, train_loader, val_loader, scoring_loader,
                                 initialization=initialization, initial_state=state, device=device,
                                 epoch_limit=epoch_limit)
            del model
            gc.collect()
            inference_cfg = dict(cfg['model'], resize=cfg['data']['resize'], tta='d4prob')
            model = inference_type(inference_cfg, pretrained=False)
            checked_load(model, trained.pop('checkpoint'))
            model.to(device).eval()
            predictions = []
            for thermal in prediction:
                x = prepare_inference(thermal, branch=branch).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, logits = model(x)
                predictions.append(logits.sigmoid().cpu().numpy()[0])
            branch_probs.append(np.stack(predictions))
            fold_reports.append(dict(branch=branch, fold=fold, training=trained))
            del model, train_loader, val_loader, scoring_loader
            gc.collect()
        probabilities[branch] = np.stack(branch_probs)
    combined = ensemble_probabilities(probabilities['temporal'], probabilities['single'], variant=variant)
    return dict(probabilities=combined, masks=[encode_mask(x) for x in combined], folds=fold_reports,
                variant=variant, initialization=initialization, epoch_limit=epoch_limit,
                batch_size=batch_size, num_workers=num_workers)
