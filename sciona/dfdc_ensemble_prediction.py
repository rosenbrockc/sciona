"""DFDC selected checkpoint states to ordered full B7 ensemble inference."""
from collections.abc import Mapping

import torch

from sciona.dfdc_classifier import build_classifier
from sciona.dfdc_prediction import predict_video
from sciona.dfdc_selection import selection_plan, select_snapshots


def build_selected_models(bundle, *, precision='float16'):
    """Validate selection metadata and strictly load every selected model state.

    Order follows the explicit plan. State metadata does not authenticate original
    competition weights. CPU float16 is the source default; float32 is explicit.
    """
    if precision not in ('float16', 'float32'):
        raise ValueError('precision must be float16 or float32')
    if not isinstance(bundle, Mapping) or not isinstance(bundle.get('plan'), Mapping):
        raise ValueError('selected checkpoints require an explicit plan')
    plan = bundle['plan']
    checked = selection_plan(plan.get('runs', ()), plan.get('requests', ()))
    if dict(plan) != checked:
        raise ValueError('selection plan metadata is inconsistent')
    snapshots = bundle.get('snapshots')
    if not isinstance(snapshots, (list, tuple)) or len(snapshots) != len(checked['requests']):
        raise ValueError('snapshot count must match requested ensemble')
    registry = {tuple(key): value for key, value in zip(checked['requests'], snapshots)}
    ordered = select_snapshots(checked, registry)
    models = []
    for request, snapshot in zip(checked['requests'], ordered):
        model = build_classifier(initialization='state', seed=request[0], state=snapshot['state_dict'])
        model.eval().to(dtype=getattr(torch, precision))
        # Float32-finite inputs may overflow on conversion to source float16.
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError('selected state is nonfinite at requested inference precision')
        models.append(model)
    return models


def predict_selected(decoded_clips, bundle, *, detector, frames_per_video=32,
                     precision='float16'):
    """Load once and predict positional clips, exposing only scores and status."""
    models = build_selected_models(bundle, precision=precision)
    return [predict_video(clip, detector, models, frames_per_video=frames_per_video,
                          precision=precision) for clip in decoded_clips]
