"""Strict execution boundary for the complete periodic diffusion surrogate."""
from dataclasses import dataclass
import numpy as np
import torch
from sciona.physical_operator_state import prepare as validate, tensors, finite_json
from sciona.physical_operator_training import fit
from sciona.physical_operator_projection import project_and_smooth

@dataclass(frozen=True)
class Prepared:
    payload: dict


def prepare(payload):return Prepared(validate(payload))


def execute(prepared):
    if not isinstance(prepared,Prepared):raise ValueError('Prepared physical state required')
    p=validate(prepared.payload)
    fitted=fit(p)
    x,t=tensors(p['query'],p['length'])
    with torch.no_grad():raw=fitted.model(x,t).numpy()
    predictions=project_and_smooth(raw,p['query']['states'],smoothing=p['controls']['smoothing'])
    initial=np.asarray(p['query']['states'],dtype=np.float64)
    result=dict(predictions=predictions.tolist(),training_rows=len(p['training']['states']),
        validation_rows=len(p['validation']['states']),query_rows=len(initial),grid_points=initial.shape[1],
        epochs=p['controls']['epochs'],best_epoch=fitted.best_epoch,validation_mse=fitted.validation_mse,
        initial_training_mse=fitted.history[0]['training_mse'],final_training_mse=fitted.history[-1]['training_mse'],
        max_mean_conservation_error=float(np.max(np.abs(predictions.mean(1)-initial.mean(1)))))
    finite_json(result)
    return result
