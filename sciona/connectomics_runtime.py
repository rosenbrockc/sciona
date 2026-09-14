"""Private-input boundary for corrected, pinned Connectomics execution."""
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from sciona.connectomics_source import source_namespace


@dataclass(frozen=True)
class Prepared:
    signals: np.ndarray
    mode: str
    directivity: bool


def prepare(payload):
    if not isinstance(payload, dict) or set(payload) != {'version', 'signals', 'mode', 'directivity'}:
        raise ValueError('Expected version, signals, mode and directivity')
    if type(payload['version']) is not int or payload['version'] != 1:
        raise ValueError('Unsupported Connectomics contract version')
    if payload['mode'] not in ('simple', 'tuned') or type(payload['directivity']) is not bool:
        raise ValueError('Expected simple/tuned mode and boolean directivity')
    values = np.asarray(payload['signals'])
    if values.dtype.kind not in 'fiu' or values.ndim != 2:
        raise ValueError('Expected real numeric time-by-node matrix')
    samples, nodes = values.shape
    if nodes < 3 or samples < max(5, nodes + 1):
        raise ValueError('Need at least three nodes and more samples than nodes')
    with np.errstate(over='ignore', invalid='ignore'):
        signals = np.array(values, dtype=np.float32, order='F', copy=True)
    if not np.isfinite(signals).all() or (signals < 0).any():
        raise ValueError('Expected finite nonnegative float32-compatible signals')
    if np.any(np.ptp(signals, axis=0) == 0):
        raise ValueError('Constant channels cannot support this precision estimator')
    signals.flags.writeable = False
    return Prepared(signals, payload['mode'], payload['directivity'])


def execute(prepared):
    if not isinstance(prepared, Prepared):
        raise ValueError('Prepared Connectomics input required')
    # Revalidate even a manually constructed or subsequently modified object.
    prepared = prepare(dict(version=1, signals=prepared.signals,
                            mode=prepared.mode, directivity=prepared.directivity))
    directory = os.environ.get('SCIONA_CONNECTOMICS_SOURCE_DIR')
    if not directory:
        raise ValueError('Provision SCIONA_CONNECTOMICS_SOURCE_DIR before execution')
    ns = source_namespace(Path(directory))
    historical = ns['PCA']

    class CheckedPCA(historical):
        def fit(self, values, *args, **kwargs):
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite filtered signals')
            result = super().fit(values, *args, **kwargs)
            if not np.isfinite(self.noise_variance_) or self.noise_variance_ <= 0:
                raise ValueError('Degenerate residual covariance')
            return result

        def get_precision(self):
            result = super().get_precision()
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite historical precision')
            return result

    original_scale = ns['scale']

    def checked_scale(values):
        diagonal_min = values.copy()
        np.fill_diagonal(diagonal_min, diagonal_min.min())
        if not np.isfinite(diagonal_min).all() or np.ptp(diagonal_min) <= 0:
            raise ValueError('Degenerate score normalization')
        result = original_scale(values)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite normalized scores')
        return result

    ns.update(PCA=CheckedPCA, scale=checked_scale, print=lambda *a, **kw: None)
    scores = ns['make_' + prepared.mode + '_inference'](prepared.signals)
    if prepared.directivity:
        directed = ns['make_prediction_directivity'](prepared.signals, n_jobs=1)
        scores = .997 * scores + .003 * directed
    return dict(version=1, mode=prepared.mode, directivity=prepared.directivity,
                pca_fits=240 if prepared.mode == 'simple' else 480,
                scores=scores.tolist())
