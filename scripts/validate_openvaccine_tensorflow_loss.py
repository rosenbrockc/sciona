"""Pinned source loss value and gradient comparisons on synthetic tensors."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ.setdefault('KERAS_HOME', '/private/tmp/sciona_openvaccine_keras')
sys.path.insert(0, '/private/tmp/sciona_openvaccine_tensorflow')
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]


def main():
    cache = Path('/private/tmp/sciona_openvaccine_source')
    manifest = json.loads((cache/'manifest.json').read_text())
    name = 'scripts/nullrecurrent_inference.py'
    raw = (cache/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == next(p['sha256'] for p in manifest['pins'] if p['software_path'] == name)
    definition = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == 'MSE')
    weights = np.array([0.3, 0.3, 0.3, 0.05, 0.05])
    ns = dict(tf=tf, losses=tf.keras.losses, LOSS_WGTS=weights.tolist())
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<pinned-source-loss>', 'exec'), ns)
    loss = ns['MSE'](reduction=tf.keras.losses.Reduction.NONE)
    rng = np.random.default_rng(237)
    cases = 0
    for batch, length in [(1, 2), (2, 5), (4, 7)]:
        for masking in ['none', 'partial', 'all']:
            truth, predicted = rng.normal(size=(2, batch, length, 5)).astype(np.float32)
            if masking == 'partial': truth[:, ::2, :] = np.nan
            if masking == 'all': truth[:] = np.nan
            sample_weight = rng.uniform(0.1, 2, size=batch).astype(np.float32)
            variable = tf.Variable(predicted)
            with tf.GradientTape() as tape:
                value = loss(tf.constant(truth), variable, tf.constant(sample_weight))
            gradient = tape.gradient(value, variable)
            assert gradient is not None
            residual = np.where(np.isnan(truth), 0, predicted.astype(np.float64)-truth.astype(np.float64))
            rms = np.sqrt(np.mean(residual**2, axis=1) + 1e-12)
            normalized = sample_weight.astype(np.float64)/sum(sample_weight.astype(np.float64))
            expected_value = np.sum(rms * normalized[:, None] * weights[None, :])
            expected_gradient = (residual / (length*rms[:, None, :])
                                 * normalized[:, None, None] * weights[None, None, :])
            np.testing.assert_allclose(value.numpy(), expected_value, rtol=2e-6, atol=1e-7)
            np.testing.assert_allclose(gradient.numpy(), expected_gradient, rtol=3e-6, atol=1e-7)
            assert np.all(gradient.numpy()[np.isnan(truth)] == 0)
            cases += 1
    try:
        loss(tf.zeros((1, 2, 5)), tf.ones((1, 2, 5)))
    except (ValueError, TypeError) as error:
        missing_weight_error = type(error).__name__
    else:
        raise AssertionError('Source unexpectedly accepts omitted sample weights')
    report = dict(status='passed', synthetic_only=True, source_commit=manifest['commit'],
                  tensorflow_version=tf.__version__, legacy_keras=True, loss_gradient_cases=cases,
                  masked_gradients_zero=True, normalization='All sequence positions, including masked positions',
                  all_masked_loss=1e-6, omitted_sample_weights_error=missing_weight_error,
                  scope='Exact source custom loss and gradients only; no full model or training lifecycle claim.',
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_tensorflow_loss.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
