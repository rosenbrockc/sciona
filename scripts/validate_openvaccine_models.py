"""Full source topology and optimizer-step smoke tests; synthetic inputs only."""
import ast
import gc
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
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    cache = Path('/private/tmp/sciona_openvaccine_source')
    manifest = json.loads((cache/'manifest.json').read_text())
    name = 'scripts/nullrecurrent_inference.py'
    raw = (cache/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == next(p['sha256'] for p in manifest['pins'] if p['software_path'] == name)
    wanted = {'MSE', 'get_base', 'get_ae_model', 'get_model', 'forward', 'res', 'attention',
              'multi_head_attention', 'adj_attn', 'gru_layer', 'lstm_layer', 'wave_block', 'get_optimizer'}
    definitions = [n for n in ast.parse(raw).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    assert {n.name for n in definitions} == wanted
    rng = np.random.default_rng(923)
    nodes = rng.uniform(0, 1, size=(2, 10, 55)).astype(np.float32)
    adjacency = np.repeat(np.eye(10, dtype=np.float32)[None, :, :, None], 2, axis=0)
    adjacency = np.repeat(adjacency, 8, axis=-1)
    ns = dict(np=np, tf=tf, L=tf.keras.layers, K=tf.keras.backend,
              losses=tf.keras.losses, LOSS_WGTS=[0.3,0.3,0.3,0.05,0.05], X_node=nodes, As=adjacency)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-full-models>', 'exec'), ns)
    results = []
    for family in ['autoencoder', 'gru', 'lstm', 'forward', 'wave']:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(183)
        base = ns['get_base']({}, nodes, adjacency)
        model = (ns['get_ae_model'](base, {}) if family == 'autoencoder'
                 else ns['get_model'](base, {}, family, nodes, adjacency))
        inputs = [tf.constant(nodes), tf.constant(adjacency)]
        with tf.GradientTape() as tape:
            output = model(inputs, training=True)
            if family == 'autoencoder':
                value = tf.reduce_mean(output)
            else:
                assert tuple(output.shape) == (2, 8, 5)
                truth = rng.normal(size=(2, 8, 5)).astype(np.float32)
                truth[:, ::3, :] = np.nan
                value = model.loss(tf.constant(truth), output, tf.ones((2,)))
        gradients = tape.gradient(value, model.trainable_variables)
        assert np.isfinite(value.numpy()).all()
        assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
        before = model.trainable_variables[0].numpy().copy()
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        assert int(model.optimizer.iterations.numpy()) == 1
        assert not np.array_equal(before, model.trainable_variables[0].numpy())
        result = dict(family=family, parameters=model.count_params(),
                      trainable_tensors=len(gradients), optimizer_steps=1,
                      finite_gradients=True, weights_changed=True)
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, base, gradients, output, value, tape
        gc.collect()
    report = dict(status='passed', synthetic_only=True, source_commit=manifest['commit'],
                  tensorflow_version=tf.__version__, legacy_keras=True, source_modified=False,
                  full_topologies=results,
                  scope='Full-width source autoencoder and four prediction families with one optimizer step each; no ensemble, checkpoint or complete training lifecycle claim.',
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_models.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
