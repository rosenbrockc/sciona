"""Synthetic transfer and deterministic optimizer checkpoint replay on full models."""
import ast
import gc
import hashlib
import json
from pathlib import Path
import tempfile

import validate_openvaccine_models as setup
np, tf, ROOT = setup.np, setup.tf, setup.ROOT


def arrays_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a, b)


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    cache = Path('/private/tmp/sciona_openvaccine_source')
    manifest = json.loads((cache/'manifest.json').read_text())
    raw = (cache/'scripts/nullrecurrent_inference.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == next(p['sha256'] for p in manifest['pins'] if p['software_path']=='scripts/nullrecurrent_inference.py')
    names = {'MSE','get_base','get_ae_model','get_model','forward','res','attention',
             'multi_head_attention','adj_attn','gru_layer','lstm_layer','wave_block','get_optimizer'}
    definitions = [n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in definitions} == names
    rng = np.random.default_rng(772)
    nodes = rng.uniform(0,1,(2,10,55)).astype(np.float32)
    adj = np.tile(np.eye(10,dtype=np.float32)[None,:,:,None],(2,1,1,8))
    inputs = [tf.constant(nodes),tf.constant(adj)]
    ns = dict(np=np,tf=tf,L=tf.keras.layers,K=tf.keras.backend,losses=tf.keras.losses,
              LOSS_WGTS=[.3,.3,.3,.05,.05],X_node=nodes,As=adj)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'<pinned-models>','exec'),ns)
    tf.keras.utils.set_random_seed(48)
    base = ns['get_base']({},nodes,adj)
    before = base.get_weights()
    ae = ns['get_ae_model'](base,{})
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(ae(inputs,training=True))
    grads = tape.gradient(loss,ae.trainable_variables)
    ae.optimizer.apply_gradients(zip(grads,ae.trainable_variables))
    learned = base.get_weights()
    assert any(not np.array_equal(a,b) for a,b in zip(before,learned))
    features = base(inputs,training=False).numpy()
    del ae,base,before,grads,tape
    gc.collect()
    results=[]
    for family in ['gru','lstm','forward','wave']:
        tf.keras.backend.clear_session()
        new_base=ns['get_base']({},nodes,adj)
        new_base.set_weights(learned)
        np.testing.assert_array_equal(new_base(inputs,training=False).numpy(),features)
        model=ns['get_model'](new_base,{},family,nodes,adj)
        assert int(model.optimizer.iterations.numpy())==0
        truth=tf.constant(rng.normal(size=(2,8,5)).astype(np.float32))
        def step(target):
            # Deliberately deterministic: this isolates optimizer restoration.
            # Dropout/RNG replay and training-mode BatchNorm remain separate work.
            with tf.GradientTape() as tape:
                value=target.loss(truth,target(inputs,training=False),tf.ones((2,)))
            gradients=tape.gradient(value,target.trainable_variables)
            assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
            target.optimizer.apply_gradients(zip(gradients,target.trainable_variables))
            return value.numpy()
        step(model)
        prediction=model(inputs,training=False).numpy()
        with tempfile.TemporaryDirectory(prefix='sciona_openvaccine_synthetic_checkpoint_') as temp:
            checkpoint=tf.train.Checkpoint(model=model,optimizer=model.optimizer)
            saved=checkpoint.save(str(Path(temp)/'state'))
            expected_loss=step(model)
            expected_weights=model.get_weights()
            expected_opt=[v.numpy() for v in model.optimizer.variables()]
            del model,new_base,checkpoint
            gc.collect()
            fresh_base=ns['get_base']({},nodes,adj)
            restored=ns['get_model'](fresh_base,{},family,nodes,adj)
            restored.optimizer.build(restored.trainable_variables)
            checkpoint=tf.train.Checkpoint(model=restored,optimizer=restored.optimizer)
            checkpoint.restore(saved).assert_consumed()
            assert int(restored.optimizer.iterations.numpy())==1
            np.testing.assert_array_equal(restored(inputs,training=False).numpy(),prediction)
            np.testing.assert_array_equal(step(restored),expected_loss)
            arrays_equal(restored.get_weights(),expected_weights)
            arrays_equal([v.numpy() for v in restored.optimizer.variables()],expected_opt)
        results.append(dict(family=family,autoencoder_transfer_exact=True,
                            checkpoint_prediction_exact=True,next_deterministic_step_exact=True,
                            optimizer_state_exact=True))
        print(json.dumps(results[-1]),flush=True)
        del restored,fresh_base,checkpoint,expected_weights,expected_opt
        gc.collect()
    report=dict(status='passed',synthetic_only=True,source_commit=manifest['commit'],
                tensorflow_version=tf.__version__,families=results,
                scope='Autoencoder base transfer and full model/Adam restoration with dropout disabled and BatchNorm inference mode for deterministic replay; stochastic training replay and complete lifecycle remain unverified.',
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                setup_sha256=hashlib.sha256(Path(setup.__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_checkpoints.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
