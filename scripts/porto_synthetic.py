"""Synthetic-only explicit Porto ensemble fixture."""
import numpy as np

def payload():
    rng = np.random.default_rng(5)
    z = rng.uniform(-1, 1, (96, 1))
    a = np.linspace(-0.9, 0.9, 8)[:, None]
    x = np.column_stack((z, z * 0.7, -z * 0.4, (z[:, 0] > 0).astype(float)))
    q = np.column_stack((a, a * 0.7, -a * 0.4, (a[:, 0] > 0).astype(float)))
    prep = dict(dropped_columns=[], categorical_columns=[3], binary_columns=[])
    models = []
    for i, width in enumerate((12, 16, 10, 14, 18)):
        dae = dict(hidden=[width, width // 2, width], feature_layers=[0, 1, 2] if i % 2 == 0 else [1], epochs=60, batch_size=32, learning_rate=0.15, decay=0.995, swap_probability=0.07, momentum=0.5)
        neural = dict(hidden=[12, 8], epochs=60, batch_size=32, learning_rate=0.15, decay=0.995, l2=0.001, momentum=0.5, dropout=0.1, input_dropout=0.05, dropout_scaling='inverted')
        models.append(dict(seed=7 + i, dae_controls=dae, neural_controls=neural))
    tree = dict(seed=7, controls=dict(rounds=12, num_leaves=3, learning_rate=0.2, min_data_in_leaf=2, feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, lambda_l2=0.0))
    return dict(version=1, training=dict(values=x.tolist(), labels=(z[:,0]>0).astype(int).tolist(), identities=[f'r{i}' for i in range(len(x))]), query=dict(values=q.tolist(), identities=[f'q{i}' for i in range(len(q))]), controls=dict(preparation=prep, neural_models=models, tree=tree))
