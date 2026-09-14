"""Independent NumPy reference for the winner's 5-fold, 10-seed neural bag.

Eight ReLU units, two softmax outputs, mean cross entropy, Glorot-uniform
weights and zero biases. AdaGrad uses sqrt(accumulator + 1e-6), learning rate
0.01. Modern stratified split allocation and float64 arithmetic are explicit
reconstruction choices, not a claim of historical Lasagne bitwise parity.
"""
import io
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def loss_gradients(x, y, parameters):
    a, b, c, d = parameters
    hidden_raw = x @ a + b
    hidden = np.maximum(hidden_raw, 0)
    logits = hidden @ c + d
    logits -= logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    loss = np.mean(np.log(exponent.sum(axis=1)) - logits[np.arange(len(y)), y])
    delta = probabilities.copy()
    delta[np.arange(len(y)), y] -= 1
    delta /= len(y)
    back = (delta @ c.T) * (hidden_raw > 0)
    return loss, (x.T @ back, back.sum(axis=0), hidden.T @ delta, delta.sum(axis=0))


def predict(x, parameters):
    a, b, c, d = parameters
    logits = np.maximum(x @ a + b, 0) @ c + d
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp[:, 1] / exp.sum(axis=1)


def neural_partitions(labels):
    """Validate labels and construct the complete fold plan before any fitting."""
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) < 5 or not np.isfinite(y).all() or set(y) != {0, 1}:
        raise ValueError('Finite binary labels and at least five rows required')
    y = y.astype(np.int64)
    partitions = []
    for fit, held in KFold(5, shuffle=True, random_state=555).split(y):
        if np.bincount(y[fit], minlength=2).min() < 200:
            raise ValueError('Every outer training fold needs at least 200 rows per class')
        inner, validation = next(StratifiedKFold(200, shuffle=False).split(y[fit, None], y[fit]))
        partitions.append((fit, held, inner, validation))
    return partitions


def checkpoint_predictions(raw_features, parameters, mean, scale):
    """Serialize parameters and scaler as numeric arrays; replay from raw inputs."""
    buffer = io.BytesIO()
    np.savez(buffer, *parameters, mean=mean, scale=scale)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        restored = [saved['arr_'+str(i)] for i in range(4)]
        transformed = (raw_features - saved['mean']) / saved['scale']
        return predict(transformed, restored)


def train_neural(features, labels, query_features):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    query = np.asarray(query_features, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] == 0 or y.shape != (len(x),)
            or set(y) != {0, 1} or not np.isfinite(x).all()):
        raise ValueError('Finite training features and aligned binary labels required')
    if (query.ndim != 2 or query.shape[1] != x.shape[1] or len(query) == 0
            or not np.isfinite(query).all()):
        raise ValueError('Finite nonempty query features must match training width')
    y = y.astype(np.int64)
    partitions = neural_partitions(y)
    oof = np.zeros(len(x))
    query_sum = np.zeros(len(query))
    fits = []
    for fold, (fit, held, inner, validation) in enumerate(partitions):
        scaler = StandardScaler().fit(x[fit])
        scaled = scaler.transform(x[fit])
        tx, ty = scaled[inner], y[fit][inner]
        hx, qx = scaler.transform(x[held]), scaler.transform(query)
        for seed in range(10):
            rng = np.random.RandomState(seed)
            params = [rng.uniform(-np.sqrt(6 / (x.shape[1] + 8)),
                                  np.sqrt(6 / (x.shape[1] + 8)), (x.shape[1], 8)),
                      np.zeros(8), rng.uniform(-np.sqrt(.6), np.sqrt(.6), (8, 2)), np.zeros(2)]
            sums = [np.zeros_like(p) for p in params]
            updates = 0
            for epoch in range(3000):
                for start in range(0, len(tx), 128):
                    loss, gradients = loss_gradients(tx[start:start+128], ty[start:start+128], params)
                    if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients):
                        raise ValueError('Nonfinite neural optimization state')
                    for p, accumulator, gradient in zip(params, sums, gradients):
                        accumulator += gradient * gradient
                        p -= .01 * gradient / np.sqrt(accumulator + 1e-6)
                    updates += 1
            validation_loss, _ = loss_gradients(scaled[validation], y[fit][validation], params)
            hp, qp = predict(hx, params), predict(qx, params)
            if not np.isfinite(hp).all() or not np.isfinite(qp).all():
                raise ValueError('Nonfinite neural prediction')
            if not np.isfinite(validation_loss):
                raise ValueError('Nonfinite neural validation loss')
            for raw, expected in ((x[held], hp), (query, qp)):
                replay = checkpoint_predictions(raw, params, scaler.mean_, scaler.scale_)
                if not np.array_equal(replay, expected):
                    raise RuntimeError('Neural parameter/scaler checkpoint replay differed')
            oof[held] += hp / 10
            query_sum += qp / 50
            fits.append(dict(fold=fold, seed=seed, epochs=3000, optimizer_updates=updates,
                             validation_loss=float(validation_loss)))
    return dict(oof=oof, query=query_sum, fits=fits, checkpoint_replays_exact=True)
