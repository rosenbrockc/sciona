"""Map reviewed upstream B4 tensors into the explicit Keras reconstruction."""
from sciona.cassava_efficientnet_model import build_model
from sciona.cassava_efficientnet_weights import verified_weights
from sciona.cassava_training_images import adapt_training_normalization


def build_adapted_model(*, weights, expected_sha256, training_jpegs, batch_size):
    """Initialize all backbone parameters, then adapt on the training population.

Only the three normalization-state tensors may be absent from the upstream
release. A fresh five-class head is retained. Integrity and numerical mapping
are independent of permission to redistribute the upstream weight artifact.
"""
    if weights is None:
        raise ValueError('Upstream reconstruction requires pretrained weights')
    with verified_weights(weights, expected_sha256=expected_sha256) as snapshot:
        import h5py
        import numpy as np
        from tensorflow import keras
        values = {}
        def collect(name, value):
            if isinstance(value, h5py.Dataset):
                parts = name.split('/')
                if len(parts) != 3:
                    raise ValueError('Unsupported upstream parameter hierarchy')
                key = (parts[0], parts[-1])
                if key in values:
                    raise ValueError('Ambiguous upstream parameter mapping')
                values[key] = value[()]
        with h5py.File(snapshot) as artifact:
            artifact.visititems(collect)
        model = build_model(weights=None)
        backbone = model.get_layer('efficientnetb4')
        assignments = []
        for layer in backbone.layers:
            if isinstance(layer, keras.layers.Normalization):
                if len(layer.weights) != 3:
                    raise ValueError('Unexpected normalization state')
                continue
            for variable in layer.weights:
                key = (layer.name, variable.name.split('/')[-1])
                if key not in values:
                    raise ValueError('Missing upstream parameter')
                value = values.pop(key)
                if (tuple(variable.shape) != value.shape
                        or np.dtype(variable.dtype.as_numpy_dtype) != value.dtype
                        or not np.isfinite(value).all()):
                    raise ValueError('Invalid upstream parameter shape, dtype or value')
                assignments.append((variable, value))
        if values or len(assignments) != 608:
            raise ValueError('Upstream parameter inventory mismatch')
        for variable, value in assignments:
            variable.assign(value)
    adapt_training_normalization(model, training_jpegs, batch_size=batch_size)
    return model
