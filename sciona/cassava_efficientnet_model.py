"""Winning B4 topology in the explicit legacy-Keras reference runtime."""


def build_model(*, weights, expected_sha256=None):
    """Build the full 512-pixel B4 classifier with frozen BatchNorm layers.

``weights`` must be passed explicitly. None is for untrained diagnostics only;
publication still requires verified NoisyStudent pretrained-weight provenance.
No decoder, population normalization adaptation or optimizer is supplied here.
"""
    from sciona.cassava_efficientnet_weights import verified_weights
    with verified_weights(weights, expected_sha256=expected_sha256) as verified:
        from tensorflow import keras
        backbone = keras.applications.EfficientNetB4(
            include_top=False, weights=verified, input_shape=(512, 512, 3), drop_connect_rate=.4)
    for layer in backbone.layers:
        layer.trainable = not isinstance(layer, keras.layers.BatchNormalization)
    inputs = keras.Input(shape=(512, 512, 3))
    features = backbone(inputs)
    pooled = keras.layers.GlobalAveragePooling2D()(features)
    dropped = keras.layers.Dropout(.5)(pooled)
    probabilities = keras.layers.Dense(5, activation='softmax')(dropped)
    return keras.Model(inputs, probabilities)
