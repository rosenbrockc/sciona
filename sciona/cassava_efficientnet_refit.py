"""Final full-population B4 refit from the winner's described 14-epoch stage."""
from pathlib import Path

from sciona.cassava_efficientnet_initialization import initialize_population
from sciona.cassava_efficientnet_schedule import learning_rate
from sciona.cassava_efficientnet_training import _population, keras_focal_loss
from sciona.cassava_training_stream import epoch_steps, training_stream


def train_final(jpegs, labels, *, weights, batch_size, output_directory, expected_sha256=None, weight_format='keras'):
    """Refit a new model for exactly 14 epochs, retaining the 20-epoch LR clock.

The original full-data block is commented notebook code. Its normalization
input uses an unresolved scratch variable; this reconstruction explicitly
adapts on the supplied full training population. No validation/early stopping.
CPU replica count is one. None weights and small batches are diagnostics only.
"""
    import tensorflow as tf
    from tensorflow import keras
    _population(jpegs, labels)
    steps = epoch_steps(len(jpegs), batch_size)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    model = initialize_population(weights=weights, expected_sha256=expected_sha256,
        weight_format=weight_format, jpegs=jpegs, batch_size=batch_size)
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-6),
                  loss=keras_focal_loss, metrics=['categorical_accuracy'])
    records = tf.data.Dataset.from_tensor_slices((list(jpegs), list(labels)))
    history = model.fit(training_stream(records, batch_size=batch_size), epochs=14,
                        steps_per_epoch=steps, verbose=0,
                        callbacks=[keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate(epoch))])
    artifact = directory / 'final.h5'
    model.save(artifact)
    return model, history.history, artifact


def predict_images(model, images):
    """Average each image's ten source-defined views into five probabilities."""
    import numpy as np
    from sciona.cassava_image_views import efficientnet_views
    if not isinstance(images, (list, tuple)) or not images:
        raise ValueError('Nonempty decoded-image sequence required')
    results = []
    for image in images:
        probabilities = np.asarray(model(efficientnet_views(image), training=False))
        if (probabilities.shape != (10, 5) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(-1), 1., atol=1e-6)):
            raise ValueError('Model must return ten finite five-class probability rows')
        results.append(probabilities.mean(axis=0))
    return np.stack(results)
