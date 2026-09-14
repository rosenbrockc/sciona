"""EfficientNet fold lifecycle for the explicit CPU legacy-Keras reconstruction."""
from pathlib import Path

from sciona.cassava_efficientnet_initialization import initialize_population
from sciona.cassava_efficientnet_schedule import learning_rate
from sciona.cassava_training_stream import epoch_steps, training_stream, validation_stream


def keras_focal_loss(targets, probabilities):
    """Source softmax-modulated focal loss with legacy Keras BCE dispatch."""
    import tensorflow as tf
    from tensorflow import keras
    probabilities = tf.convert_to_tensor(probabilities)
    labels = tf.cast(targets * .9 + .02, probabilities.dtype)
    ce = keras.backend.binary_crossentropy(labels, probabilities, from_logits=False)
    p_target = labels * probabilities + (1 - labels) * (1 - probabilities)
    balancing = labels * .25 + (1 - labels) * .75
    return tf.reduce_sum(balancing * tf.square(1 - p_target) * ce, axis=-1)


def _population(jpegs, labels):
    if (not isinstance(jpegs, (list, tuple)) or not jpegs
            or not all(isinstance(x, bytes) and x for x in jpegs)
            or not isinstance(labels, (list, tuple)) or len(labels) != len(jpegs)
            or not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 5 for x in labels)):
        raise ValueError('Nonempty aligned encoded images and integer five-class labels required')


def fold_callbacks(checkpoint):
    from tensorflow import keras
    return [keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode='max', patience=4, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_categorical_accuracy', save_best_only=True),
            keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate(epoch))]


def train_fold(training_jpegs, training_labels, validation_jpegs, validation_labels,
               *, weights, batch_size, checkpoint_directory, expected_sha256=None, weight_format='keras'):
    """Train for up to 20 epochs with the source callback order and selection.

Caller supplies disjoint fold populations and a new private checkpoint
directory. No external population provenance is inferred here. Explicit
weights=None and reduced batch sizes support diagnostics only; they do not
qualify pretrained or original-TPU training. The CPU replica count is one.
"""
    import tensorflow as tf
    from tensorflow import keras
    _population(training_jpegs, training_labels)
    _population(validation_jpegs, validation_labels)
    steps = epoch_steps(len(training_jpegs), batch_size)
    directory = Path(checkpoint_directory)
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory / 'best.h5'
    model = initialize_population(weights=weights, expected_sha256=expected_sha256,
        weight_format=weight_format, jpegs=training_jpegs, batch_size=batch_size)
    training = tf.data.Dataset.from_tensor_slices((list(training_jpegs), list(training_labels)))
    validation = tf.data.Dataset.from_tensor_slices((list(validation_jpegs), list(validation_labels)))
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-6),
                  loss=keras_focal_loss, metrics=['categorical_accuracy'])
    history = model.fit(training_stream(training, batch_size=batch_size), epochs=20,
                        steps_per_epoch=steps,
                        validation_data=validation_stream(validation, batch_size=batch_size),
                        callbacks=fold_callbacks(checkpoint), verbose=0)
    return model, history.history, checkpoint
