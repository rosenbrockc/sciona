"""Source-order EfficientNet streams from decoded-record-independent inputs."""


def _batch_size(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('Positive integer batch size required')


def prepare_record(encoded, label):
    import tensorflow as tf
    from sciona.cassava_training_images import decode_training_jpeg
    tf.debugging.assert_rank(label, 0)
    if label.dtype not in (tf.int32, tf.int64):
        raise ValueError('Integer class index required')
    tf.debugging.assert_greater_equal(label, tf.cast(0, label.dtype))
    tf.debugging.assert_less(label, tf.cast(5, label.dtype))
    return decode_training_jpeg(encoded), tf.one_hot(tf.cast(label, tf.int32), 5)


def _augment_pair(image, label):
    from sciona.cassava_training_images import augment_training_image
    return augment_training_image(image), label


def training_stream(records, *, batch_size):
    """Decode, repeat, augment, shuffle 1000, batch, prefetch, in that order.

The buffer intentionally contains augmented pixels. Changing it to encoded
records changes random-operation ordering and is not source-equivalent.
The lifecycle must bound consumption using floor(population/batch_size) steps.
"""
    import tensorflow as tf
    _batch_size(batch_size)
    return (records.map(prepare_record).repeat().map(_augment_pair)
            .shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE))


def validation_stream(records, *, batch_size):
    """Finite, ordered decoded stream; include the final partial batch."""
    import tensorflow as tf
    _batch_size(batch_size)
    return records.map(prepare_record).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def epoch_steps(population_size, batch_size):
    """Source floor division; reject empty epochs before training starts."""
    _batch_size(batch_size)
    if isinstance(population_size, bool) or not isinstance(population_size, int) or population_size < batch_size:
        raise ValueError('Training population must contain at least one full batch')
    return population_size // batch_size
