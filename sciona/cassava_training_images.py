"""EfficientNet training pixel path in the reference TensorFlow runtime."""


def decode_training_jpeg(encoded):
    """Decode an already-sized JPEG into source bfloat16 RGB training pixels."""
    import tensorflow as tf
    pixels = tf.image.decode_jpeg(encoded, channels=3)
    tf.debugging.assert_equal(tf.shape(pixels), [512, 512, 3])
    return tf.cast(pixels, tf.bfloat16)


def augment_training_image(image):
    """Keep source RNG draw order, bfloat16 color operations and final resize."""
    import tensorflow as tf
    image = tf.convert_to_tensor(image)
    if image.dtype != tf.bfloat16:
        raise ValueError('Source training augmentation requires bfloat16 pixels')
    tf.debugging.assert_equal(tf.shape(image), [512, 512, 3])
    tf.debugging.assert_all_finite(tf.cast(image, tf.float32), 'Nonfinite image')
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if tf.random.uniform([], 0., 1.) > .75:
        image = tf.image.transpose(image)
    rotation = tf.random.uniform([], 0., 1.)
    if rotation > .75:
        image = tf.image.rot90(image, 3)
    elif rotation > .5:
        image = tf.image.rot90(image, 2)
    elif rotation > .25:
        image = tf.image.rot90(image, 1)
    if tf.random.uniform([], 0., 1.) >= .4:
        image = tf.image.random_saturation(image, .8, 1.2)
    if tf.random.uniform([], 0., 1.) >= .4:
        image = tf.image.random_contrast(image, .8, 1.2)
    if tf.random.uniform([], 0., 1.) >= .4:
        image = tf.image.random_brightness(image, .1)
    crop = tf.random.uniform([], 0., 1.)
    if crop > .7:
        if crop > .9:
            image = tf.image.central_crop(image, .7)
        elif crop > .8:
            image = tf.image.central_crop(image, .8)
        else:
            image = tf.image.central_crop(image, .9)
    elif crop > .5:
        size = tf.random.uniform([], int(512 * .8), 512, dtype=tf.int32)
        image = tf.image.random_crop(image, [size, size, 3])
    return tf.reshape(tf.image.resize(image, (512, 512)), (512, 512, 3))


def normalization_pixels(decoded):
    """Scale the unaugmented decoded stream before backbone normalization adapt.

Division remains bfloat16 as in the source; casting to float32 first changes
the statistics. Population selection and batching belong to the lifecycle.
"""
    import tensorflow as tf
    decoded = tf.convert_to_tensor(decoded)
    if decoded.dtype != tf.bfloat16:
        raise ValueError('Decoded bfloat16 pixels required')
    tf.debugging.assert_equal(tf.shape(decoded), [512, 512, 3])
    return decoded / 255.


def adapt_training_normalization(model, training_jpegs, *, batch_size):
    """Adapt only from the caller's explicit training population, without augmentation."""
    import tensorflow as tf
    if (not isinstance(training_jpegs, (list, tuple)) or not training_jpegs
            or not all(isinstance(x, bytes) and x for x in training_jpegs)
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError('Nonempty encoded training population and positive batch size required')
    population = tf.data.Dataset.from_tensor_slices(list(training_jpegs))
    population = population.map(lambda value: normalization_pixels(decode_training_jpeg(value)))
    population = population.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    model.get_layer('efficientnetb4').get_layer('normalization').adapt(population)
