"""Cassava winner inference views using an explicit TensorFlow reference runtime.

Inputs are decoded RGB pixels; decoding, model evaluation and training
augmentation are separate. The fixed canvas matches the source crop recipe.
"""
import cv2
import numpy as np


def _pixels(image):
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.shape != (600, 800, 3):
        raise ValueError('Decoded uint8 RGB image on a 600-by-800 canvas required')
    return array


def efficientnet_tiles(image):
    """Four overlapping corner crops and one resized central strip, unscaled."""
    pixels = _pixels(image).astype(np.float32)
    corners = [pixels[y:y + 512, x:x + 512] for y in (0, 88) for x in (0, 288)]
    central = cv2.resize(pixels[:, 100:700], (512, 512))
    return np.stack([*corners, central])


def _augment(tile):
    import tensorflow as tf
    spatial = tf.random.uniform([], 0, 1., dtype=tf.float32)
    rotation = tf.random.uniform([], 0, 1., dtype=tf.float32)
    tile = tf.image.random_flip_left_right(tile)
    tile = tf.image.random_flip_up_down(tile)
    if spatial > .75:
        tile = tf.image.transpose(tile)
    if rotation > .75:
        tile = tf.image.rot90(tile, k=3)
    elif rotation > .5:
        tile = tf.image.rot90(tile, k=2)
    elif rotation > .25:
        tile = tf.image.rot90(tile, k=1)
    return tf.reshape(tf.image.resize(tile, (512, 512)), (512, 512, 3))


def efficientnet_views(image):
    """Eight stochastic corner views then two identical unaugmented centers.

The caller owns TensorFlow RNG state. Corner order is repeated wholesale;
views are not grouped in adjacent pairs by corner. No external normalization.
"""
    import tensorflow as tf
    tiles = efficientnet_tiles(image)
    augmented = tf.map_fn(_augment, tf.concat([tiles[:4], tiles[:4]], axis=0))
    return tf.concat([augmented, np.broadcast_to(tiles[4], (2, 512, 512, 3))], axis=0)


def cropnet_view(image):
    """Scale to [0,1], central-crop by .8 in both axes, then resize to 224."""
    import tensorflow as tf
    scaled = (_pixels(image) / 255.).astype(np.float32)
    crop = tf.image.central_crop(scaled, .8)
    return tf.expand_dims(tf.image.resize(crop, (224, 224)), 0)
