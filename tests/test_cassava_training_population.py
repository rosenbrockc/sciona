"""Regression for tuple populations passed by the full pipeline boundary."""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')

from sciona.cassava_training_images import adapt_training_normalization


def test_tuple_and_list_populations_adapt_identically():
    def model():
        normalization = tf.keras.layers.Normalization(name='normalization')
        inputs = tf.keras.Input((512, 512, 3))
        backbone = tf.keras.Model(inputs, normalization(inputs), name='efficientnetb4')
        outer = tf.keras.Input((512, 512, 3))
        return tf.keras.Model(outer, backbone(outer)), normalization
    rng = np.random.default_rng(933)
    population = [tf.image.encode_jpeg(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)).numpy()
                  for _ in range(2)]
    as_list, list_normalization = model()
    as_tuple, tuple_normalization = model()
    tf.random.set_seed(17)
    adapt_training_normalization(as_list, population, batch_size=1)
    tf.random.set_seed(17)
    adapt_training_normalization(as_tuple, tuple(population), batch_size=1)
    for expected, actual in zip(list_normalization.get_weights(), tuple_normalization.get_weights(), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert len(population) == 2
