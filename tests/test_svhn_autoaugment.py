import random

import numpy as np
from PIL import Image
import pytest

from sciona.svhn_autoaugment import SVHNAutoAugment


def test_reproducible_owned_stream_does_not_mutate_input_or_global_rng():
    image = Image.fromarray(np.random.default_rng(87).integers(0, 256, (137, 236, 3), dtype=np.uint8))
    original = np.asarray(image).copy()
    global_state = random.getstate()
    a, b = SVHNAutoAugment(seed=14), SVHNAutoAugment(seed=14)
    for _ in range(30):
        first, second = a(image), b(image)
        assert first.mode == 'RGB' and first.size == (236, 137)
        np.testing.assert_array_equal(first, second)
    assert random.getstate() == global_state
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('image', [None, Image.new('L', (10, 10)), Image.new('RGB', (0, 10))])
def test_invalid_boundary_does_not_consume_randomness(image):
    policy = SVHNAutoAugment(seed=87)
    state = policy.rng.getstate()
    with pytest.raises(ValueError):
        policy(image)
    assert policy.rng.getstate() == state
