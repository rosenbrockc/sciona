"""Synthetic RGB images verify resize and normalization boundaries."""
import numpy as np
import pytest
import torch

from sciona.aptos_models import FAMILIES
from sciona.aptos_preprocessing import prepare_rgb


@pytest.mark.parametrize('family', FAMILIES)
def test_channel_order_normalization_and_native_resolution(family):
    source = np.full((5, 9, 3), [0, 127, 255], dtype=np.uint8)
    before = source.copy()
    output = prepare_rgb(source, family)
    if family.startswith('inception_'):
        expected = [-1., (127 / 255 - .5) / .5, 1.]
    else:
        expected = [(0 - .485) / .229, (127 / 255 - .456) / .224, (1 - .406) / .225]
    assert output.shape == (3, FAMILIES[family][3], FAMILIES[family][3])
    assert output.dtype == torch.float32 and output.is_contiguous()
    for channel, value in enumerate(expected):
        torch.testing.assert_close(output[channel], torch.full_like(output[channel], value), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(source, before)


def test_entire_non_square_image_is_resized_without_center_crop():
    source = np.zeros((3, 11, 3), dtype=np.uint8)
    source[:, 0] = [255, 0, 0]
    source[:, -1] = [0, 255, 0]
    output = prepare_rgb(source, 'inception_v4')
    torch.testing.assert_close(output[:, 0, 0], torch.tensor([1., -1., -1.]))
    torch.testing.assert_close(output[:, -1, -1], torch.tensor([-1., 1., -1.]))


@pytest.mark.parametrize('image', [np.zeros((0, 2, 3), dtype=np.uint8),
    np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2, 4), dtype=np.uint8),
    np.zeros((2, 2, 3), dtype=np.float32)])
def test_invalid_image_contract_rejects(image):
    with pytest.raises(ValueError):
        prepare_rgb(image, 'inception_v4')
