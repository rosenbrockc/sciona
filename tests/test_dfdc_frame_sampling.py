"""Synthetic sampling expectations independent of the source reader."""

import numpy as np
import pytest

from sciona.dfdc_frame_sampling import select_frames


def clip(count):
    return np.broadcast_to(np.arange(count, dtype=np.uint8)[:, None, None, None],
                           (count, 2, 3, 3)).copy()


@pytest.mark.parametrize('count', [1, 2, 10, 31])
def test_short_clip_retains_first_frame_only(count):
    frames, indices = select_frames(clip(count))
    assert indices == [0]
    np.testing.assert_array_equal(frames, clip(count)[:1])


def test_exact_count_selects_every_frame_and_copies():
    source = clip(32)
    frames, indices = select_frames(source)
    assert indices == list(range(32))
    np.testing.assert_array_equal(frames, source)
    frames[:] = 255
    assert source[0, 0, 0, 0] == 0


def test_endpoint_inclusive_long_clip():
    frames, indices = select_frames(clip(64))
    expected = [(63 * k) // 31 for k in range(32)]
    assert indices == expected
    np.testing.assert_array_equal(frames[:, 0, 0, 0], expected)


def test_single_requested_frame_is_first_not_middle():
    _, indices = select_frames(clip(64), 1)
    assert indices == [0]


def test_empty_clip_matches_no_frames_result():
    assert select_frames(clip(0)) is None


@pytest.mark.parametrize('value', [0, -1, True, 2.5])
def test_invalid_request_rejected(value):
    with pytest.raises(ValueError):
        select_frames(clip(32), value)


@pytest.mark.parametrize('value', [
    np.zeros((2, 2, 3), dtype=np.uint8),
    np.zeros((2, 2, 2, 3), dtype=np.float32),
    np.zeros((2, 0, 2, 3), dtype=np.uint8),
])
def test_invalid_decoded_array_rejected(value):
    with pytest.raises(ValueError):
        select_frames(value)
