import numpy as np
import pytest

from sciona.dfdc_faces import detector_image, expanded_crop, extract_faces


def frame():
    return np.arange(24 * 30 * 3, dtype=np.uint16).reshape(24, 30, 3).astype(np.uint8)


def test_half_size_truncates_odd_dimensions():
    assert detector_image(np.zeros((25, 31, 3), dtype=np.uint8)).size == (15, 12)


def test_fractional_box_truncation_and_third_margin():
    image = frame()
    # doubled box (8,6,20,18); margins 4 in each dimension.
    np.testing.assert_array_equal(expanded_crop(image, [4.2, 3.4, 10.2, 9.4]), image[2:22, 4:24])


def test_borders_use_source_numpy_slicing():
    image = frame()
    np.testing.assert_array_equal(expanded_crop(image, [-1., -1., 20., 20.]), image)


def test_degenerate_box_remains_empty():
    assert expanded_crop(frame(), [4., 4., 4., 4.]).shape == (0, 0, 3)


def test_numeric_object_boxes_from_actual_mtcnn():
    box = np.array([4., 3., 10., 9.], dtype=object)
    np.testing.assert_array_equal(expanded_crop(frame(), box), frame()[2:22, 4:24])


def test_no_confidence_filter_or_alignment():
    class Detector:
        def detect(self, image, *, landmarks):
            assert landmarks is False
            return [[4., 3., 10., 9.], None], [.01, .99]
    result = extract_faces(frame()[None], Detector())
    assert len(result) == 1 and result[0]['frame_index'] == 0
    assert result[0]['scores'] == [.01]
    np.testing.assert_array_equal(result[0]['faces'][0], frame()[2:22, 4:24])


def test_none_boxes_omitted_but_empty_boxes_retained():
    class Detector:
        count = 0
        def detect(self, image, *, landmarks):
            self.count += 1
            return (None, None) if self.count == 1 else ([], [])
    result = extract_faces(np.stack([frame(), frame()]), Detector())
    assert result == [{'frame_index': 1, 'faces': [], 'scores': []}]


@pytest.mark.parametrize('box', [[1, 2, 3], [1, 2, 3, np.nan], ['1', '2', '3', '4']])
def test_invalid_box_rejected(box):
    with pytest.raises(ValueError):
        expanded_crop(frame(), box)


@pytest.mark.parametrize('value', [np.zeros((1, 4, 3), dtype=np.uint8),
                                 np.zeros((4, 4, 3), dtype=np.float32)])
def test_invalid_frame_rejected(value):
    with pytest.raises(ValueError):
        detector_image(value)
