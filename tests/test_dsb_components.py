import numpy as np
import pytest

from sciona.dsb_components import combine_volume, oversample_by_size, sample_proposals, split_volume, mask_and_remove_bone, process_lung_mask
from sciona.dsb_components import preprocess_from_masks, resample_volume
from sciona.dsb_components import segment_volume, preprocess_volume
from sciona.dsb_components import assign_anchor_labels
from sciona.dsb_components import decode_proposals, suppress_proposals
from sciona.dsb_components import prepare_detector_tiles
from sciona.dsb_components import crop_classifier_proposal


def test_classifier_crop_center_and_edge_padding_keep_input_unchanged():
    volume = np.arange(20**3, dtype=np.float32).reshape(1, 20, 20, 20)
    original = volume.copy()
    crop, coordinates = crop_classifier_proposal(volume, [10, 10, 10, 4], crop_size=8)
    np.testing.assert_array_equal(crop, volume[:, 6:14, 6:14, 6:14])
    assert coordinates.shape == (3, 2, 2, 2)
    edge, _ = crop_classifier_proposal(volume, [0, 0, 0, 4], crop_size=8)
    assert edge[0, 0, 0, 0] == 160
    assert edge[0, 4, 4, 4] == volume[0, 0, 0, 0]
    np.testing.assert_array_equal(volume, original)


@pytest.mark.parametrize('seed', [0, 9, 41])
def test_training_crop_is_seeded_and_fixed_size(seed):
    volume = np.zeros((1, 25, 25, 25), dtype=np.uint8)
    args = dict(crop_size=16, phase='train', scale_enabled=True, random_state=seed)
    first = crop_classifier_proposal(volume, [3, 12, 21, 10], **args)
    second = crop_classifier_proposal(volume, [3, 12, 21, 10], **args)
    assert first[0].shape == (1, 16, 16, 16)
    assert first[1].shape == (3, 4, 4, 4)
    for a, b in zip(first, second): np.testing.assert_array_equal(a, b)


def test_detector_tiles_use_global_coordinates_and_aligned_padding():
    volume = np.zeros((1, 19, 17, 21), dtype=np.uint8)
    tiles, coords, grid = prepare_detector_tiles(volume, side_len=16, max_stride=8, stride=4, margin=8)
    assert grid == (2, 2, 2)
    assert tiles.shape == (8, 1, 32, 32, 32)
    assert coords.shape == (8, 3, 8, 8, 8)
    # Neighboring tiles share the same global coordinates in overlap regions.
    np.testing.assert_array_equal(coords[0, :, :, :, 4:], coords[1, :, :, :, :4])
    assert coords[0, 2, 2, 2, 2] == -.5
    assert coords[1, 2, 2, 2, 2] > -.5
    assert np.any(tiles == (170.-128)/128)


def test_single_coordinate_sample_preserves_source_linspace_behavior():
    _, coords, grid = prepare_detector_tiles(np.zeros((1, 1, 1, 1)), 4, 4, 4, 0)
    assert grid == (1, 1, 1)
    np.testing.assert_array_equal(coords, np.full((1, 3, 1, 1, 1), -.5, dtype=np.float32))


def test_decoding_strict_threshold_geometry_and_input_retention():
    output = np.zeros((2, 2, 2, 3, 5), dtype=np.float32)
    output[..., 0] = -3
    output[1, 0, 1, 0] = [2, .2, -.1, .5, np.log(2)]
    before = output.copy()
    proposals, indices = decode_proposals(output)
    assert proposals.shape == (1, 5)
    np.testing.assert_allclose(proposals[0], [2, 7.5, .5, 10.5, 20])
    np.testing.assert_array_equal(np.array(indices)[:, 0], [1, 0, 1, 0])
    np.testing.assert_array_equal(output, before)


def test_suppression_inclusive_threshold_and_source_output_dtype():
    proposals = np.array([[3., 0., 0., 0., 10.], [2., 0., 0., 0., 10.], [1., 30., 30., 30., 10.]])
    result = suppress_proposals(proposals, 1.)
    np.testing.assert_array_equal(result, proposals[[0, 2]])
    assert result.dtype == np.float32
    assert suppress_proposals(proposals[:0]).shape == (0, 5)


def test_decode_rejects_selected_overflow_but_ignores_unselected_geometry():
    output = np.zeros((1, 1, 1, 3, 5), dtype=np.float32)
    output[..., 0] = -4
    output[..., 4] = 1000
    assert decode_proposals(output)[0].shape == (0, 5)
    output[0, 0, 0, 0, 0] = 0
    with pytest.raises(ValueError, match='geometry'):
        decode_proposals(output)


def test_positive_label_encoding_decodes_to_selected_target():
    target = np.array([17.5, 17.5, 17.5, 10.])
    labels = assign_anchor_labels((32, 32, 32), target, target[None], random_state=9)
    proposals, _ = decode_proposals(labels, threshold=.5)
    assert proposals.shape == (1, 5)
    np.testing.assert_allclose(proposals[0, 1:], target, rtol=1e-6)


def test_anchor_assignment_keeps_target_separate_and_samples_training_negatives():
    target = np.array([17.5, 17.5, 17.5, 10.])
    boxes = np.array([[5.5, 5.5, 5.5, 10.], target])
    train = assign_anchor_labels((32, 32, 32), target, boxes, num_neg=7, random_state=4)
    assert train.dtype == np.float32
    assert np.sum(train[..., 0] == 1) == 1
    assert np.sum(train[..., 0] == -1) == 7
    pos = np.argwhere(train[..., 0] == 1)[0]
    centers = 1.5 + 4 * pos[:3]
    anchor = np.array([10., 30., 60.])[pos[3]]
    np.testing.assert_allclose(centers + train[tuple(pos)][1:4] * anchor, target[:3])
    val = assign_anchor_labels((32, 32, 32), target, boxes, phase='val', num_neg=7, random_state=4)
    assert np.sum(val[..., 0] == -1) > 7


def test_negative_crop_has_no_positive_and_rng_is_local():
    import random
    before = random.getstate()
    labels = assign_anchor_labels((16, 16, 16), None, np.empty((0, 4)), num_neg=9, random_state=3)
    assert np.sum(labels[..., 0] == 1) == 0
    assert np.sum(labels[..., 0] == -1) == 9
    assert before == random.getstate()


def test_anchor_fallback_and_invalid_grid():
    labels = assign_anchor_labels((16, 16, 16), [2., 2., 2., 1.], np.empty((0, 4)), random_state=2)
    assert np.sum(labels[..., 0] == 1) == 1
    with pytest.raises(ValueError):
        assign_anchor_labels((15, 16, 16), None, np.empty((0, 4)))


def synthetic_two_components():
    z, y, x = np.indices((24, 64, 64))
    volume = np.full(z.shape, 50.)
    first = ((z-12)/9)**2 + ((y-32)/17)**2 + ((x-20)/9)**2 < 1
    second = ((z-12)/9)**2 + ((y-32)/17)**2 + ((x-44)/9)**2 < 1
    volume[first | second] = -900.
    return volume


def test_segmentation_returns_separate_nonempty_masks_and_connects_preprocessing():
    volume = synthetic_two_components()
    first, second = segment_volume(volume, [6., 6., 6.])
    assert first.any() and second.any() and not np.any(first & second)
    assert first.dtype == second.dtype == np.bool_
    # Use requested spacing equal to input spacing for a bounded synthetic run.
    result = preprocess_volume(volume, [6., 6., 6.], [6., 6., 6.])
    reference = preprocess_from_masks(volume, first, second, [6., 6., 6.], [6., 6., 6.])
    for actual, expected in zip(result, reference):
        np.testing.assert_array_equal(actual, expected)


def test_segmentation_empty_case_and_source_shape_contract():
    first, second = segment_volume(np.zeros((4, 16, 16)), [1., 1., 1.])
    assert not first.any() and not second.any()
    with pytest.raises(ValueError, match='square'):
        segment_volume(np.zeros((4, 16, 17)), [1., 1., 1.])


def test_preprocessing_composes_crop_and_retains_effective_spacing():
    volume = np.full((15, 21, 25), -600., dtype=float)
    mask = np.zeros(volume.shape, dtype=bool)
    mask[7, 10, 12] = True
    crop, spacing, bounds = preprocess_from_masks(volume, mask, np.zeros_like(mask), [1., 1., 1.])
    np.testing.assert_array_equal(bounds, [[2, 15], [5, 20], [7, 22]])
    np.testing.assert_array_equal(spacing, [1., 1., 1.])
    assert crop.shape == (13, 15, 15) and crop.dtype == np.uint8


def test_resampling_uses_rounded_shape_not_requested_spacing_as_effective():
    volume = np.zeros((3, 5, 7), dtype=np.uint8)
    result, actual = resample_volume(volume, [1.1, 1.1, 1.1])
    assert result.shape == (3, 6, 8)
    np.testing.assert_allclose(actual, np.array([3, 5, 7]) * 1.1 / [3, 6, 8])


@pytest.mark.parametrize('spacing', [[0., 1., 1.], [np.nan, 1., 1.], [0.01, 0.01, 0.01]])
def test_invalid_or_zero_rounded_grid_rejected(spacing):
    with pytest.raises(ValueError):
        resample_volume(np.zeros((2, 2, 2)), spacing)


def test_full_preprocessing_rejects_empty_segmentation():
    volume = np.zeros((2, 3, 4))
    mask = np.zeros(volume.shape, dtype=bool)
    with pytest.raises(ValueError, match='empty segmentation'):
        preprocess_from_masks(volume, mask, mask, [1., 1., 1.])


def test_bone_replacement_only_affects_dilated_boundary_and_keeps_inputs():
    volume = np.full((1, 31, 31), 600.)
    left = np.zeros(volume.shape, dtype=bool)
    right = left.copy()
    left[0, 15, 15] = True
    result = mask_and_remove_bone(volume, left, right)
    assert result.dtype == np.uint8
    assert result[0, 15, 15] == 255  # Original interior is not replaced.
    assert result[0, 15, 16] == 170  # Bone inside the dilated boundary.
    assert result[0, 0, 0] == 170  # Outside-mask padding.
    assert np.all(volume == 600.)
    assert left.sum() == 1 and not right.any()


def test_bone_threshold_is_strict_in_transformed_units():
    volume = np.full((1, 3, 3), -1200.)
    left = np.zeros(volume.shape, dtype=bool)
    left[0, 1, 1] = True
    volume[0, 1, 0] = 210.5 / 255. * 1800. - 1200.
    volume[0, 1, 2] = 211.5 / 255. * 1800. - 1200.
    volume[0, 0, 0] = np.nan
    result = mask_and_remove_bone(volume, left, np.zeros_like(left))
    assert result[0, 1, 0] == 210
    assert result[0, 1, 2] == 170
    assert result[0, 0, 0] == 0
    assert np.isnan(volume[0, 0, 0])


def test_empty_masks_produce_padding_and_reject_numeric_masks():
    volume = np.zeros((2, 3, 4))
    masks = np.zeros(volume.shape, dtype=bool)
    assert np.all(mask_and_remove_bone(volume, masks, masks) == 170)
    with pytest.raises(ValueError):
        process_lung_mask(masks.astype(float))


@pytest.mark.parametrize('shape,side,margin', [((2, 5, 7, 9), 4, 2), ((1, 1, 1, 1), 2, 0), ((1, 8, 4, 12), 4, 0)])
def test_split_identity_inference_reconstructs_edge_padded_volume(shape, side, margin):
    volume = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tiles, grid = split_volume(volume, side, 1, margin)
    # A synthetic detector emits input channels as anchors, with one output each.
    predictions = np.moveaxis(tiles, 1, -1)[..., None]
    actual = combine_volume(predictions, grid, side, 1, margin)
    expected = np.pad(volume, [(0, 0)] + [(0, g * side - n) for n, g in zip(shape[1:], grid)], mode='edge')
    np.testing.assert_array_equal(actual[..., 0], np.moveaxis(expected, 0, -1))


def test_sampling_extreme_scores_keeps_support_and_unique_indices():
    result = sample_proposals([0., -1000., -2000.], 2, random_state=3)
    assert len(result) == len(set(result)) == 2
    assert result[0] == 0
    assert np.all(np.isin(result, [0, 1, 2]))


def test_sampling_handles_finite_overflow_scale():
    result = sample_proposals([1e308, -1e308, 0.], 2, temperature=1e-300, random_state=1)
    assert result[0] == 0
    assert len(set(result)) == 2


def test_covering_all_proposals_preserves_order_and_empty_inputs_work():
    np.testing.assert_array_equal(sample_proposals([0., 4., 3.], 5), [0, 1, 2])
    assert sample_proposals([], 0).shape == (0,)
    assert sample_proposals([1., 2.], 0).shape == (0,)


def test_oversampling_threshold_boundaries_and_empty_selection():
    boxes = np.array([[i, size] for i, size in enumerate([6., 7., 30., 31., 40., 41.])])
    result = oversample_by_size(boxes, 1)
    np.testing.assert_array_equal(result[:, 0], np.repeat(np.arange(6), [0, 1, 1, 3, 3, 7]))
    assert oversample_by_size(boxes[:1], 1).shape == (0, 2)
    assert oversample_by_size(boxes[:0], 1).shape == (0, 2)


@pytest.mark.parametrize('call', [
    lambda: split_volume(np.zeros((1, 4, 4, 4)), 4, 3, 1),
    lambda: split_volume(np.zeros((1, 0, 4, 4)), 4, 1, 1),
    lambda: combine_volume(np.zeros((1, 4, 4, 4, 1, 1)), (2, 1, 1), 4, 1, 0),
    lambda: combine_volume(np.zeros((1, 4, 4, 4, 1, 1)), (1, 1, 1), 4, 3, 0),
    lambda: sample_proposals([np.nan], 1),
    lambda: sample_proposals([1.], -1),
    lambda: sample_proposals([1.], 1, temperature=0),
    lambda: oversample_by_size(np.array([[1., -1.]]), 1),
])
def test_invalid_contracts_rejected(call):
    with pytest.raises(ValueError):
        call()
