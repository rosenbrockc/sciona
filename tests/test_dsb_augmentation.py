import numpy as np

from sciona.dsb_augmentation import augment_detection, augment_classifier
from sciona.dsb_detector_crop import crop_detection


def test_detector_crop_translates_target_and_boxes_together():
    volume=np.zeros((1,48,48,48),dtype=np.uint8)
    target=np.array([24.,24.,24.,10.]);boxes=np.array([target,[20.,21.,22.,4.]])
    crop,actual_target,actual_boxes,coord=crop_detection(volume,target,boxes,crop_size=32,bound_size=4,rng=np.random.RandomState(3))
    assert crop.shape==(1,32,32,32) and coord.shape==(3,8,8,8)
    np.testing.assert_array_equal(actual_target,actual_boxes[0])
    np.testing.assert_array_equal(actual_boxes[1,:3]-actual_target[:3],boxes[1,:3]-target[:3])


def test_random_detector_crop_marks_target_absent():
    result=crop_detection(np.zeros((1,48,48,48)),[24.,24.,24.,10.],np.empty((0,4)),
                          crop_size=32,bound_size=4,random_crop=True,rng=np.random.RandomState(4))
    assert np.isnan(result[1]).all()
    assert result[0].shape==(1,32,32,32)


def test_scaled_detector_crop_preserves_size_and_label_ratios():
    target=np.array([24.,24.,24.,10.]);boxes=target[None].copy()
    result=crop_detection(np.zeros((1,48,48,48)),target,boxes,crop_size=32,bound_size=4,
                          scale=True,rng=np.random.RandomState(9))
    assert result[0].shape==(1,32,32,32)
    np.testing.assert_array_equal(result[1],result[2][0])


def arrays():
    return np.arange(16**3, dtype=float).reshape(1,16,16,16), np.stack(np.meshgrid(*([np.linspace(-.5,.5,4)]*3), indexing='ij'))


def test_augmentations_do_not_mutate_callers():
    sample, coord = arrays()
    target = np.array([8.,8.,8.,2.]); boxes = target[None].copy()
    before = [a.copy() for a in [sample,coord,target,boxes]]
    result = augment_detection(sample,target,boxes,coord,rng=np.random.RandomState(4))
    assert result[0].shape == sample.shape and result[3].shape == coord.shape
    for a,b in zip([sample,coord,target,boxes],before): np.testing.assert_array_equal(a,b)


def test_classifier_source_rotation_leaves_coordinate_grid_unrotated():
    sample, coord = arrays()
    actual, actual_coord = augment_classifier(sample,coord,flip=False,swap=False,rotate=True,rng=np.random.RandomState(2))
    assert not np.array_equal(actual,sample)
    np.testing.assert_array_equal(actual_coord,coord)


def test_rejected_detector_rotation_preserves_edge_target_and_arrays():
    sample, coord = arrays()
    target = np.array([0.,0.,0.,3.])
    actual = augment_detection(sample,target,target[None],coord,flip=False,swap=False,rng=np.random.RandomState(9))
    np.testing.assert_array_equal(actual[0],sample)
    np.testing.assert_array_equal(actual[1],target)
