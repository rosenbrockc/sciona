import numpy as np
from sciona.dsb_components import process_lung_mask
from sciona.dsb_training_preprocessing import training_lung_mask,training_voxel_annotations,prepare_training_masks


def test_training_hull_guard_differs_from_inference():
    mask=np.zeros((1,100,100),dtype=bool)
    mask[:,10:90,10:30]=True;mask[:,10:90,60:80]=True
    training=training_lung_mask(mask);inference=process_lung_mask(mask)
    assert training.sum()<inference.sum()
    assert np.all(training[mask])


def test_annotation_axis_order_requested_spacing_and_empty_sentinels():
    bounds=np.array([[2,40],[3,40],[4,40]])
    actual=training_voxel_annotations([[5,7,9,4]],[3,2,1],bounds,[2,1,1])
    np.testing.assert_array_equal(actual,[[11.5,11,1,8]])
    assert training_voxel_annotations(np.empty((0,4)),[3,2,1],bounds).shape==(0,4)
    assert training_voxel_annotations([[0,0,0,0]],[3,2,1],bounds).shape==(0,4)


def test_training_mask_preparation_preserves_inputs_and_returns_bounded_crop():
    volume=np.full((24,32,32),50.,dtype=np.float32)
    volume[0,0,0]=np.nan
    left=np.zeros(volume.shape,dtype=bool);left[5:19,8:24,4:13]=True
    right=np.zeros_like(left);right[5:19,8:24,19:28]=True
    before=volume.copy()
    result,spacing,bounds=prepare_training_masks(volume,left,right,[2,1,1])
    assert result.dtype==np.uint8 and result.shape[0]==1
    np.testing.assert_array_equal(result.shape[1:],bounds[:,1]-bounds[:,0])
    np.testing.assert_array_equal(volume,before)
