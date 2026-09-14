import numpy as np
from sciona.dsb_world_preprocessing import world_training_annotations,prepare_world_training_volume
from sciona.dsb_training_preprocessing import prepare_training_masks


def test_world_origin_plane_is_not_empty_and_flip_uses_size_not_size_minus_one():
    bounds=np.array([[0,30],[0,30],[0,30]])
    actual=world_training_annotations([[12,24,30,8]],[30,20,10],[3,2,1],bounds,[10,20,30],flip=True)
    np.testing.assert_array_equal(actual,[[0,36,28,8]])
    assert world_training_annotations(np.empty((0,4)),[0,0,0],[1,1,1],bounds,[10,20,30]).shape==(0,4)


def test_world_volume_and_masks_flip_together_without_mutating_inputs():
    volume=np.arange(16*32*32,dtype=np.float64).reshape(16,32,32)%1800-1200
    left=np.zeros(volume.shape,dtype=bool);left[3:12,6:20,3:10]=True
    right=np.zeros_like(left);right[3:12,6:20,21:28]=True
    before=volume.copy()
    actual=prepare_world_training_volume(volume,left,right,[2,1,1],[0,0,0],[[7,12,12,6]],flip=True)
    expected=prepare_training_masks(volume[:,::-1,::-1],left[:,::-1,::-1],right[:,::-1,::-1],[2,1,1])
    for a,b in zip([actual[0],actual[2],actual[3]],expected):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(volume,before)
