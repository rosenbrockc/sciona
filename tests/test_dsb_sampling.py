import numpy as np
import pytest
from sciona.dsb_sampling import detector_sampling_table,detector_epoch_length,select_detector_sample
from sciona.dsb_training import detector_indexed_training_sample


def test_sampling_table_preserves_image_association_and_threshold_counts():
    boxes=[np.array([[1.,2.,3.,6.],[1.,2.,3.,7.]]),np.array([[4.,5.,6.,41.]])]
    table=detector_sampling_table(boxes)
    np.testing.assert_array_equal(table[:,0],[0]+[1]*7)
    assert detector_epoch_length(table,.3)==11
    with pytest.raises(ValueError):detector_epoch_length(np.empty((0,5)))


def test_random_subset_maps_to_actual_image_boxes():
    table=detector_sampling_table([np.array([[24.,24.,24.,10.]])]*3)
    # Seed 1 selects independent-image branch. Filtered position 0 means image 2.
    selected=select_detector_sample(3,table,[2],image_count=3,rng=np.random.RandomState(1))
    assert selected['independent_image'] and selected['image_index']==2 and selected['target'] is None
    volumes=[np.full((1,48,48,48),i*50.,dtype=np.float32) for i in range(3)]
    boxes=[np.empty((0,4)),np.empty((0,4)),np.array([[24.,24.,24.,10.]])]
    sample,labels,coord=detector_indexed_training_sample(3,volumes,boxes,table,[2],
        rng=np.random.RandomState(1),crop_size=32,bound_size=4,label_seed=3,num_neg=7)
    assert np.any(sample==100.) and not np.any(sample==0.)
    assert labels.shape==(8,8,8,3,5) and coord.shape==(3,8,8,8)
