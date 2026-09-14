import random
import numpy as np
from sciona.dsb_intake import annotate_classifier_proposals,classifier_validation_sample,detector_validation_sample


def test_annotation_filters_before_nms_and_keeps_known_labels_aligned():
    proposals=np.array([[-1.,0,0,0,8],[1.,20,20,20,8],[3.,20,20,20,8],[2.,40,40,40,8]])
    retained,known=annotate_classifier_proposals(proposals,np.array([[20.,20,20,8]]))
    np.testing.assert_array_equal(retained[:,0],[3,2])
    np.testing.assert_array_equal(known,[True,False])
    # IoU equal to the detection threshold is not a positive label.
    _,known=annotate_classifier_proposals(proposals,np.array([[20.,20,20,8]]),detection_threshold=1.)
    assert not known.any()


def test_validation_classifier_orders_labels_and_retains_empty_slots():
    volume=np.full((1,48,48,48),100,dtype=np.uint8)
    proposals=np.array([[1.,24,24,24,8],[3.,16,16,16,8]])
    images,coords,known,label=classifier_validation_sample(volume,proposals,[0,1],1,topk=3,crop_size=16)
    np.testing.assert_array_equal(known,[1,0,0])
    assert known.dtype==np.int32
    assert not images[2].any() and not coords[2].any()
    np.testing.assert_array_equal(label,[1])


def test_validation_detector_keeps_all_negatives_and_raw_intensities():
    volume=np.full((1,64,64,64),200,dtype=np.uint8)
    box=np.array([32.,32,32,10])
    images,labels,coords=detector_validation_sample(volume,box,box[None],crop_size=32,bound_size=4,
        rng=np.random.RandomState(5),label_rng=random.Random(5))
    assert np.all(images==200) and images.dtype==np.float32
    assert np.count_nonzero(labels[...,0]==-1)>800
    assert np.count_nonzero(labels[...,0]==1)==1
    assert coords.shape==(3,8,8,8)
