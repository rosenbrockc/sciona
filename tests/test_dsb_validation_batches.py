import random
import numpy as np
import pytest
from sciona.dsb_validation_batches import ValidationBatchFactory


def test_validation_factory_keeps_source_repeats_and_distinct_case_collections():
    volume=np.full((1,64,64,64),100,dtype=np.uint8)
    boxes=np.array([[24.,24,24,10],[32.,32,32,31]])
    proposals=np.column_stack(([1.,3.],boxes))
    heldout=dict(volumes=[volume]*3,proposals=[proposals]*3,boxes=[boxes]*3,labels=[0,1,0])
    training={**heldout,'labels':[1,0,1]}
    factory=ValidationBatchFactory(dict(detector=dict(volumes=[volume],boxes=[boxes]),
        classifier_validation=heldout,classifier_training_validation=training),
        detector_batch_size=3,classifier_batch_size=2,topk=3,detector_crop_size=32,
        classifier_crop_size=16,numpy_rng=np.random.RandomState(9),label_rng=random.Random(9))
    det=list(factory('detector'))
    assert [len(b[0]) for b in det]==[3,1]  # 1 + (1+2) repeats; no random extras.
    held=list(factory('classifier_validation'));train=list(factory('classifier_training_validation'))
    assert [len(b[0]) for b in held]==[2,1]
    np.testing.assert_array_equal(np.concatenate([b[3] for b in held]).ravel(),[0,1,0])
    np.testing.assert_array_equal(np.concatenate([b[3] for b in train]).ravel(),[1,0,1])
    for batch in held:
        np.testing.assert_array_equal(batch[2],np.tile([1,1,0],(len(batch[0]),1)))
        assert not batch[0][:,2].any()
    again=list(factory('classifier_validation'))
    for first,second in zip(held,again):
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError):list(factory('unknown'))
