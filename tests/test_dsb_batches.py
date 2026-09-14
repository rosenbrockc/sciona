import random
import numpy as np
import pytest
import torch

from sciona.dsb_batches import TrainingBatchFactory
from sciona.dsb_training import (detector_indexed_training_sample,classifier_training_sample,
                                 shared_training_models,run_epoch_lifecycle)


def factory():
    volume=np.full((1,48,48,48),100,dtype=np.uint8)
    boxes=np.array([[24.,24.,24.,10.],[18.,18.,18.,12.],[30.,30.,30.,14.]])
    proposals=np.column_stack(([3.,2.,1.],boxes))
    return TrainingBatchFactory([volume],[boxes],[0],[volume]*3,[proposals]*3,
        [np.array([1,0,1])]*3,[1,0,1],detector_order=[3,2,0,1],classifier_order=[2,0,1],
        detector_batch_size=3,classifier_batch_size=2,topk=2,detector_crop_size=32,
        classifier_crop_size=16,numpy_rng=np.random.RandomState(7),label_rng=random.Random(9))


def test_factory_preserves_sample_order_partial_batches_and_rng_continuation():
    actual=factory();reference=factory()
    profile=dict(flip=True,rotate=False,swap=False,scale=False)
    for task in ['classifier','detector','classifier']:
        batches=list(actual(task,profile))
        expected=[]
        for index in reference.orders[task]:
            if task=='detector':
                sample=detector_indexed_training_sample(index,reference.detector_volumes,
                    reference.boxes_by_image,reference.table,reference.eligible,rng=reference.rng,
                    label_rng=reference.label_rng,crop_size=32,**profile)
            else:
                images,coords,known,_=classifier_training_sample(reference.classifier_volumes[index],
                    reference.proposals[index],reference.known[index],topk=2,crop_size=16,
                    rng=reference.rng,**profile)
                sample=(images,coords,known,reference.labels[index:index+1])
            expected.append(sample)
        assert [len(b[0]) for b in batches]==([3,1] if task=='detector' else [2,1])
        for component in range(len(expected[0])):
            np.testing.assert_array_equal(np.concatenate([b[component] for b in batches]),
                                          np.stack([s[component] for s in expected]))
    np.testing.assert_array_equal(actual.rng.rand(10),reference.rng.rand(10))
    assert actual.label_rng.getstate()==reference.label_rng.getstate()


def test_runtime_arrays_drive_complete_alternating_epoch():
    torch.set_num_threads(1);torch.manual_seed(31)
    training=factory();validation=factory()
    profile=dict(flip=False,rotate=False,swap=False,scale=False)
    det_batches=list(validation('detector',profile));case_batches=list(validation('classifier',profile))
    detector,classifier=shared_training_models(2)
    opts=[torch.optim.SGD(m.parameters(),lr=.01,momentum=.9) for m in [detector,classifier]]
    before=classifier.fc2.weight.detach().clone()
    result=run_epoch_lifecycle(detector,classifier,*opts,training,
        lambda task:det_batches if task=='detector' else case_batches,
        epoch=121,start_epoch=121,classifier_variant=4,freeze_batchnorm=True)
    assert [p['batches'] for p in result['training']]==[2,2,2]
    assert len(result['validation'])==3 and isinstance(result['checkpoint'],bytes)
    assert not torch.equal(before,classifier.fc2.weight)
    assert all(np.isfinite(p['mean_loss']) for p in result['training'])


def test_epoch_order_cannot_drop_or_duplicate_samples():
    from sciona.dsb_batches import _order
    for values in [[0,0,2],[0,1],[-1,1,2],[0,1,3]]:
        with pytest.raises(ValueError):_order(values,3,'order')
