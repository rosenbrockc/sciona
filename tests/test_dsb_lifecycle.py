import io
import pytest
import torch
from sciona.dsb_training import shared_training_models
from sciona.dsb_checkpoint import checkpoint_bytes,restore_checkpoint
from sciona.dsb_validation import validate_detector,validate_classifier
from sciona.dsb_training import run_epoch_lifecycle


def test_initial_alternating_lifecycle_modes_and_checkpoint_state():
    torch.set_num_threads(1);torch.manual_seed(19)
    detector,classifier=shared_training_models(2)
    optimizers=[torch.optim.SGD(m.parameters(),lr=.01,momentum=.9) for m in [detector,classifier]]
    images=torch.randn(2,1,16,16,16);coords=torch.randn(2,3,4,4,4)
    labels=torch.zeros(2,4,4,4,3,5);labels[...,0]=-1
    labels[:,1,1,1,0,0]=1
    detector_batch=(images,labels,coords)
    classifier_batch=(images[None],coords[None],torch.ones(1,2),torch.ones(1))
    events=[]
    def training(task,profile):
        events.append('train_'+task)
        model=detector if task=='detector' else classifier
        assert model.training and detector.preBlock[1].training
        return [detector_batch] if task=='detector' else [classifier_batch]*6
    def validation(task):
        # Check at iteration time: validation sets eval after creating the loader.
        events.append(task)
        model=detector if task=='detector' else classifier
        assert not model.training and not detector.preBlock[1].training
        yield detector_batch if task=='detector' else classifier_batch
    result=run_epoch_lifecycle(detector,classifier,*optimizers,training,validation,
        epoch=30,start_epoch=30,save_frequency=5)
    assert events==['train_classifier','train_detector','detector','train_classifier',
                    'classifier_validation','classifier_training_validation']
    assert [entry['batches'] for entry in result['training']]==[5,1,6]
    # Only training passes update shared BatchNorm counters; validation is inert.
    assert detector.preBlock[1].num_batches_tracked.item()==12
    _,restored=shared_training_models(2)
    assert restore_checkpoint(restored,result['checkpoint'])==31
    for key,value in classifier.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)


def test_epoch_validation_order_and_checkpoint_save_cadence():
    torch.set_num_threads(1);torch.manual_seed(5)
    detector,classifier=shared_training_models(2)
    optimizers=[torch.optim.SGD(m.parameters(),lr=.001,momentum=.9) for m in [detector,classifier]]
    batch=(torch.randn(1,2,1,16,16,16),torch.randn(1,2,3,4,4,4),torch.ones(1,2),torch.ones(1,1))
    calls=[]
    def validation(key):calls.append(key);return [batch]
    result=run_epoch_lifecycle(detector,classifier,*optimizers,lambda task,profile:[batch],validation,
        epoch=160,start_epoch=121,classifier_variant=4,freeze_batchnorm=True,save_frequency=5)
    assert calls==['classifier_validation','classifier_training_validation']
    assert not classifier.training
    assert isinstance(result['checkpoint'],bytes)
    assert restore_checkpoint(classifier,result['checkpoint'])==161
    result=run_epoch_lifecycle(detector,classifier,*optimizers,lambda task,profile:[batch],validation,
        epoch=161,start_epoch=121,classifier_variant=4,freeze_batchnorm=True,save_frequency=5)
    assert result['checkpoint'] is None


def test_checkpoint_preserves_shared_network_and_resume_epoch():
    torch.set_num_threads(1);torch.manual_seed(4)
    detector,classifier=shared_training_models(2)
    blob=checkpoint_bytes(classifier,30)
    second,restored=shared_training_models(2)
    assert restore_checkpoint(restored,blob)==31
    assert restore_checkpoint(restored,blob,121)==121
    assert restored.NoduleNet is second
    for key,value in classifier.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)


def test_bad_checkpoint_rejected_before_any_weight_mutation():
    torch.set_num_threads(1)
    _,model=shared_training_models(2)
    before=model.baseline.detach().clone()
    data=torch.load(io.BytesIO(checkpoint_bytes(model,1)),weights_only=True)
    data['state_dict']['baseline'].fill_(2.)
    data['state_dict']['fc2.bias']=torch.zeros(9)
    stream=io.BytesIO();torch.save(data,stream)
    with pytest.raises(ValueError):restore_checkpoint(model,stream.getvalue())
    torch.testing.assert_close(model.baseline,before)


def test_validation_is_aggregate_only_and_preserves_model_buffers():
    torch.set_num_threads(1);torch.manual_seed(8)
    detector,classifier=shared_training_models(2)
    before={key:value.clone() for key,value in classifier.state_dict().items()}
    images=torch.randn(2,1,16,16,16);coords=torch.randn(2,3,4,4,4)
    labels=torch.zeros(2,4,4,4,3,5);labels[...,0]=-1
    detection=validate_detector(detector,[(images,labels,coords)])
    assert detection['true_positive_rate'] is None
    assert detection['negative_count']==2*4**3*3
    cases=validate_classifier(classifier,[(images[None],coords[None],torch.ones(1,2),torch.ones(1,1))])
    assert cases['case_count']==1
    for key,value in classifier.state_dict().items():torch.testing.assert_close(value,before[key])
    assert all(p.grad is None for p in classifier.parameters())
