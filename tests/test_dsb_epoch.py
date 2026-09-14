import torch
from sciona.dsb_training import shared_training_models,run_training_epoch


def test_first_epoch_reopens_factory_and_applies_variant_profiles():
    torch.set_num_threads(1);torch.manual_seed(17)
    detector,classifier=shared_training_models(2)
    opt1=torch.optim.SGD(detector.parameters(),lr=.01,momentum=.9)
    opt2=torch.optim.SGD(classifier.parameters(),lr=.01,momentum=.9)
    images=torch.randn(2,1,16,16,16);coords=torch.randn(2,3,4,4,4)
    labels=torch.zeros(2,4,4,4,3,5);labels[...,0]=-1;labels[:,1,1,1,0,0]=1
    calls=[]
    def factory(task,profile):
        calls.append((task,profile))
        if task=='detector':return [(images,labels,coords)]
        return [(images[None],coords[None],torch.ones(1,2),torch.ones(1))]*6
    before=detector.preBlock[1].running_mean.clone()
    report=run_training_epoch(detector,classifier,opt1,opt2,factory,epoch=121,start_epoch=121,
                               classifier_variant=4,freeze_batchnorm=True)
    assert [r['task'] for r in report]==['classifier','detector','classifier']
    assert [r['batches'] for r in report]==[5,1,6]
    assert report[0]['learning_rate']==0.
    assert calls[0][1]==dict(flip=True,swap=True,scale=True,rotate=True)
    assert calls[1][1]==dict(flip=True,swap=False,scale=True,rotate=False)
    torch.testing.assert_close(detector.preBlock[1].running_mean,before)
    assert all(r['mean_loss'] is not None for r in report)


def test_zero_learning_rate_pass_updates_state_without_changing_weights():
    torch.set_num_threads(1);torch.manual_seed(23)
    detector,classifier=shared_training_models(2)
    opt1=torch.optim.SGD(detector.parameters(),lr=.01,momentum=.9)
    opt2=torch.optim.SGD(classifier.parameters(),lr=.01,momentum=.9)
    before=classifier.fc2.weight.detach().clone()
    images=torch.randn(1,2,1,16,16,16);coords=torch.randn(1,2,3,4,4,4)
    def factory(task,profile):
        if task=='detector':return []
        return [(images,coords,torch.ones(1,2),torch.ones(1))]
    report=run_training_epoch(detector,classifier,opt1,opt2,factory,epoch=1,start_epoch=1)
    torch.testing.assert_close(classifier.fc2.weight,before)
    assert 'momentum_buffer' in opt2.state[classifier.fc2.weight]
    assert detector.preBlock[1].num_batches_tracked.item()==1
    assert report[0]['batches']==1
