import numpy as np
import torch

from sciona.dsb_training import detector_training_sample, classifier_training_sample
from sciona.dsb_network import DetectorNet, CaseNet
from sciona.dsb_losses import detector_loss, classifier_loss
from sciona.dsb_training import shared_training_models


def test_alternating_detector_loader_retains_raw_intensities():
    sample,_,_=detector_training_sample(np.full((1,48,48,48),200,dtype=np.uint8),
        [24.,24.,24.,10.],np.empty((0,4)),crop_size=16,bound_size=4,
        scale=False,flip=False,rotate=False,swap=False,rng=np.random.RandomState(2),label_seed=3)
    assert np.all(sample==200.) and sample.dtype==np.float32


def test_absent_target_random_image_crop_and_shared_label_rng():
    import random
    volume=np.zeros((1,48,48,48),dtype=np.uint8)
    rng=random.Random(7)
    before=rng.getstate()
    sample,labels,coord=detector_training_sample(volume,None,np.empty((0,4)),crop_size=32,
        bound_size=4,scale=False,random_crop=True,num_neg=7,rng=np.random.RandomState(3),label_rng=rng)
    assert np.sum(labels[...,0]==1)==0 and np.sum(labels[...,0]==-1)==7
    assert rng.getstate()!=before
    assert sample.shape==(1,32,32,32) and coord.shape==(3,8,8,8)


def test_alternating_updates_share_features_and_keep_separate_momentum_states():
    torch.set_num_threads(1);torch.manual_seed(41)
    detector,case=shared_training_models(2)
    assert case.NoduleNet is detector
    first=torch.optim.SGD(detector.parameters(),lr=.001,momentum=.9)
    second=torch.optim.SGD(case.parameters(),lr=.001,momentum=.9)
    images=torch.randn(2,1,16,16,16);coordinates=torch.randn(2,3,4,4,4)
    first.zero_grad()
    _,predicted=detector(images,coordinates)
    labels=torch.zeros_like(predicted);labels[...,0]=-1;labels[:,1,1,1,0,0]=1
    detector_loss(predicted,labels)['total'].backward();first.step()
    weight=detector.preBlock[0].weight
    after_detector=weight.detach().clone()
    first_momentum=first.state[weight]['momentum_buffer'].clone()
    second.zero_grad()
    _,probability,each=case(images[None],coordinates[None])
    classifier_loss(probability,each,torch.ones(1),torch.ones_like(each))['total'].backward();second.step()
    assert not torch.equal(weight,after_detector)
    torch.testing.assert_close(first.state[weight]['momentum_buffer'],first_momentum)
    assert first.state[weight]['momentum_buffer'].data_ptr()!=second.state[weight]['momentum_buffer'].data_ptr()


def test_detector_assembled_samples_drive_optimizer_update():
    torch.set_num_threads(1); torch.manual_seed(8)
    volume=np.zeros((1,48,48,48),dtype=np.uint8)
    target=np.array([24.,24.,24.,10.]); boxes=target[None]
    samples=[detector_training_sample(volume,target,boxes,crop_size=32,bound_size=4,
              rng=np.random.RandomState(seed),label_seed=seed,num_neg=20) for seed in [3,7]]
    model=DetectorNet().train(); optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    before=model.preBlock[0].weight.detach().clone()
    x,labels,coords=[torch.from_numpy(np.stack([sample[i] for sample in samples])) for i in range(3)]
    optimizer.zero_grad()
    loss=detector_loss(model(x,coords),labels)['total']
    assert torch.isfinite(loss)
    loss.backward();optimizer.step()
    assert not torch.equal(before,model.preBlock[0].weight)


def test_classifier_shared_rng_pipeline_drives_optimizer_update():
    torch.set_num_threads(1);torch.manual_seed(9)
    volume=np.full((1,40,40,40),100,dtype=np.uint8)
    proposals=np.array([[3.,20.,20.,20.,8.],[2.,12.,14.,16.,10.],[1.,30.,30.,30.,6.]])
    rng=np.random.RandomState(5)
    crops,coords,known,chosen=classifier_training_sample(volume,proposals,[1,0,1],topk=2,crop_size=16,rng=rng)
    repeat=classifier_training_sample(volume,proposals,[1,0,1],topk=2,crop_size=16,rng=np.random.RandomState(5))
    for a,b in zip((crops,coords,known,chosen),repeat):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(known,np.array([1,0,1])[chosen])
    model=CaseNet(2).train();optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    before=model.fc2.weight.detach().clone()
    optimizer.zero_grad()
    _,case,each=model(torch.from_numpy(crops[None]),torch.from_numpy(coords[None]))
    loss=classifier_loss(case,each,torch.ones(1),torch.from_numpy(known[None]))['total']
    assert torch.isfinite(loss)
    loss.backward();optimizer.step()
    assert not torch.equal(before,model.fc2.weight)
