import pytest
import torch
from torch import nn
from sciona.bengali_font_guidance import FrozenFontGuidance


def classifier():
    return nn.Sequential(nn.BatchNorm2d(3), nn.Dropout(.5), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(3,14784))


def test_parent_training_preserves_classifier_buffers_and_pixel_gradients():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        torch.manual_seed(482)
        parent=nn.Module();parent.guidance=FrozenFontGuidance(classifier(),weight=4.)
        parent.train()
        module=parent.guidance
        assert not module.classifier.training
        before={k:v.clone() for k,v in module.classifier.state_dict().items()}
        images=torch.rand(2,3,224,224,requires_grad=True)
        loss=module(images,torch.tensor([0,14783]));loss.backward()
        assert images.grad is not None and torch.isfinite(images.grad).all() and images.grad.abs().sum()>0
        assert all(p.grad is None for p in module.classifier.parameters())
        for k,v in module.classifier.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        with torch.no_grad():
            torch.testing.assert_close(module(images,torch.tensor([0,14783])),loss.detach(),rtol=0,atol=0)
    finally:torch.set_num_threads(previous)


def test_external_classifier_mode_mutation_fails():
    module=FrozenFontGuidance(classifier(),weight=1.)
    module.classifier.train()
    with pytest.raises(ValueError,match='remain frozen'):
        module(torch.zeros(1,3,224,224),torch.tensor([0]))


def test_joint_class_range_rejected():
    module=FrozenFontGuidance(classifier(),weight=1.)
    with pytest.raises(ValueError,match='joint-class'):
        module(torch.zeros(1,3,224,224),torch.tensor([14784]))
