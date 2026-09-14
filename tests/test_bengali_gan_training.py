import pytest
import torch
from torch import nn

from sciona.bengali_gan_training import CycleGANTraining


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.7))

    def forward(self, x):
        return x * self.scale


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.3))
        self.calls = []

    def forward(self, x):
        self.calls.append((self.scale.requires_grad, x.requires_grad))
        return x.mean((1,2,3)) * self.scale


def make_training(total_steps=4):
    classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(3), nn.Linear(3,14784))
    return CycleGANTraining(generator_a=Generator(), generator_b=Generator(),
        discriminator_a=Discriminator(), discriminator_b=Discriminator(),
        classifier=classifier, classifier_weight=1., total_steps=total_steps,
        replay_seed_a=3, replay_seed_b=4)


def test_updates_isolate_gradients_and_replay_pre_update_fakes():
    torch.set_num_threads(2)
    torch.manual_seed(61)
    trainer = make_training()
    frozen = {k:v.clone() for k,v in trainer.guidance.classifier.state_dict().items()}
    old_g = [p.detach().clone() for p in trainer.generator_parameters]
    old_d = [p.detach().clone() for p in trainer.discriminator_parameters]
    a, b = torch.full((2,3,224,224), .8), torch.full((2,3,224,224), .4)
    labels = torch.tensor([0,14783])
    gradient_calls = []
    handles = [p.register_hook(lambda grad: gradient_calls.append(grad.clone())) for p in trainer.generator_parameters]
    result = trainer.step(a,b,labels)
    for handle in handles:
        handle.remove()
    assert result['step'] == 1
    assert len(gradient_calls) == 2  # No second backward into generators from discriminator.
    assert all(not torch.equal(old,p) for old,p in zip(old_g,trainer.generator_parameters))
    assert all(not torch.equal(old,p) for old,p in zip(old_d,trainer.discriminator_parameters))
    for discriminator in [trainer.discriminator_a, trainer.discriminator_b]:
        assert discriminator.calls == [(False,True),(True,False),(True,False)]
    torch.testing.assert_close(torch.stack(trainer.pool_a.images), b*.7, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(trainer.pool_b.images), a*.7, rtol=0, atol=0)
    assert not trainer.guidance.classifier.training
    assert all(p.grad is None for p in trainer.guidance.classifier.parameters())
    for name,value in trainer.guidance.classifier.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)


def test_source_flat_then_decay_schedule_and_exhaustion():
    trainer = make_training()
    images = torch.full((1,3,224,224), .2)
    rates = [trainer.step(images,images,torch.tensor([0]))['learning_rate'] for _ in range(4)]
    assert rates == [.0002,.0002,.0002,.0001]
    assert trainer.generator_optimizer.param_groups[0]['lr'] == 0
    assert trainer.discriminator_optimizer.param_groups[0]['lr'] == 0
    with pytest.raises(ValueError, match='exhausted'):
        trainer.step(images,images,torch.tensor([0]))


def test_invalid_labels_rejected_before_any_training_mutation():
    trainer = make_training()
    original = [p.clone() for p in trainer.generator_parameters + trainer.discriminator_parameters]
    with pytest.raises(ValueError, match='labels'):
        trainer.step(torch.ones(1,3,224,224),torch.ones(1,3,224,224),torch.tensor([14784]))
    assert trainer.completed_steps == 0 and not trainer.pool_a.images and not trainer.pool_b.images
    assert not trainer.generator_optimizer.state and not trainer.discriminator_optimizer.state
    for old,p in zip(original,trainer.generator_parameters + trainer.discriminator_parameters):
        torch.testing.assert_close(old,p,rtol=0,atol=0)
