import numpy as np
import pytest
import torch

from sciona.bengali_font_sampling import FontParameterSampler
from sciona.bengali_population import ImagePopulation,training_batches
from sciona.bengali_sampling import SamplingBudget,paired_batches


def sampler():
    return FontParameterSampler(python_seed=92,image_seed=39,backend='sfc64')


def test_corrected_labels_and_image_alignment_survive_paired_sampling():
    images=np.stack([np.full((137,236),i*40,dtype=np.uint8) for i in range(3)])
    hand=ImagePopulation.corrected(images,['synthetic-a','synthetic-b','synthetic-c'],
        [[0,0,0],[1,0,0],[2,0,0]],['synthetic-c'],[[167,10,7]])
    font=ImagePopulation(images,[9,10,11])
    budget=SamplingBudget(3,3,2,2)
    expected=list(paired_batches(budget,hand_seed=4,font_seed=5))
    batches=list(training_batches(hand,font,budget,hand_seed=4,font_seed=5,font_sampler=sampler()))
    assert len(batches)==2
    for pair,batch in zip(expected,batches):
        a,b=pair['hand_indices'].numpy(),pair['font_indices'].numpy()
        torch.testing.assert_close(batch['labels_a'],torch.tensor([0,88,14783])[a])
        torch.testing.assert_close(batch['labels_b'],torch.tensor([9,10,11])[b])
        intensity=torch.tensor(a*40,dtype=torch.float32).div(255).sub(.5).div(.5)
        torch.testing.assert_close(batch['images_a'],intensity[:,None,None,None].expand(2,3,224,224),rtol=0,atol=0)
        assert batch['images_b'].shape==(2,3,224,224)
        assert torch.isfinite(batch['images_b']).all()
        assert batch['images_b'].min()>=-1 and batch['images_b'].max()<=1


def test_population_owns_inputs_and_full_batches_replay():
    images=np.random.default_rng(71).integers(0,256,(3,137,236),dtype=np.uint8)
    population=ImagePopulation(images,[0,1,2]);images[:]=0
    assert population.images.any()
    budget=SamplingBudget(3,3,2,2)
    def run():return list(training_batches(population,population,budget,hand_seed=5,font_seed=6,font_sampler=sampler()))
    for left,right in zip(run(),run()):
        for key in ['images_a','images_b','labels_a','labels_b']:
            torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)


def test_budget_mismatch_rejected_before_augmentation_draws():
    population=ImagePopulation(np.zeros((2,137,236),dtype=np.uint8),[0,1])
    rng=sampler();before=rng.python.getstate()
    with pytest.raises(ValueError,match='budget'):
        next(training_batches(population,population,SamplingBudget(3,2,2,1),hand_seed=1,font_seed=2,font_sampler=rng))
    assert rng.python.getstate()==before
