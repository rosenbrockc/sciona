import pytest
import torch

from sciona.bengali_sampling import SamplingBudget,paired_batches


def test_matches_source_loader_consumption_across_epoch_boundaries():
    budget=SamplingBudget(hand_count=5,font_count=19,batch_size=8,epochs=3)
    assert budget.requested_samples_per_domain==57
    assert budget.steps_per_epoch==2 and budget.total_steps==6
    assert budget.unused_samples_per_domain==9  # One unused full batch plus a tail.
    actual=list(paired_batches(budget,hand_seed=19,font_seed=47))
    for count,seed,key in [(5,19,'hand_indices'),(19,47,'font_indices')]:
        sampler=torch.utils.data.RandomSampler(range(count),True,57,generator=torch.Generator().manual_seed(seed))
        loader=torch.utils.data.DataLoader(torch.arange(count),batch_size=8,drop_last=True,sampler=sampler)
        assert len(loader)//budget.epochs==budget.steps_per_epoch
        iterator=iter(loader)  # Once, not once per epoch.
        for batch in actual:
            torch.testing.assert_close(batch[key],next(iterator),rtol=0,atol=0)
    assert [batch['epoch'] for batch in actual]==[0,0,1,1,2,2]
    assert len(torch.unique(actual[0]['hand_indices'])) < 8  # Replacement is required.


def test_seed_replay_is_independent_of_global_rng():
    budget=SamplingBudget(9,17,8,2)
    left=list(paired_batches(budget,hand_seed=7,font_seed=8))
    torch.rand(500)
    right=list(paired_batches(budget,hand_seed=7,font_seed=8))
    for a,b in zip(left,right):
        for key in ['hand_indices','font_indices']:
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


@pytest.mark.parametrize('values',[(0,10,8,1),(10,10,0,1),(10,10,8,0),(2,3,8,1),(True,10,8,1)])
def test_invalid_or_zero_update_budget_rejected(values):
    with pytest.raises(ValueError):
        SamplingBudget(*values)
