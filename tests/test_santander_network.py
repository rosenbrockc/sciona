"""Synthetic attention and gradient evidence for the reconstructed network."""
import pytest
import torch
from sciona.santander_network import GroupedAttentionClassifier


def case(rows=8, groups=4):
    rng = torch.Generator().manual_seed(13)
    return (torch.randint(6, (rows, groups), generator=rng),
        torch.randn(rows, groups, generator=rng), torch.randn(rows, groups, generator=rng))


def test_uniform_attention_matches_independent_branch_average():
    torch.manual_seed(12)
    model = GroupedAttentionClassifier(4).eval()
    with torch.no_grad():
        model.weight_network[-1].weight.zero_()
        model.weight_network[-1].bias.zero_()
    captured = {}
    handles = [model.pooled_norm.register_forward_pre_hook(lambda m, args: captured.update(pooled=args[0]))]
    for name in ('category_norm', 'raw_norm', 'substituted_norm'):
        handles.append(getattr(model, name).register_forward_hook(
            lambda m, args, output, name=name: captured.update({name: output})))
    logits, weights = model(*case(), return_weights=True)
    for handle in handles: handle.remove()
    torch.testing.assert_close(weights, torch.full((8, 4), 0.25))
    expected = torch.cat([captured[name].mean(dim=1) for name in
        ('category_norm', 'raw_norm', 'substituted_norm')], dim=1)
    torch.testing.assert_close(captured['pooled'], expected)
    assert logits.shape == (8, 2) and torch.isfinite(logits).all()


def test_learned_attention_and_all_paths_receive_gradients():
    torch.manual_seed(17)
    model = GroupedAttentionClassifier(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.head[-1].weight.detach().clone()
    logits, weights = model(*case(), return_weights=True)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(8))
    assert (weights > 0).all() and not torch.allclose(weights, torch.full_like(weights, 0.25))
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(8) % 2)
    loss.backward()
    for module in (model.category, model.raw_branch, model.substituted_branch,
                   model.weight_network, model.value_intercept, model.weight_intercept, model.head):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    optimizer.step()
    assert not torch.equal(before, model.head[-1].weight)


def test_eval_single_row_is_deterministic_without_state_updates():
    model = GroupedAttentionClassifier(4).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    inputs = case(rows=1)
    torch.testing.assert_close(model(*inputs), model(*inputs), rtol=0, atol=0)
    for k, value in model.state_dict().items(): torch.testing.assert_close(before[k], value, rtol=0, atol=0)


@pytest.mark.parametrize('invalid', ['groups','shape','category','nonfinite','dtype','singleton'])
def test_reject_invalid_inputs(invalid):
    model = GroupedAttentionClassifier(4)
    values = list(case())
    if invalid == 'groups': values = list(case(groups=3))
    if invalid == 'shape': values[1] = values[1][:-1]
    if invalid == 'category': values[0][0,0] = 6
    if invalid == 'nonfinite': values[1][0,0] = float('nan')
    if invalid == 'dtype': values[2] = values[2].double()
    if invalid == 'singleton': values = list(case(rows=1))
    with pytest.raises(ValueError): model(*values)
