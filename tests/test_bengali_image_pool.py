import pytest
import torch
from sciona.bengali_image_pool import ImageReplayPool


def images(start, count):
    return torch.arange(start,start+count,dtype=torch.float32).reshape(count,1,1,1).expand(-1,3,2,2).clone()


def test_fill_and_replay_are_detached_and_do_not_alias_inputs():
    pool=ImageReplayPool(seed=12)
    source=images(0,50).requires_grad_()
    output=pool.query(source)
    assert not output.requires_grad
    with torch.no_grad():source.zero_()
    assert pool.images[-1][0,0,0]==49
    output.zero_()
    assert pool.images[-1][0,0,0]==49
    replay=pool.query(images(50,100))[:,0,0,0]
    current=torch.arange(50,150,dtype=torch.float32)
    assert (replay==current).any() and (replay<current).any()
    assert len(pool.images)==50


def test_snapshot_restores_exact_history_and_rng_sequence():
    pool=ImageReplayPool(seed=73);pool.query(images(0,80))
    state=pool.snapshot();expected=pool.query(images(80,100))
    restored=ImageReplayPool(seed=0);restored.restore(state)
    for image in state['images']:image.zero_()
    torch.testing.assert_close(restored.query(images(80,100)),expected,rtol=0,atol=0)


def test_invalid_query_does_not_change_replay_state():
    pool=ImageReplayPool(seed=18);pool.query(images(0,50));state=pool.snapshot()
    with pytest.raises(ValueError):pool.query(torch.full((1,3,2,2),float('nan')))
    assert pool.snapshot()['random_state']==state['random_state']
    for a,b in zip(pool.images,state['images']):torch.testing.assert_close(a,b,rtol=0,atol=0)
