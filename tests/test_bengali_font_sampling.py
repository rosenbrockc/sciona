import random
import numpy as np
import pytest

from sciona.bengali_font_sampling import FontParameterSampler


@pytest.mark.parametrize('backend',['legacy','sfc64'])
def test_explicit_seeds_replay_and_preserve_global_rng(backend):
    a=FontParameterSampler(python_seed=18,image_seed=61,backend=backend)
    b=FontParameterSampler(python_seed=18,image_seed=61,backend=backend)
    before=random.getstate()
    values=[a.draw() for _ in range(40)]
    assert values==[b.draw() for _ in range(40)]
    assert random.getstate()==before
    assert len({v.shear for v in values})==40
    assert all(float(np.float32(v.shear))==v.shear for v in values)


def test_probability_draws_are_not_skipped():
    sampler=FontParameterSampler(python_seed=18,image_seed=61,backend='sfc64')
    reference=random.Random(18)
    for _ in range(32):
        actual=sampler.draw()
        for _ in range(6):reference.random()
        assert actual.angle==reference.uniform(-5,5)
        assert actual.scale==reference.uniform(.9,1.1)
        assert actual.dx==reference.uniform(-.0625,.0625)
        assert actual.dy==reference.uniform(-.0625,.0625)
        reference.random()
        assert actual.h_start==reference.random() and actual.w_start==reference.random()
    assert sampler.python.getstate()==reference.getstate()


def test_backend_is_explicit():
    with pytest.raises(ValueError):FontParameterSampler(python_seed=1,image_seed=2,backend='auto')
