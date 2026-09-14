import random
import pytest
from sciona.bengali_pretraining_sampling import PretrainingParameterSampler


@pytest.mark.parametrize('backend',['legacy','sfc64'])
def test_pretraining_decisions_and_cutout_match_source_order(backend):
    sampler=PretrainingParameterSampler(python_seed=17,image_seed=98,backend=backend)
    reference=random.Random(17)
    shears=[]
    for _ in range(64):
        actual=sampler.draw();shears.append(actual.shear)
        for _ in range(6):reference.random()
        assert actual.angle==reference.uniform(-20,20)
        assert actual.scale==reference.uniform(.9,1.1)
        assert actual.dx==reference.uniform(-.0625,.0625)
        assert actual.dy==reference.uniform(-.0625,.0625)
        reference.random()
        assert actual.h_start==reference.random() and actual.w_start==reference.random()
        reference.random()
        assert actual.cutout_y==reference.randint(0,224)
        assert actual.cutout_x==reference.randint(0,224)
    assert sampler.python.getstate()==reference.getstate()
    assert max(shears)>5 and min(shears)<-5
