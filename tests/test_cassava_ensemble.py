import numpy as np
import pytest
from sciona.cassava_ensemble import combine


def test_stacked_mean_differs_from_equal_four_family_average():
    a=np.array([[.9,.1,0,0,0]])
    b=np.array([[.1,.9,0,0,0]])
    result=combine(a,a,b,b)
    np.testing.assert_allclose(result['scores'],[[1.1,1.9,0,0,0]])
    assert result['labels'].tolist()==[1]
    assert np.argmax((a+a+b+b)/4,axis=1).tolist()==[0]


def test_random_scalar_oracle_and_input_immutability():
    rng=np.random.default_rng(394)
    arrays=[rng.dirichlet(np.ones(5),size=20) for _ in range(4)]
    before=[a.copy() for a in arrays]
    actual=combine(*arrays)
    expected=np.array([[(arrays[0][i,j]+arrays[1][i,j])/2+arrays[2][i,j]+arrays[3][i,j]
                        for j in range(5)] for i in range(20)])
    np.testing.assert_allclose(actual['scores'],expected)
    np.testing.assert_allclose(actual['scores'].sum(axis=1),3.)
    for a,b in zip(arrays,before):np.testing.assert_array_equal(a,b)


def test_class_ties_select_first_index():
    values=np.full((2,5),.2)
    np.testing.assert_array_equal(combine(values,values,values,values)['labels'],[0,0])


@pytest.mark.parametrize('bad',[
    np.full((1,4),.25),np.full((2,5),.2),np.full((1,5),np.nan),
    np.array([[-1.,2.,0,0,0]]),np.zeros((1,5)),np.ones((1,5),dtype=bool)])
def test_invalid_family_outputs_reject(bad):
    good=np.full((1,5),.2)
    with pytest.raises(ValueError):combine(good,good,good,bad)
def test_unknown_mass_is_distributed_not_dropped():
    from sciona.cassava_ensemble import distribute_unknown
    import numpy as np
    raw = np.array([[.1, .2, .05, .1, .05, .5], [0, 0, 0, 0, 0, 1.]])
    before = raw.copy()
    mapped = distribute_unknown(raw)
    np.testing.assert_allclose(mapped, [[.2, .3, .15, .2, .15], [.2] * 5])
    np.testing.assert_allclose(mapped.sum(1), 1.)
    np.testing.assert_array_equal(raw, before)

