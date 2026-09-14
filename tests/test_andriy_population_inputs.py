import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_population_inputs import andriy_population_inputs,witness_andriy_population_inputs


def test_training_order_joint_normalization_and_prediction_retention():
    groups=[[np.full((19,1965),value)] for value in [1.,3.,5.,7.]]
    groups[0][0][0,0]=np.nan
    groups[3][0][0,0]=np.nan
    originals=[g[0].copy() for g in groups]
    train,labels,pred,valid=andriy_population_inputs(*groups)
    fit=np.r_[np.full(18,1.),np.full(19,3.),np.full(19,5.),np.full(18,7.)]
    expected=(np.r_[np.full(18,1.),np.full(19,3.),np.full(19,5.)]-fit.mean())/fit.std(ddof=1)
    np.testing.assert_allclose(train[:,0],expected)
    np.testing.assert_array_equal(labels,np.r_[np.ones(37),np.zeros(19)])
    assert pred.shape==(19,1965) and np.isnan(pred[0,0]) and not valid[0] and valid[1:].all()
    for group,original in zip(groups,originals): np.testing.assert_array_equal(group[0],original)


def test_empty_auxiliary_and_constant_column_source_semantics():
    rng=np.random.default_rng(701)
    p,n,t=[rng.normal(size=(19,1965)) for _ in range(3)]
    for x in [p,n,t]: x[:,0]=1
    train,labels,pred,valid=andriy_population_inputs([p],[],[n],[t])
    assert np.isnan(train[:,0]).all() and np.isnan(pred[:,0]).all()
    assert valid.all()  # source validity is fixed before normalization
    assert labels.shape==(38,)


def test_witness_represents_unknown_retained_count_without_fabrication():
    train,labels,pred,valid=witness_andriy_population_inputs([],[],[],[None,None])
    assert train.shape==('retained_training_windows','1965')
    assert labels.shape==('retained_training_windows',)
    assert pred.shape==(38,1965) and valid.shape==(38,)


@pytest.mark.parametrize('kind',['shape','infinity','missing_class','prediction'])
def test_invalid_groups(kind):
    groups=[[np.ones((19,1965))] for _ in range(4)]
    if kind=='shape': groups[0]=[np.ones((18,1965))]
    elif kind=='infinity': groups[2][0][0,0]=np.inf
    elif kind=='missing_class': groups[2][0][:]=np.nan
    else: groups[3]=[]
    with pytest.raises(ValueError): andriy_population_inputs(*groups)
