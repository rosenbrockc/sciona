"""Synthetic fold-local table preprocessing contracts."""
import numpy as np
import pytest
from sciona.tabular_ensemble_features import TabularFeatures


def test_hand_imputation_duplicate_constant_removal_and_clipping():
    n=[[0.,0.,9.,None],[2.,2.,9.,None],[None,None,9.,None],[10.,10.,9.,None]]
    c=[['a'],['a'],[None],['b']]
    f=TabularFeatures().fit(n,c,clip_low=0,clip_high=1)
    assert f.kept==[0]
    np.testing.assert_array_equal(f.medians,[2.,2.,9.,0.])
    x=f.transform([[100.,100.,-5.,7.]],[['a']])
    assert x[0,0]==10. and x[0,-1]==.5
    assert x.shape==(1,5)


def test_missing_sentinel_cannot_collide_with_real_string_and_unknown_is_zero():
    f=TabularFeatures().fit([[],[],[]],[[None],['m:'],['']])
    x=f.transform([[],[],[],[]],[[None],['m:'],[''],['unseen']])
    assert len({tuple(row) for row in x[:3]})==3
    assert not x[-1].any()


def test_query_population_does_not_change_fitted_statistics():
    f=TabularFeatures().fit([[0.],[2.],[4.]],[['a'],['a'],['b']],clip_low=0,clip_high=1)
    med=f.medians.copy();counts=[dict(c) for c in f.counts]
    alone=f.transform([[1.]],[['a']])
    larger=f.transform([[1.],[9999.]],[['a'],['new']])
    np.testing.assert_array_equal(alone[0],larger[0])
    np.testing.assert_array_equal(med,f.medians)
    assert counts==[dict(c) for c in f.counts]


def test_fold_fit_uses_only_selected_training_population():
    train=TabularFeatures().fit([[0.],[2.],[None]],[[],[],[]],clip_low=0,clip_high=1)
    leaked=TabularFeatures().fit([[0.],[2.],[None],[100.]],[[],[],[],[]],clip_low=0,clip_high=1)
    assert train.transform([[None]],[[]])[0,0]==1.
    assert leaked.transform([[None]],[[]])[0,0]==2.


@pytest.mark.parametrize('n,c',[([],[]),([[1.],[2.,3.]],[[],[]]),([[float('inf')]],[[]]),([[True]],[[]]),([[]],[[1]]),([[1.]],[[],[]])])
def test_invalid_tables(n,c):
    with pytest.raises(ValueError):TabularFeatures().fit(n,c)


def test_all_constant_numeric_rejected():
    with pytest.raises(ValueError):TabularFeatures().fit([[None],[None]],[[],[]])
