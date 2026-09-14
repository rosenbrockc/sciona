"""Synthetic arithmetic and failure boundaries for pseudo-label construction."""
import numpy as np
import pytest
from sciona.openvaccine_targets import uncertainty_targets, validation_rollback_required


def test_uncertainty_boundary_and_explicit_draws_preserve_inputs():
    mean=np.ones((1,2,5),dtype=np.float32)
    std=np.array([0,.25,.5,.75,1]*2,dtype=np.float32).reshape(mean.shape)
    eligible=np.ones(mean.shape,dtype=bool);eligible[0,1,0]=False
    draws=np.full(mean.shape,2,dtype=np.float32)
    copies=[x.copy() for x in [mean,std,eligible,draws]]
    out=uncertainty_targets(mean,std,eligible,maximum_uncertainty=.5,perturbations=draws)
    expected=np.array([1,1.5,2,np.nan,np.nan,np.nan,1.5,2,np.nan,np.nan],dtype=np.float32).reshape(mean.shape)
    np.testing.assert_array_equal(out,expected)
    for actual,copy in zip([mean,std,eligible,draws],copies):np.testing.assert_array_equal(actual,copy)


@pytest.mark.parametrize('problem',['negative_std','nan_draw','empty_mask','shape','dtype','overflow'])
def test_reject_invalid_targets(problem):
    mean=np.ones((1,2,5),dtype=np.float32);std=mean.copy();draws=mean.copy();eligible=np.ones(mean.shape,dtype=bool)
    if problem=='negative_std':std[0,0,0]=-1
    if problem=='nan_draw':draws[0,0,0]=np.nan
    if problem=='empty_mask':eligible[:]=False
    if problem=='shape':draws=draws[:,:1]
    if problem=='dtype':eligible=eligible.astype(np.float32)
    if problem=='overflow':mean[:]=np.finfo(np.float32).max;draws[:]=np.finfo(np.float32).max
    with pytest.raises(ValueError):uncertainty_targets(mean,std,eligible,maximum_uncertainty=1,perturbations=draws)


@pytest.mark.parametrize('after,expected',[(.25,False),(.5,False),(.625,False),(.75,True)])
def test_rollback_strict_absolute_threshold(after,expected):
    assert validation_rollback_required(.5,after,absolute_tolerance=.125) is expected


@pytest.mark.parametrize('before,after,tolerance',[(float('nan'),1,0),(0,float('inf'),0),(0,1,-1),(False,1,0)])
def test_rollback_invalid_scores(before,after,tolerance):
    with pytest.raises(ValueError):validation_rollback_required(before,after,absolute_tolerance=tolerance)
