"""Preflight rejection must occur before inference and generation writes."""
import copy
import numpy as np
import pytest
from sciona.openvaccine_round_contract import validate_recipes
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS
from sciona import openvaccine_round as rounds


def fixture():
    n=np.zeros((2,4,55),dtype=np.float32);a=np.zeros((2,4,4,8),dtype=np.float32)
    t=np.ones((2,2,5),dtype=np.float32);w=np.ones(2,dtype=np.float32)
    recipe=dict(supervised_batches=[(n,a,t,w)],validation_batch=(n,a,t,w),
                eligible=np.ones(t.shape,dtype=bool),maximum_uncertainty=1.,perturbations=np.zeros_like(t),
                sample_weights=w,reverse_flags=np.zeros(2,dtype=bool))
    return n,a,{key:copy.deepcopy(recipe) for key in EXPECTED_MEMBERS}


def test_complete_recipe_is_accepted_without_tensorflow():
    n,a,r=fixture();validate_recipes(r,n,a,std_ddof=0)


@pytest.mark.parametrize('bad',['missing_member','extra_field','empty_section','target_inf','validation_inf',
                               'zero_weights','no_weighted_targets','empty_eligibility','nan_draw','negative_limit','bad_flags','bad_ddof'])
def test_invalid_recipe_rejected_before_any_execution(tmp_path,monkeypatch,bad):
    n,a,r=fixture();key=next(iter(r));recipe=r[key];ddof=0
    if bad=='missing_member':del r[key]
    if bad=='extra_field':recipe['unexpected']=1
    if bad=='empty_section':recipe['supervised_batches']=[]
    if bad=='target_inf':recipe['supervised_batches'][0][2][0,0,0]=np.inf
    if bad=='validation_inf':recipe['validation_batch'][2][0,0,0]=np.inf
    if bad=='zero_weights':recipe['sample_weights'][:]=0
    if bad=='no_weighted_targets':
        recipe['supervised_batches'][0][2][0]=np.nan
        recipe['supervised_batches'][0][3][:]=[1,0]
    if bad=='empty_eligibility':recipe['eligible'][:]=False
    if bad=='nan_draw':recipe['perturbations'][0,0,0]=np.nan
    if bad=='negative_limit':recipe['maximum_uncertainty']=-1
    if bad=='bad_flags':recipe['reverse_flags']=np.zeros(2,dtype=np.int64)
    if bad=='bad_ddof':ddof=True
    def forbidden(*args,**kwargs):raise AssertionError('Model inference must not start')
    monkeypatch.setattr(rounds,'predict_teacher',forbidden)
    output=tmp_path/'new_generation'
    with pytest.raises(ValueError):
        rounds.refine_round('unused',[],n,a,recipes=r,output_directory=output,std_ddof=ddof,
                            rollback_policy='model_and_optimizer',absolute_tolerance=0.)
    assert not output.exists()


def test_uncertainty_mask_cannot_leave_only_zero_weight_labels(tmp_path,monkeypatch):
    n,a,r=fixture()
    for recipe in r.values():recipe['sample_weights']=np.array([1,0],dtype=np.float32)
    std=np.zeros((2,2,5),dtype=np.float32);std[0]=2
    monkeypatch.setattr(rounds,'predict_teacher',lambda *args,**kwargs:dict(teacher_mean=np.ones_like(std),teacher_std=std))
    output=tmp_path/'new_generation'
    with pytest.raises(ValueError,match='No positive-weight'):
        rounds.refine_round('unused',[],n,a,recipes=r,output_directory=output,std_ddof=0,
                            rollback_policy='model_and_optimizer',absolute_tolerance=0.)
    assert not output.exists()


def test_partially_observed_validation_accepted():
    n,a,r=fixture()
    for recipe in r.values():
        vn,va,vt,vw=recipe['validation_batch']
        masked=vt.copy();masked[:,0,:]=np.nan
        recipe['validation_batch']=(vn,va,masked,vw)
    validate_recipes(r,n,a,std_ddof=0)


def test_validation_without_positive_weight_observations_rejected():
    n,a,r=fixture();recipe=next(iter(r.values()))
    vn,va,vt,vw=recipe['validation_batch']
    masked=vt.copy();masked[0]=np.nan
    recipe['validation_batch']=(vn,va,masked,np.array([1,0],dtype=np.float32))
    with pytest.raises(ValueError,match='positive weight'):validate_recipes(r,n,a,std_ddof=0)
