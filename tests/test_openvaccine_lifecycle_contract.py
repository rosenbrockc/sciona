"""Invalid raw lifecycle requests must fail before folding/model execution."""
import copy
import numpy as np
import pytest
from sciona import openvaccine_lifecycle as lifecycle
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS
from sciona.openvaccine_lifecycle_contract import validate_lifecycle
from sciona.openvaccine_splits import validate_member_splits


def request():
    plan={key:dict(supervised_reverse_flags=np.zeros(2,dtype=bool),eligible=np.ones((1,6,5),dtype=bool),
                  maximum_uncertainty=1.,perturbations=np.zeros((1,6,5),dtype=np.float32),
                  sample_weights=np.ones(1,dtype=np.float32),reverse_flags=np.zeros(1,dtype=bool)) for key in EXPECTED_MEMBERS}
    return dict(binary='unused',parameters='unused',sequences=['ACGU']*4,targets=np.ones((4,4,5),dtype=np.float32),
                cluster_ids=np.arange(4),proximity_factors=np.ones(4,dtype=np.float32),
                splits={key:dict(train=[0,1],validation=[2,3]) for key in EXPECTED_MEMBERS},pseudo_sequences=['ACGUAC'],
                prediction_sequences=['AC'],pseudo_rounds=[plan,copy.deepcopy(plan)],pretraining_steps=1,
                initial_supervised_steps=1,rollback_policy='model_and_optimizer',absolute_tolerance=0.,std_ddof=0)


def test_valid_different_length_populations_and_two_rounds():
    r=request();m=validate_member_splits(r['targets'],r['proximity_factors'],r['splits'])
    validate_lifecycle(r['sequences'],r['targets'],r['pseudo_sequences'],r['prediction_sequences'],r['pseudo_rounds'],m,
        rollback_policy=r['rollback_policy'],absolute_tolerance=r['absolute_tolerance'],std_ddof=r['std_ddof'])


@pytest.mark.parametrize('bad',['alphabet','mixed_lengths','target_length','missing_later_member','extra_field','nan_draw',
                               'wrong_flags','zero_weights','empty_eligibility','negative_threshold','policy','ddof','steps'])
def test_all_rounds_preflight_before_folding(monkeypatch,bad):
    r=request();plan=r['pseudo_rounds'][1];key=next(iter(plan));p=plan[key]
    if bad=='alphabet':r['prediction_sequences']=['AX']
    if bad=='mixed_lengths':r['pseudo_sequences']=['AC','ACG']
    if bad=='target_length':r['targets']=np.ones((4,3,5),dtype=np.float32)
    if bad=='missing_later_member':del plan[key]
    if bad=='extra_field':p['unexpected']=1
    if bad=='nan_draw':p['perturbations'][0,0,0]=np.nan
    if bad=='wrong_flags':p['supervised_reverse_flags']=np.ones(1,dtype=bool)
    if bad=='zero_weights':p['sample_weights'][:]=0
    if bad=='empty_eligibility':p['eligible'][:]=False
    if bad=='negative_threshold':p['maximum_uncertainty']=-1
    if bad=='policy':r['rollback_policy']='unknown'
    if bad=='ddof':r['std_ddof']=True
    if bad=='steps':r['pretraining_steps']=0
    def forbidden(*args,**kwargs):raise AssertionError('Folding/model must not execute')
    monkeypatch.setattr(lifecycle,'fold_features',forbidden);monkeypatch.setattr(lifecycle,'create_model',forbidden)
    with pytest.raises(ValueError):lifecycle.execute_population('unused',**r)
