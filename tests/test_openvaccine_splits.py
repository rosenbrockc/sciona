"""Synthetic explicit membership boundaries, including masked validation."""
import copy
import numpy as np
import pytest
from sciona.openvaccine_splits import validate_member_splits
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS


def inputs():
    targets=np.ones((4,3,5),dtype=np.float32);targets[:,0,:]=np.nan
    weights=np.ones(4,dtype=np.float32)
    splits={key:dict(train=[0,1],validation=[2,3]) for key in EXPECTED_MEMBERS}
    return targets,weights,splits


def test_complete_membership_copies_indices_and_preserves_masks():
    targets,weights,splits=inputs();before=targets.copy()
    result=validate_member_splits(targets,weights,splits)
    key=next(iter(splits));splits[key]['train'][0]=3
    assert result[key].train==(0,1) and result[key].validation==(2,3)
    with pytest.raises(TypeError):result[key]=None
    np.testing.assert_array_equal(targets,before)


@pytest.mark.parametrize('problem',['missing','extra','overlap','duplicate','outside','empty','boolean','fields',
                                     'unobserved_validation','zero_training_weight','infinite_target','negative_weight'])
def test_invalid_membership(problem):
    targets,weights,splits=inputs();key=next(iter(splits));entry=splits[key]
    if problem=='missing':del splits[key]
    if problem=='extra':splits[('other',0)]=copy.deepcopy(entry)
    if problem=='overlap':entry['validation']=[1,2]
    if problem=='duplicate':entry['train']=[0,0]
    if problem=='outside':entry['validation']=[4]
    if problem=='empty':entry['train']=[]
    if problem=='boolean':entry['train']=[False,1]
    if problem=='fields':entry['unexpected']=1
    if problem=='unobserved_validation':targets[2:]=np.nan
    if problem=='zero_training_weight':weights[:2]=0
    if problem=='infinite_target':targets[0,1,0]=np.inf
    if problem=='negative_weight':weights[0]=-1
    with pytest.raises(ValueError):validate_member_splits(targets,weights,splits)
