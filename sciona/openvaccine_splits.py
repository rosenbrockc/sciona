"""Caller-defined ensemble membership without inferred historical folds.

Disjoint record indices prevent direct within-member overlap. They do not prove
sequence/cluster independence or remove cross-member pseudo-label leakage.
"""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS


@dataclass(frozen=True)
class MemberSplit:
    train: tuple[int,...]
    validation: tuple[int,...]


def validate_member_splits(targets,weights,splits):
    targets,weights=np.asarray(targets),np.asarray(weights)
    if (targets.dtype!=np.float32 or targets.ndim!=3 or targets.shape[0]<2
            or targets.shape[1]<2 or targets.shape[2]!=5 or np.isinf(targets).any()):
        raise ValueError('Population requires supported float32 masked targets')
    count=len(targets)
    if (weights.dtype!=np.float32 or weights.shape!=(count,) or not np.isfinite(weights).all()
            or (weights<0).any()):
        raise ValueError('Population requires finite nonnegative float32 weights')
    if not isinstance(splits,dict) or set(splits)!=EXPECTED_MEMBERS:
        raise ValueError('Explicit splits for all twenty model identities required')
    checked={}
    for key,entry in splits.items():
        if not isinstance(key,tuple) or len(key)!=2 or type(key[1]) is not int:
            raise ValueError('Invalid model identity')
        if not isinstance(entry,dict) or set(entry)!={'train','validation'}:
            raise ValueError('Exact train and validation membership fields required')
        groups={}
        for role in ('train','validation'):
            values=entry[role]
            if not isinstance(values,(list,tuple)) or not values or any(type(i) is not int for i in values):
                raise ValueError('Nonempty integer index sequences required')
            indices=tuple(values)
            if len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=count:
                raise ValueError('Duplicate or out-of-range split index')
            selected=np.array(indices,dtype=np.int64)
            selected_weights=weights[selected]
            if not np.isfinite(selected_weights.sum()) or not (np.isfinite(targets[selected]) & (selected_weights>0)[:,None,None]).any():
                raise ValueError('Each split needs positive-weight observed targets')
            groups[role]=indices
        if set(groups['train']) & set(groups['validation']):
            raise ValueError('Training and validation indices overlap')
        checked[key]=MemberSplit(**groups)
    return MappingProxyType(checked)
