"""Strict serialized private population boundary for reconstructed OpenVaccine."""
import copy
import json
from dataclasses import dataclass
import os
import numpy as np
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS
from sciona.openvaccine_population import population_weights
from sciona.openvaccine_splits import validate_member_splits
from sciona.openvaccine_lifecycle_contract import validate_lifecycle

_FIELDS={'version','seed','sequences','targets','cluster_ids','proximity_factors','splits',
         'pseudo_sequences','prediction_sequences','pseudo_rounds','pretraining_steps',
         'initial_supervised_steps','rollback_policy','absolute_tolerance','std_ddof'}
_PLAN={'supervised_reverse_flags','eligible','maximum_uncertainty','perturbations','sample_weights','reverse_flags'}


@dataclass(frozen=True)
class Prepared:
    payload: dict


def _numeric(value,masked=False):
    try:
        raw=np.asarray(value,dtype=object)
        if any((v is None and not masked) or (v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)))) for v in raw.flat):
            raise ValueError()
        with np.errstate(over='ignore',invalid='ignore'):return np.asarray(value,dtype=np.float32)
    except (ValueError,TypeError,OverflowError):raise ValueError('Invalid numeric payload array') from None


def _members(entries,fields):
    if not isinstance(entries,list):raise ValueError('Explicit member list required')
    result={}
    for entry in entries:
        if not isinstance(entry,dict) or set(entry)!=fields|{'family','slot'}:raise ValueError('Invalid member fields')
        if not isinstance(entry['family'],str) or type(entry['slot']) is not int:raise ValueError('Invalid member identity')
        key=entry['family'],entry['slot']
        if key not in EXPECTED_MEMBERS or key in result:raise ValueError('Unknown or duplicate member')
        result[key]={k:v for k,v in entry.items() if k not in ('family','slot')}
    if set(result)!=EXPECTED_MEMBERS:raise ValueError('Missing ensemble members')
    return result


def _arguments(payload):
    if not isinstance(payload,dict) or set(payload)!=_FIELDS:raise ValueError('Unexpected OpenVaccine payload fields')
    try:json.dumps(payload,allow_nan=False)
    except (ValueError,TypeError,OverflowError):raise ValueError('Finite JSON payload required; use null for masked targets') from None
    if not isinstance(payload['cluster_ids'],list) or any(type(v) is not int for v in payload['cluster_ids']):raise ValueError('Integer cluster membership required')
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported payload version')
    if type(payload['seed']) is not int or not 0<=payload['seed']<2**31:raise ValueError('Invalid initialization seed')
    for name in ('pretraining_steps','initial_supervised_steps'):
        if type(payload[name]) is not int or payload[name]<1:raise ValueError('Positive explicit training steps required')
    splits=_members(payload['splits'],{'train','validation'})
    targets=_numeric(payload['targets'],masked=True)
    try:clusters=np.asarray(payload['cluster_ids'])
    except (ValueError,TypeError):raise ValueError('Invalid cluster membership') from None
    proximity=_numeric(payload['proximity_factors'])
    weights=population_weights(clusters,proximity)
    membership=validate_member_splits(targets,weights,splits)
    if not isinstance(payload['pseudo_rounds'],list) or not payload['pseudo_rounds']:raise ValueError('Explicit rounds required')
    rounds=[]
    for entries in payload['pseudo_rounds']:
        plan=_members(entries,_PLAN)
        for recipe in plan.values():
            for name in ('supervised_reverse_flags','eligible','reverse_flags'):
                recipe[name]=np.asarray(recipe[name])
                if recipe[name].dtype!=np.bool_:raise ValueError('Boolean flags required')
            for name in ('perturbations','sample_weights'):recipe[name]=_numeric(recipe[name])
        rounds.append(plan)
    validate_lifecycle(payload['sequences'],targets,payload['pseudo_sequences'],payload['prediction_sequences'],rounds,membership,
        rollback_policy=payload['rollback_policy'],absolute_tolerance=payload['absolute_tolerance'],std_ddof=payload['std_ddof'])
    args={k:payload[k] for k in ['sequences','pseudo_sequences','prediction_sequences','pretraining_steps','initial_supervised_steps','rollback_policy','absolute_tolerance','std_ddof']}
    return dict(args,targets=targets,cluster_ids=clusters,proximity_factors=proximity,splits=splits,pseudo_rounds=rounds)


def prepare(payload):
    _arguments(payload)
    return Prepared(copy.deepcopy(payload))


def execute(prepared):
    if not isinstance(prepared,Prepared):raise ValueError('Prepared OpenVaccine payload required')
    payload=copy.deepcopy(prepared.payload);args=_arguments(payload)
    source=os.environ.get('SCIONA_OPENVACCINE_SOURCE_DIR')
    binary=os.environ.get('SCIONA_OPENVACCINE_FOLD_BINARY')
    parameters=os.environ.get('SCIONA_OPENVACCINE_FOLD_PARAMETERS')
    if not source or not binary or not parameters:raise ValueError('Provision reviewed OpenVaccine source and folding dependencies')
    import tensorflow as tf
    tf.keras.utils.set_random_seed(payload['seed'])
    from sciona.openvaccine_lifecycle import execute_population
    result=execute_population(source,binary=binary,parameters=parameters,**args)
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
