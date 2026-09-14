"""Finite JSON and group-isolated lifecycle for precomputed video features."""
import copy
import math
from dataclasses import dataclass
from sciona.video_multilabel_training import fit,validate_features
from sciona.video_multilabel_thresholds import label_matrix,calibrate,decisions

@dataclass(frozen=True)
class Prepared:
    payload: dict


def finite_json(value):
    if value is None or type(value) in (str,int,bool):return
    if type(value) is float and math.isfinite(value):return
    if type(value) is list:
        for item in value:finite_json(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():finite_json(item)
        return
    raise ValueError('Expected finite JSON values')


def prepare(payload):
    finite_json(payload)
    if type(payload) is not dict or set(payload)!={'version','training','calibration','query','controls'} or type(payload['version']) is not int or payload['version']!=1:raise ValueError('Invalid payload fields/version')
    widths=[];groups=[];label_count=None
    for name in ('training','calibration','query'):
        part=payload[name];fields={'videos','sparse','groups'}
        if name!='query':fields.add('labels')
        if type(part) is not dict or set(part)!=fields:raise ValueError('Invalid population fields')
        widths.append(validate_features(part['videos'],part['sparse']))
        rows=len(part['videos']);ids=part['groups']
        if type(ids) is not list or len(ids)!=rows or any(type(g) is not str or not g for g in ids):raise ValueError('Aligned nonempty group identifiers required')
        groups.append(set(ids))
        if name!='query':
            labels=part['labels']
            if type(labels) is not list or not labels or type(labels[0]) is not list or not labels[0]:raise ValueError('Nonempty label matrix required')
            if label_count is None:label_count=len(labels[0])
            label_matrix(labels,(rows,label_count))
    if len(set(widths))!=1:raise ValueError('Population feature widths differ')
    if any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError('Overlapping population groups')
    if not any(payload['training']['sparse']):raise ValueError('Training sparse vocabulary empty')
    controls=payload['controls']
    if type(controls) is not dict or set(controls)!={'seed','hidden','epochs','batch_size','learning_rate'}:raise ValueError('Invalid controls')
    for name in ('seed','hidden','epochs','batch_size'):
        if type(controls[name]) is not int or controls[name]<(0 if name=='seed' else 1):raise ValueError('Invalid integer control')
    if controls['seed']>=2**32:raise ValueError('Invalid random seed')
    if type(controls['learning_rate']) not in (int,float) or controls['learning_rate']<=0:raise ValueError('Positive learning rate required')
    return Prepared(copy.deepcopy(payload))


def execute(prepared):
    if not isinstance(prepared,Prepared):raise ValueError('Prepared video input required')
    p=prepare(prepared.payload).payload;training=p['training'];calibration=p['calibration'];query=p['query']
    fitted=fit(training['videos'],training['sparse'],training['labels'],**p['controls'])
    thresholds=calibrate(fitted.predict(calibration['videos'],calibration['sparse']),calibration['labels'])
    scores=fitted.predict(query['videos'],query['sparse'])
    result=dict(scores=scores.tolist(),labels=decisions(scores,thresholds),thresholds=thresholds,
        training_rows=len(training['videos']),calibration_rows=len(calibration['videos']),query_rows=len(query['videos']),
        label_count=len(thresholds),sparse_models=len(fitted.baseline),epochs=p['controls']['epochs'],
        initial_training_loss=fitted.history[0],final_training_loss=fitted.history[-1])
    finite_json(result)
    return result
