"""Strict JSON populations for group-isolated image/tabular ensembles."""
from dataclasses import dataclass
import copy
import math
import numpy as np
from sciona.image_tabular_features import image_features,metadata
from sciona.image_tabular_training import fit,validate_split,validate_controls

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
    validate_controls(payload['controls']);widths=[];groups=[]
    for name in ('training','calibration','query'):
        part=payload[name];fields={'images','numeric','categorical','groups'}
        if name!='query':fields.add('labels')
        if name=='training':fields.add('folds')
        if type(part) is not dict or set(part)!=fields:raise ValueError('Invalid population fields')
        visual=image_features(part['images']);count=len(visual)
        values,tokens=metadata(part['numeric'],part['categorical'],count)
        widths.append((values.shape[1],tokens.shape[1]))
        ids=part['groups']
        if type(ids) is not list or len(ids)!=count or any(type(g) is not str or not g for g in ids):raise ValueError('Aligned group identifiers required')
        groups.append(set(ids))
        if name!='query':
            labels=part['labels']
            if type(labels) is not list or len(labels)!=count or any(type(v) is not int or v not in (0,1) for v in labels) or set(labels)!={0,1}:raise ValueError('Both binary classes required')
        if name=='training':validate_split(part['labels'],part['folds'],ids,count)
    if len(set(widths))!=1:raise ValueError('Metadata widths differ between populations')
    if any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError('Groups overlap populations')
    return Prepared(copy.deepcopy(payload))

def threshold(scores,labels):
    values=np.asarray(scores,dtype=np.float64);y=np.asarray(labels)
    if values.ndim!=1 or y.shape!=values.shape or not len(values) or not np.isfinite(values).all() or (values<0).any() or (values>1).any() or set(y)!={0,1}:raise ValueError('Invalid calibration scores/labels')
    best=(-1.,-1.)
    for cut in np.append(np.unique(values),np.nextafter(values.max(),np.inf)):
        predicted=values>=cut;tp=np.count_nonzero(predicted & (y==1))
        score=2*tp/(predicted.sum()+y.sum())
        best=max(best,(float(score),float(cut)))
    return best[1]


def execute(prepared):
    if not isinstance(prepared,Prepared):raise ValueError('Prepared image-tabular input required')
    p=prepare(prepared.payload).payload
    fitted=fit(**p['training'],controls=p['controls'])
    def predict(part):return fitted.predict(part['images'],part['numeric'],part['categorical'])
    cut=threshold(predict(p['calibration']),p['calibration']['labels'])
    scores=predict(p['query'])
    result=dict(scores=scores.tolist(),classes=(scores>=cut).astype(int).tolist(),threshold=cut,
        training_rows=len(p['training']['images']),calibration_rows=len(p['calibration']['images']),query_rows=len(scores),
        folds=len(fitted.models),oof_rows=len(fitted.oof),visual_features=31,epochs=p['controls']['epochs'],
        initial_training_loss=float(np.mean([m.history[0] for m in fitted.models])),final_training_loss=float(np.mean([m.history[-1] for m in fitted.models])))
    finite_json(result)
    return result
