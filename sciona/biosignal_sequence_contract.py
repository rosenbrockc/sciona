"""Strict subject-separated signal lifecycle with recording-level F1 decisions."""
from dataclasses import dataclass
import copy
import math
import numpy as np
from sciona.biosignal_sequence_features import signal_array
from sciona.biosignal_sequence_training import fit

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
    c=payload['controls']
    fields={'sample_rate','window_size','stride','n_fft','hop','seed','width','epochs','batch_size','learning_rate','max_frequency','max_time'}
    if type(c) is not dict or set(c)!=fields:raise ValueError('Invalid controls')
    for name in fields-{'sample_rate','learning_rate'}:
        if type(c[name]) is not int or c[name]<(0 if name in ('seed','max_frequency','max_time') else 1):raise ValueError('Invalid integer control')
    for name in ('sample_rate','learning_rate'):
        if type(c[name]) not in (int,float) or c[name]<=0:raise ValueError('Positive numeric control required')
    if c['seed']>=2**32 or c['stride']>c['window_size'] or not 4<=c['n_fft']<=c['window_size'] or c['hop']>c['n_fft']:raise ValueError('Invalid window/FFT controls')
    if c['max_frequency']>=c['n_fft']//2+1 or c['max_time']>=(c['window_size']-c['n_fft'])//c['hop']+1:raise ValueError('Mask must leave bins unmasked')
    groups=[];channels=None
    for name in ('training','calibration','query'):
        part=payload[name];fields={'signals','subjects'}
        if name!='query':fields.add('labels')
        if type(part) is not dict or set(part)!=fields or type(part['signals']) is not list or not part['signals']:raise ValueError('Invalid population fields')
        for signal in part['signals']:
            array=signal_array(signal)
            if channels is None:channels=array.shape[0]
            if array.shape[0]!=channels or array.shape[1]<c['window_size']:raise ValueError('Channel mismatch or short recording')
        subjects=part['subjects']
        if type(subjects) is not list or len(subjects)!=len(part['signals']) or any(type(s) is not str or not s for s in subjects):raise ValueError('Aligned subject identifiers required')
        groups.append(set(subjects))
        if name!='query':
            labels=part['labels']
            if type(labels) is not list or len(labels)!=len(subjects) or any(type(v) is not int or v not in (0,1) for v in labels) or set(labels)!={0,1}:raise ValueError('Both binary classes required')
    if any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError('Subjects overlap populations')
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
    if not isinstance(prepared,Prepared):raise ValueError('Prepared signals required')
    p=prepare(prepared.payload).payload;c=p['controls']
    trained=fit(p['training']['signals'],p['training']['labels'],c)
    calibration,_,_=trained.predict(p['calibration']['signals'])
    cut=threshold(calibration,p['calibration']['labels'])
    scores,counts,tails=trained.predict(p['query']['signals'])
    result=dict(scores=scores,classes=[int(v>=cut) for v in scores],threshold=cut,
        training_rows=len(p['training']['signals']),calibration_rows=len(calibration),query_rows=len(scores),
        training_windows=sum(trained.counts),query_windows=sum(counts),omitted_query_samples=sum(tails),
        epochs=c['epochs'],initial_training_loss=trained.history[0],final_training_loss=trained.history[-1])
    finite_json(result)
    return result
