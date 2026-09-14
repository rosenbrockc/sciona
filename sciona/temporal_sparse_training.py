"""One-pass sparse log-target ensemble and frozen future-stream evaluation."""
from dataclasses import dataclass
import copy
from itertools import islice
import math
import numpy as np
from sklearn.linear_model import SGDRegressor
from sciona.temporal_sparse_features import TemporalSparseEncoder,validate_event


def bounded_batches(events,size):
    iterator=iter(events)
    while batch:=list(islice(iterator,size)):yield batch


def clipped(matrix,limit):
    result=matrix.copy();np.clip(result.data,-limit,limit,out=result.data)
    return result


def validate_controls(c):
    expected={'chunk_size','seed','learning_rate','alpha','feature_clip','max_prediction','gap'}
    if type(c) is not dict or set(c)!=expected:raise ValueError('Invalid streaming model controls')
    for name in ('chunk_size','seed'):
        if type(c[name]) is not int or c[name]<(0 if name=='seed' else 1):raise ValueError('Invalid integer control')
    if c['seed']>=2**32:raise ValueError('Invalid seed')
    for name in expected-{'chunk_size','seed'}:
        v=c[name]
        if type(v) not in (int,float) or not math.isfinite(v) or v<(0 if name in ('alpha','gap') else np.nextafter(0.,1.)):raise ValueError('Invalid finite numeric control')


@dataclass
class StreamEnsemble:
    models: list
    encoder: object
    controls: dict
    training_rows: int
    updates_per_model: int

    def predict_batches(self,events,*,labeled=False):
        if type(labeled) is not bool:raise ValueError('Boolean label flag required')
        encoder=copy.deepcopy(self.encoder);c=self.controls
        cutoff=self.encoder.time+c['gap']
        for batch in bounded_batches(events,c['chunk_size']):
            for event in batch:
                validate_event(event,labeled=labeled)
                if event['time']<=cutoff:raise ValueError('Evaluation must follow training with requested gap')
                if labeled and event['target']<0:raise ValueError('Nonnegative target required')
            # Validation labels never enter the encoder or estimator.
            inputs=[{k:v for k,v in event.items() if k!='target'} for event in batch]
            [(matrix,_)]=list(encoder.encode(inputs,chunk_size=c['chunk_size'],labeled=False))
            x=clipped(matrix,c['feature_clip'])
            logs=np.mean([m.predict(x) for m in self.models],axis=0)
            if not np.isfinite(logs).all():raise ValueError('Nonfinite log predictions')
            predictions=np.expm1(np.clip(logs,0,math.log1p(c['max_prediction'])))
            yield predictions,np.asarray([e['target'] for e in batch],dtype=np.float64) if labeled else None

    def evaluate(self,events):
        count=0;squared=0.
        for predicted,target in self.predict_batches(events,labeled=True):
            errors=np.log1p(predicted)-np.log1p(target)
            squared+=float(errors@errors);count+=len(errors)
        if not count or not math.isfinite(squared):raise ValueError('Nonempty finite validation required')
        return dict(rows=count,rmsle=math.sqrt(squared/count))


def fit(events,*,encoder_controls,controls):
    validate_controls(controls);c=copy.deepcopy(controls)
    encoder=TemporalSparseEncoder(**encoder_controls)
    models=[SGDRegressor(loss=loss,penalty='l2',alpha=c['alpha'],learning_rate='constant',eta0=c['learning_rate'],random_state=c['seed'],shuffle=False,average=False) for loss in ('squared_error','huber')]
    def checked():
        for event in events:
            validate_event(event,labeled=True)
            if event['target']<0:raise ValueError('Nonnegative targets required')
            yield event
    count=0;updates=0
    for matrix,target in encoder.encode(checked(),chunk_size=c['chunk_size'],labeled=True):
        x=clipped(matrix,c['feature_clip']);y=np.log1p(target)
        for model in models:
            model.partial_fit(x,y)
            if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():raise ValueError('Nonfinite online model')
        count+=len(target);updates+=1
    if not count:raise ValueError('Training stream empty')
    return StreamEnsemble(models,encoder,c,count,updates)
