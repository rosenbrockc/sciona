"""Forward-validated ExtraTrees fusion with fold-local target history."""
import copy
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sciona.geotemporal_features import rows,feature_matrix
from sciona.geotemporal_validation import forward_splits
from sciona.geotemporal_postprocess import smooth


def points(observations):return [{k:r[k] for k in ('x','y','time')} for r in observations]


def controls_valid(c):
    if type(c) is not dict or set(c)!={'radius','lookback','period','neighbors','gap','seed','trees','max_depth','min_leaf','smoothing'}:raise ValueError('Invalid controls')
    for name in ('seed','trees','max_depth','min_leaf','neighbors'):
        if type(c[name]) is not int or c[name]<(0 if name=='seed' else 1):raise ValueError('Invalid integer control')
    if c['seed']>=2**32:raise ValueError('Invalid seed')
    for name in ('radius','lookback','period','gap','smoothing'):
        if type(c[name]) not in (int,float) or not np.isfinite(c[name]) or c[name]<(0 if name in ('gap','smoothing') else np.nextafter(0.,1.)):raise ValueError('Invalid numeric control')
    if c['smoothing']>1:raise ValueError('Smoothing exceeds one')


def features(query,context,history,c):
    return feature_matrix(query,context,history,**{k:c[k] for k in ('radius','lookback','period','neighbors')})


def model(c):
    return ExtraTreesRegressor(n_estimators=c['trees'],max_depth=c['max_depth'],min_samples_leaf=c['min_leaf'],random_state=c['seed'],n_jobs=1)


@dataclass
class GeoModel:
    estimator: object
    context: list
    history: list
    controls: dict
    folds: list
    validation_predictions: list
    validation_mse: float

    def predict(self,query):
        rows(query,())
        if min(r['time'] for r in query)<=max(r['time'] for r in self.history)+self.controls['gap']:raise ValueError('Query must follow training with required gap')
        raw=self.estimator.predict(features(query,self.context,self.history,self.controls))
        return smooth(query,raw,radius=self.controls['radius'],strength=self.controls['smoothing'])


def fit(observations,context,blocks,controls):
    rows(observations,('target',));rows(context,('value',));controls_valid(controls)
    if any(r['target']<0 for r in observations):raise ValueError('Nonnegative targets required')
    splits=forward_splits([r['time'] for r in observations],blocks,gap=controls['gap'])
    # Validate duplicate identities and feature controls for the full population.
    features(points(observations),context,observations,controls)
    predictions=[None]*len(observations);audits=[];errors=[]
    for train,valid in splits:
        history=[observations[int(i)] for i in train]
        heldout=[observations[int(i)] for i in valid]
        x=features(points(history),context,history,controls)
        vx=features(points(heldout),context,history,controls)
        estimator=model(controls).fit(x,[r['target'] for r in history])
        # Smoothing only sees co-timed held-out predictions, never labels.
        predicted=smooth(points(heldout),estimator.predict(vx),radius=controls['radius'],strength=controls['smoothing'])
        for index,value in zip(valid,predicted):predictions[int(index)]=float(value)
        errors.extend((np.asarray(predicted)-[r['target'] for r in heldout])**2)
        audits.append(dict(train_indices=train.tolist(),validation_indices=valid.tolist(),estimator=estimator,training_features=x,validation_features=vx,predictions=predicted))
    full=model(controls).fit(features(points(observations),context,observations,controls),[r['target'] for r in observations])
    mse=float(np.mean(errors))
    if not np.isfinite(mse):raise ValueError('Nonfinite validation metric')
    return GeoModel(full,copy.deepcopy(context),copy.deepcopy(observations),copy.deepcopy(controls),audits,predictions,mse)
