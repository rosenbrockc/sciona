"""Group-consistent out-of-fold stacking for a generic binary tabular model."""
from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sciona.tabular_ensemble_features import TabularFeatures,validate_table


def binary_labels(labels,rows):
    if not isinstance(labels,list) or len(labels)!=rows or any(type(v) is not int or v not in (0,1) for v in labels) or len(set(labels))!=2:
        raise ValueError('Expected aligned integer binary labels with both classes')
    return np.asarray(labels,dtype=np.int64)


def group_ids(groups,rows):
    if not isinstance(groups,list) or len(groups)!=rows or any(type(g) is not str or not g for g in groups):
        raise ValueError('Expected aligned nonempty string groups')
    return frozenset(groups)


@dataclass
class BaseModels:
    features: object
    linear: object
    trees: object

    def predict(self,numeric,categorical):
        x=self.features.transform(numeric,categorical)
        return np.column_stack((self.linear.predict_proba(x)[:,1],self.trees.predict_proba(x)[:,1]))


@dataclass
class StackedModel:
    base: BaseModels
    stacker: object
    training_groups: frozenset
    oof_predictions: object
    fold_models: dict
    fold_counts: list

    def predict(self,numeric,categorical):
        scores=self.stacker.predict_proba(self.base.predict(numeric,categorical))[:,1]
        if not np.isfinite(scores).all():raise ValueError('Nonfinite stacker output')
        return scores


def _fit_base(numeric,categorical,labels,controls):
    features=TabularFeatures().fit(numeric,categorical,clip_low=controls['clip_low'],clip_high=controls['clip_high'])
    x=features.transform(numeric,categorical)
    linear=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=controls['seed']))
    trees=ExtraTreesClassifier(n_estimators=controls['trees'],max_depth=controls['max_depth'],min_samples_leaf=controls['min_leaf'],random_state=controls['seed'],n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        linear.fit(x,labels)
    trees.fit(x,labels)
    return BaseModels(features,linear,trees)


def fit_stacker(numeric,categorical,labels,folds,groups,*,seed=42,trees=64,max_depth=6,min_leaf=1,clip_low=.01,clip_high=.99):
    validate_table(numeric,categorical)
    y=binary_labels(labels,len(numeric));identities=group_ids(groups,len(y))
    if not isinstance(folds,list) or len(folds)!=len(y) or any(type(f) is not int or f<0 for f in folds):raise ValueError('Invalid fold assignment')
    fold_set=set(folds)
    if len(fold_set)<2 or fold_set!=set(range(len(fold_set))):raise ValueError('Expected contiguous fold indices from zero')
    assignments={}
    for group,fold in zip(groups,folds):
        if group in assignments and assignments[group]!=fold:raise ValueError('Group crosses validation folds')
        assignments[group]=fold
    for name,value in [('seed',seed),('trees',trees),('max_depth',max_depth),('min_leaf',min_leaf)]:
        if type(value) is not int or value<(0 if name=='seed' else 1):raise ValueError('Invalid base-model control')
    if seed>=2**32:raise ValueError('Invalid random seed')
    controls=dict(seed=seed,trees=trees,max_depth=max_depth,min_leaf=min_leaf,clip_low=clip_low,clip_high=clip_high)
    folds=np.asarray(folds);splits=[]
    for fold in sorted(fold_set):
        train=np.flatnonzero(folds!=fold);valid=np.flatnonzero(folds==fold)
        if len(np.unique(y[train]))!=2:raise ValueError('Each fitting fold requires both classes')
        splits.append((fold,train,valid))
    oof=np.full((len(y),2),np.nan);counts=np.zeros(len(y),dtype=int);models={};audit=[]
    for fold,train,valid in splits:
        model=_fit_base([numeric[i] for i in train],[categorical[i] for i in train],y[train],controls)
        oof[valid]=model.predict([numeric[i] for i in valid],[categorical[i] for i in valid]);counts[valid]+=1
        models[fold]=model
        audit.append(dict(fold=fold,fit_rows=len(train),validation_rows=len(valid)))
    if not np.isfinite(oof).all() or not np.all(counts==1):raise ValueError('Incomplete out-of-fold predictions')
    stacker=LogisticRegression(max_iter=1000,random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        stacker.fit(oof,y)
    full=_fit_base(numeric,categorical,y,controls)
    return StackedModel(full,stacker,identities,oof,models,audit)
