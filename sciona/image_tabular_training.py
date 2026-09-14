"""Group-consistent fold-local preprocessing and dropout MLP ensemble."""
from dataclasses import dataclass
import numpy as np
import torch
from sciona.image_tabular_features import FusedFeatures,image_features,metadata
from sciona.image_tabular_network import HybridHead,smoothed_loss


def validate_split(labels,folds,groups,count):
    if type(labels) is not list or len(labels)!=count or any(type(v) is not int or v not in (0,1) for v in labels) or set(labels)!={0,1}:raise ValueError('Aligned both-class binary labels required')
    if type(folds) is not list or len(folds)!=count or any(type(v) is not int or v<0 for v in folds):raise ValueError('Aligned integer folds required')
    selected=set(folds)
    if len(selected)<2 or selected!=set(range(len(selected))):raise ValueError('Contiguous folds from zero required')
    if type(groups) is not list or len(groups)!=count or any(type(g) is not str or not g for g in groups):raise ValueError('Aligned group identifiers required')
    assignments={}
    for group,fold in zip(groups,folds):
        if group in assignments and assignments[group]!=fold:raise ValueError('Group crosses folds')
        assignments[group]=fold
    for fold in selected:
        if {v for v,f in zip(labels,folds) if f!=fold}!={0,1}:raise ValueError('Every fitting fold needs both classes')


def validate_controls(c):
    if type(c) is not dict or set(c)!={'seed','hidden','epochs','learning_rate','dropout','smoothing'}:raise ValueError('Invalid controls')
    for name in ('seed','hidden','epochs'):
        if type(c[name]) is not int or c[name]<(0 if name=='seed' else 1):raise ValueError('Invalid integer controls')
    if c['seed']>=2**32:raise ValueError('Invalid seed')
    for name in ('learning_rate','dropout','smoothing'):
        if type(c[name]) not in (int,float) or not np.isfinite(c[name]):raise ValueError('Finite numeric controls required')
    if c['learning_rate']<=0 or not 0<=c['dropout']<1 or not 0<=c['smoothing']<1:raise ValueError('Invalid learning/dropout/smoothing controls')


@dataclass
class FoldModel:
    features: object
    network: object
    history: list

    def predict(self,images,numeric,categorical):
        x=torch.tensor(self.features.transform(images,numeric,categorical),dtype=torch.float64)
        self.network.eval()
        with torch.no_grad():result=torch.sigmoid(self.network(x)).numpy()
        if not np.isfinite(result).all():raise ValueError('Nonfinite predictions')
        return result


@dataclass
class HybridEnsemble:
    models: list
    oof: object
    training_groups: frozenset

    def predict(self,images,numeric,categorical):
        return np.mean([m.predict(images,numeric,categorical) for m in self.models],axis=0)


def fit(images,numeric,categorical,labels,folds,groups,controls):
    image_features(images);metadata(numeric,categorical,len(images))
    validate_split(labels,folds,groups,len(images));validate_controls(controls)
    c=controls;models=[];oof=np.full(len(images),np.nan)
    for fold in sorted(set(folds)):
        selected=[i for i,f in enumerate(folds) if f!=fold];heldout=[i for i,f in enumerate(folds) if f==fold]
        def subset(values,indices):return [values[i] for i in indices]
        prep=FusedFeatures().fit(subset(images,selected),subset(numeric,selected),subset(categorical,selected))
        x=torch.tensor(prep.transform(subset(images,selected),subset(numeric,selected),subset(categorical,selected)),dtype=torch.float64)
        y=torch.tensor(subset(labels,selected),dtype=torch.float64)
        # Isolate initialization AND dropout draws; held-out data cannot change
        # the fitting trajectory, and the caller's CPU random stream is restored.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((c['seed']+fold)%2**32)
            net=HybridHead(x.shape[1],c['hidden'],c['dropout']).double()
            optimizer=torch.optim.Adam(net.parameters(),lr=c['learning_rate']);history=[]
            for epoch in range(c['epochs']+1):
                net.eval()
                with torch.no_grad():value=float(smoothed_loss(net(x),y,smoothing=c['smoothing']))
                if not np.isfinite(value):raise ValueError('Nonfinite training loss')
                history.append(value)
                if epoch==c['epochs']:break
                net.train();optimizer.zero_grad(set_to_none=True)
                smoothed_loss(net(x),y,smoothing=c['smoothing']).backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('Invalid gradients')
                optimizer.step()
        net.eval();model=FoldModel(prep,net,history);models.append(model)
        oof[heldout]=model.predict(subset(images,heldout),subset(numeric,heldout),subset(categorical,heldout))
    if not np.isfinite(oof).all():raise ValueError('Incomplete OOF coverage')
    return HybridEnsemble(models,oof,frozenset(groups))
