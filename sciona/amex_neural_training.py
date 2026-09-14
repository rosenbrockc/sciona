"""Independent CPU lifecycle for both Amex five-fold neural variants."""
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import StratifiedKFold
from sciona.amex_sequence import pack_neural
from sciona.amex_network import Network
from sciona.amex_metric import score


def fit(series,features,labels,query_series,query_features,*,seed=42):
    train=pack_neural(series,features);query=pack_neural(query_series,query_features)
    if train['series'].shape[2]!=query['series'].shape[2] or train['features'].shape[1]!=query['features'].shape[1]:raise ValueError('Aligned neural feature widths required')
    if type(labels) is not list or len(labels)!=len(series) or any(type(y) is not int or y not in (0,1) for y in labels):raise ValueError('Aligned integer binary labels required')
    y=np.asarray(labels)
    if min(np.bincount(y,minlength=2))<5 or type(seed) is not int or not 0<=seed<2**31:raise ValueError('Valid class population and seed required')
    plans=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
    if any(len(tr)<256 for tr,va in plans):raise ValueError('Each fold must contain a full 256-row training batch')
    tensors=[torch.from_numpy(train[k]) for k in ('series','mask','features')]
    qt=[torch.from_numpy(query[k]) for k in ('series','mask','features')]
    target=torch.tensor(y,dtype=torch.float32)[:,None];variants={}
    def predict(model,inputs):
        model.eval()
        with torch.no_grad():return np.concatenate([model(*(v[i:i+256] for v in inputs)).numpy().reshape(-1) for i in range(0,len(inputs[0]),256)])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for combined,name in ((False,'series'),(True,'combined')):
            oof=np.full(len(y),np.nan);pred=np.zeros(len(query_series));records=[]
            for tr,va in plans:
                model=Network(tensors[0].shape[2],tensors[2].shape[1],combined=combined)
                optimizer=torch.optim.Adam(model.parameters(),lr=.001);criterion=nn.BCELoss()
                loader=DataLoader(TensorDataset(*(v[tr] for v in tensors),target[tr]),batch_size=256,shuffle=True,drop_last=True)
                best=0.;state=None;selected_epoch=None;history=[]
                for epoch in range(10):
                    lr=.001 if epoch<=4 else .0001 if epoch<=8 else .00001
                    for group in optimizer.param_groups:group['lr']=lr
                    model.train();total=0.;rows=0
                    for x,m,f,t in loader:
                        optimizer.zero_grad();loss=criterion(model(x,m,f),t)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                        loss.backward();optimizer.step();total+=float(loss.detach())*len(t);rows+=len(t)
                    probabilities=predict(model,[v[va] for v in tensors]);metric=score(y[va],probabilities)
                    if metric>best:best=metric;state=copy.deepcopy(model.state_dict());selected_epoch=epoch
                    history.append(dict(epoch=epoch,learning_rate=lr,training_rows=rows,loss=total/rows,metric=metric))
                if state is None:raise ValueError('No source-eligible positive-metric checkpoint')
                restored=Network(tensors[0].shape[2],tensors[2].shape[1],combined=combined);restored.load_state_dict(state)
                oof[va]=predict(restored,[v[va] for v in tensors]);pred+=predict(restored,qt)/5
                if not np.isclose(score(y[va],oof[va]),best,rtol=0,atol=1e-12):raise ValueError('Checkpoint readback differs')
                records.append(dict(selected_epoch=selected_epoch,best_metric=best,history=history))
            if not np.isfinite(oof).all() or not np.isfinite(pred).all():raise ValueError('Incomplete neural predictions')
            variants[name]=dict(training_predictions=oof,query_predictions=pred,models=5,folds=records)
    return dict(variants=variants,models=10,epochs=10,hidden_width=128,
        scope='Independent CPU non-AMP lifecycle: raw BCE, drop-last batches, effective Adam schedule and strict positive-metric checkpoint selection. Historical RNG/GPU parity unqualified.')
