"""Independent supervised CPU network on Porto autoencoder activations only."""
import numpy as np
import torch
from sciona.porto_transforms import _matrix


class Classifier(torch.nn.Module):
    def __init__(self,width,controls):
        super().__init__();hidden=controls['hidden']
        self.layers=torch.nn.ModuleList(torch.nn.Linear(a,b) for a,b in zip([width]+hidden[:-1],hidden))
        self.output=torch.nn.Linear(hidden[-1],1)
        self.input_dropout=torch.nn.Dropout(controls['input_dropout'])
        self.hidden_dropout=torch.nn.Dropout(controls['dropout'])
        self.scaling=controls['dropout_scaling']

    def drop(self,x,layer):
        value=layer(x)
        if self.training and self.scaling=='unscaled':value=value*(1-layer.p)
        return value

    def forward(self,x):
        x=self.drop(x,self.input_dropout)
        for layer in self.layers:x=self.drop(torch.relu(layer(x)),self.hidden_dropout)
        return self.output(x).squeeze(1)


def fit_predict(reference,labels,query,*,seed,controls):
    fields={'hidden','epochs','batch_size','learning_rate','decay','l2','momentum','dropout','input_dropout','dropout_scaling'}
    if type(controls) is not dict or set(controls)!=fields:raise ValueError('Complete explicit neural controls required')
    if type(controls['hidden']) is not list or not controls['hidden'] or any(type(n) is not int or n<1 for n in controls['hidden']):raise ValueError('Positive hidden widths required')
    for key in ('epochs','batch_size'):
        if type(controls[key]) is not int or controls[key]<1:raise ValueError('Positive integer training controls required')
    for key in ('learning_rate','decay'):
        if type(controls[key]) not in (int,float) or not np.isfinite(controls[key]) or not 0<controls[key]<=1:raise ValueError('Positive bounded learning controls required')
    for key in ('dropout','input_dropout','momentum'):
        if type(controls[key]) not in (int,float) or not np.isfinite(controls[key]) or not 0<=controls[key]<1:raise ValueError('Dropout/momentum outside range')
    if type(controls['l2']) not in (int,float) or not np.isfinite(controls['l2']) or controls['l2']<0 or controls['dropout_scaling'] not in ('inverted','unscaled'):raise ValueError('Explicit L2/dropout semantics required')
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Bounded integer seed required')
    x=_matrix(reference);q=_matrix(query);y=np.asarray(labels)
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!={0,1}:raise ValueError('Aligned binary training population and query width required')
    with np.errstate(over='ignore'):x=x.astype(np.float32);q=q.astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('Finite float32 neural inputs required')
    train=torch.from_numpy(x);target=torch.from_numpy(y.astype(np.float32));history=[]
    rng=np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=Classifier(x.shape[1],controls)
        optimizer=torch.optim.SGD(model.parameters(),lr=controls['learning_rate'],momentum=controls['momentum'],weight_decay=controls['l2'])
        for epoch in range(controls['epochs']):
            model.train();total=0.
            order=rng.permutation(len(x))
            for start in range(0,len(x),controls['batch_size']):
                rows=order[start:start+controls['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(train[rows]),target[rows])
                if not torch.isfinite(loss):raise ValueError('Nonfinite classifier loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite classifier gradient')
                optimizer.step();total+=float(loss.detach())*len(rows)
            history.append(total/len(x))
            for group in optimizer.param_groups:group['lr']*=controls['decay']
        model.eval()
        with torch.no_grad():scores=model(torch.from_numpy(q)).sigmoid().numpy().copy()
    if scores.shape!=(len(q),) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():raise ValueError('Invalid neural probabilities')
    return dict(probabilities=scores,training_loss=history)
