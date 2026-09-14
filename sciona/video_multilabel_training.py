"""Batch-padded attention training and training-vocabulary sparse baseline fusion."""
from dataclasses import dataclass
import warnings
import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sciona.video_multilabel_network import VideoHead,pad_sequences
from sciona.video_multilabel_thresholds import label_matrix,score_matrix


def validate_features(videos,sparse):
    if type(videos) is not list or not videos or type(sparse) is not list or len(sparse)!=len(videos):raise ValueError('Aligned nonempty video and sparse populations required')
    width=None
    for video in videos:
        frames,_=pad_sequences([video])
        if width is None:width=frames.shape[2]
        if width!=frames.shape[2]:raise ValueError('Frame widths differ')
    for row in sparse:
        if type(row) is not dict or any(type(k) is not str or not k or type(v) not in (int,float) for k,v in row.items()):raise ValueError('Sparse string-to-number mappings required')
        try:values=np.asarray(list(row.values()),dtype=np.float64)
        except (ValueError,OverflowError) as error:raise ValueError('Invalid sparse values') from error
        if not np.isfinite(values).all() or (values<0).any():raise ValueError('Finite nonnegative sparse values required')
    return width


@dataclass
class VideoEnsemble:
    neural: object
    vectorizer: object
    baseline: list
    batch_size: int
    history: list

    def components(self,videos,sparse):
        width=validate_features(videos,sparse)
        if width!=self.neural.features:raise ValueError('Prediction feature width differs')
        neural=[]
        self.neural.eval()
        with torch.no_grad():
            for start in range(0,len(videos),self.batch_size):
                frames,mask=pad_sequences(videos[start:start+self.batch_size])
                neural.append(torch.sigmoid(self.neural(frames,mask)).numpy())
        features=self.vectorizer.transform(sparse)
        linear=np.column_stack([model.predict_proba(features)[:,1] for model in self.baseline])
        return score_matrix(np.concatenate(neural)),score_matrix(linear)

    def predict(self,videos,sparse):
        neural,linear=self.components(videos,sparse)
        return score_matrix((neural+linear)/2)


def fit(videos,sparse,labels,*,seed=12,hidden=8,epochs=80,batch_size=4,learning_rate=.02):
    width=validate_features(videos,sparse)
    if type(labels) is not list or not labels or type(labels[0]) is not list:raise ValueError('Label matrix required')
    y=label_matrix(labels,(len(videos),len(labels[0])))
    if not y.shape[1]:raise ValueError('At least one label required')
    for name,value in [('seed',seed),('hidden',hidden),('epochs',epochs),('batch_size',batch_size)]:
        if type(value) is not int or value<(0 if name=='seed' else 1):raise ValueError('Invalid training control')
    if seed>=2**32:raise ValueError('Invalid seed')
    if type(learning_rate) not in (int,float) or not np.isfinite(learning_rate) or learning_rate<=0:raise ValueError('Invalid learning rate')
    vectorizer=DictVectorizer(sparse=True,sort=True)
    features=vectorizer.fit_transform(sparse)
    if not features.shape[1]:raise ValueError('Training sparse vocabulary is empty')
    baseline=[]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for column in y.T:
            model=LogisticRegression(max_iter=1000,random_state=seed)
            model.fit(features,column);baseline.append(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=VideoHead(width,hidden,y.shape[1]).double()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    target=torch.tensor(y,dtype=torch.float64);rng=np.random.default_rng(seed);history=[]
    for epoch in range(epochs+1):
        model.eval();total=0.
        with torch.no_grad():
            for start in range(0,len(videos),batch_size):
                frames,mask=pad_sequences(videos[start:start+batch_size])
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(frames,mask),target[start:start+batch_size])
                total+=float(loss)*len(frames)
        loss_value=total/len(videos)
        if not np.isfinite(loss_value):raise ValueError('Nonfinite training loss')
        history.append(loss_value)
        if epoch==epochs:break
        model.train();order=rng.permutation(len(videos))
        for start in range(0,len(videos),batch_size):
            indices=order[start:start+batch_size]
            frames,mask=pad_sequences([videos[int(i)] for i in indices])
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(frames,mask),target[indices])
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Invalid neural gradients')
            optimizer.step()
    model.eval()
    return VideoEnsemble(model,vectorizer,baseline,batch_size,history)
