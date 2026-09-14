"""DFDC per-video, class-balanced validation log loss over positional clip groups.

Source MIT2020SelimSeferbekov,89c6290490bac96b29193a4061b3db9dd3933e36;
docs/licenses/DFDC-MIT.txt. No media-name parsing or prediction-file output.
"""
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import log_loss

from sciona.dfdc_dataset import PreparedDataset


class ValidationDataset(PreparedDataset):
    def __init__(self,records,*,fold=0):
        super().__init__(records,mode='val',detector=None,predictor=None,fold=fold)

    def __getitem__(self,index):
        sample=super().__getitem__(index)
        sample['clip_position']=int(self.records[int(self.indices[index])]['clip_position'])
        return sample


def score_videos(probabilities, labels, groups):
    probabilities=np.asarray(probabilities);labels=np.asarray(labels);groups=np.asarray(groups)
    if (probabilities.ndim!=1 or labels.shape!=probabilities.shape or groups.shape!=probabilities.shape
            or probabilities.size==0 or probabilities.dtype.kind not in 'fiu'
            or not np.isfinite(probabilities).all() or np.any(probabilities<0) or np.any(probabilities>1)
            or not np.isin(labels,[0,1]).all() or groups.dtype.kind not in 'iu' or np.any(groups<0)):
        raise ValueError('validation needs finite probabilities, binary labels and nonnegative clip positions')
    scores=defaultdict(list);targets=defaultdict(list)
    for probability,label,group in zip(probabilities,labels,groups):
        scores[int(group)].append(float(probability));targets[int(group)].append(float(label))
    values=[];truth=[]
    for group,score in scores.items():
        if len(set(targets[group]))!=1:
            raise ValueError('a clip must have one consistent validation label')
        values.append(np.mean(np.array(score)));truth.append(np.mean(targets[group]))
    x=np.array(values);y=np.array(truth)
    fake=y>.1;real=y<.1
    if not fake.any() or not real.any():
        raise ValueError('validation requires both classes at clip level')
    fake_loss=log_loss(y[fake],x[fake],labels=[0,1])
    real_loss=log_loss(y[real],x[real],labels=[0,1])
    return {'loss':float((fake_loss+real_loss)/2),'fake_loss':float(fake_loss),
            'real_loss':float(real_loss),'clips':len(values),'samples':len(probabilities)}


def evaluate(model, loader):
    if model.training:
        raise ValueError('validation model must be in evaluation mode')
    probabilities=[];labels=[];groups=[]
    with torch.no_grad():
        for sample in loader:
            images=sample['image']
            outputs=model(images)
            if outputs.shape!=(len(images),1):
                raise ValueError('classifier validation output must be (N,1)')
            probabilities.extend(torch.sigmoid(outputs).cpu().numpy()[:,0].tolist())
            labels.extend(sample['labels'].float().cpu().numpy()[:,0].tolist())
            groups.extend(sample['clip_position'].cpu().tolist())
    return score_videos(probabilities,labels,groups)
