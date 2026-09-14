"""Independent masked attention/context-gated multilabel video head."""
import torch
from torch import nn


class VideoHead(nn.Module):
    """Shared learned frame attention, context gate, then one logit per label.

Inputs are precomputed frame features; no raw-video feature extractor or temporal
position encoding is implied. Attention pooling is invariant to frame order.
"""
    def __init__(self,features,hidden,labels):
        super().__init__()
        if any(type(v) is not int or v<1 for v in (features,hidden,labels)):raise ValueError('Positive dimensions required')
        self.features=features
        self.attention=nn.Sequential(nn.Linear(features,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.context=nn.Linear(features,features)
        self.classifier=nn.Linear(features,labels)

    def pool(self,frames,mask):
        if frames.ndim!=3 or frames.shape[2]!=self.features or mask.shape!=frames.shape[:2] or mask.dtype!=torch.bool or not frames.is_floating_point():
            raise ValueError('Expected batch/time/feature tensor and boolean mask')
        if not frames.shape[0] or not frames.shape[1] or not mask.any(1).all():raise ValueError('Each video requires at least one frame')
        if not torch.isfinite(frames[mask]).all():raise ValueError('Nonfinite valid frame')
        clean=torch.where(mask[:,:,None],frames,torch.zeros_like(frames))
        attention=self.attention(clean).squeeze(-1).masked_fill(~mask,float('-inf')).softmax(1)
        return (clean*attention[:,:,None]).sum(1),attention

    def forward(self,frames,mask):
        pooled,_=self.pool(frames,mask)
        return self.classifier(pooled*torch.sigmoid(self.context(pooled)))


def pad_sequences(sequences):
    """Copy finite, nonempty variable-length frame matrices to CPU float64."""
    if type(sequences) is not list or not sequences:raise ValueError('Nonempty video list required')
    width=None;arrays=[]
    for sequence in sequences:
        if type(sequence) is not list or not sequence:raise ValueError('Video has no frames')
        for row in sequence:
            if type(row) is not list or not row or any(type(v) not in (int,float) for v in row):raise ValueError('Numeric frame vectors required')
            if width is None:width=len(row)
            if len(row)!=width:raise ValueError('Inconsistent frame feature width')
        try:array=torch.tensor(sequence,dtype=torch.float64)
        except (ValueError,RuntimeError,OverflowError) as error:raise ValueError('Invalid frame values') from error
        if not torch.isfinite(array).all():raise ValueError('Nonfinite frame values')
        arrays.append(array)
    length=max(len(a) for a in arrays)
    frames=torch.zeros(len(arrays),length,width,dtype=torch.float64)
    mask=torch.zeros(len(arrays),length,dtype=torch.bool)
    for i,array in enumerate(arrays):
        frames[i,:len(array)]=array;mask[i,:len(array)]=True
    return frames,mask
