"""Independent Amex packed-GRU and optional tabular neural architectures."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class Network(nn.Module):
    def __init__(self,series_width,feature_width,*,combined=False,hidden_width=128):
        super().__init__()
        if any(type(n) is not int or n<1 for n in (series_width,feature_width,hidden_width)) or type(combined) is not bool:raise ValueError('Valid explicit network dimensions required')
        self.combined=combined;self.series_width=series_width;self.feature_width=feature_width
        self.sequence_projection=nn.Sequential(nn.Linear(series_width,hidden_width),nn.LayerNorm(hidden_width))
        self.feature_projection=nn.Sequential(nn.Linear(feature_width,hidden_width),nn.BatchNorm1d(hidden_width),nn.LeakyReLU())
        self.recurrence=nn.GRU(hidden_width,hidden_width,batch_first=True,bidirectional=True)
        layers=[]
        for _ in range(2):layers.extend([nn.Linear(hidden_width,hidden_width),nn.BatchNorm1d(hidden_width),nn.Dropout(.5),nn.LeakyReLU()])
        self.feature_hidden=nn.Sequential(*layers)
        self.output=nn.Sequential(nn.Linear((3 if combined else 2)*hidden_width,hidden_width),nn.LeakyReLU(),nn.Linear(hidden_width,hidden_width),nn.LeakyReLU(),nn.Linear(hidden_width,1),nn.Sigmoid())

    def pool(self,projected,mask):
        lengths=mask.sum(dim=1).to(dtype=torch.int64,device='cpu')
        packed=pack_padded_sequence(projected,lengths,batch_first=True,enforce_sorted=False)
        result,_=self.recurrence(packed)
        unpacked,_=pad_packed_sequence(result,batch_first=True)
        return unpacked[torch.arange(len(lengths),device=unpacked.device),lengths.to(unpacked.device)-1]

    def forward(self,series,mask,features):
        if series.ndim!=3 or series.shape[2]!=self.series_width or not len(series) or mask.shape!=series.shape[:2] or features.shape!=(len(series),self.feature_width):raise ValueError('Aligned neural tensor shapes required')
        if not torch.isfinite(series).all() or not torch.isfinite(features).all() or not torch.isfinite(mask).all() or not ((mask==0)|(mask==1)).all():raise ValueError('Finite neural inputs and binary mask required')
        lengths=mask.sum(1)
        expected=torch.arange(mask.shape[1],device=mask.device)[None,:]<lengths[:,None]
        if (lengths<1).any() or not (mask==expected).all():raise ValueError('Nonempty prefix sequence mask required')
        sequence=self.pool(self.sequence_projection(series),mask)
        if self.combined:sequence=torch.cat((sequence,self.feature_hidden(self.feature_projection(features))),dim=1)
        return self.output(sequence)
