"""Source CHAMPS population split and explicit deterministic batch streams.

The seed-zero split uses the first 60% and last 20% for training and the middle
20% for validation. Source DataLoader shuffle/drop-last semantics are retained;
a caller-owned generator replaces the launcher's process-global RNG boundary.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset


def _validate(packed):
    if len(packed)!=10 or any(not isinstance(t,torch.Tensor) or t.ndim<1 for t in packed):
        raise ValueError('CHAMPS population requires ten batched tensors')
    count=packed[0].shape[0]
    if count==0 or any(t.shape[0]!=count for t in packed):
        raise ValueError('CHAMPS population must be nonempty and aligned')
    return count


def split_population(packed):
    count=_validate(packed)
    order=np.random.RandomState(0).permutation(count)
    train=torch.tensor(np.concatenate((order[:int(.6*count)],order[int(.8*count):])),dtype=torch.long)
    validation=torch.tensor(order[int(.6*count):int(.8*count)],dtype=torch.long)
    return tuple(t[train] for t in packed),tuple(t[validation] for t in packed)


def population_batches(packed,*,batch_size:int,shuffle:bool,drop_last:bool,seed:int):
    _validate(packed)
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<=0:
        raise ValueError('CHAMPS batch size must be a positive integer')
    if not isinstance(shuffle,bool) or not isinstance(drop_last,bool):
        raise ValueError('CHAMPS shuffle/drop-last settings must be boolean')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:
        raise ValueError('CHAMPS batch seed must be unsigned 32-bit')
    return DataLoader(TensorDataset(*packed),batch_size=batch_size,shuffle=shuffle,
                      drop_last=drop_last,generator=torch.Generator().manual_seed(seed),num_workers=0)
