"""Independent joint-vocabulary categorical sequence summaries for Amex.

Inputs are ordered sequences of numeric category codes. NaN denotes missing;
joint vocabulary excludes missing values while their one-hot rows remain zero.
Caller supplies explicit time-presence masks; no private schema is embedded.
"""
import numpy as np
from sciona.amex_numeric import _matrix


def summarize_populations(sequences,time_present,*,windowed=False):
    if type(windowed) is not bool or type(sequences) is not list or not sequences:raise ValueError('Explicit sequence population required')
    xs=[_matrix(x) for x in sequences];width=xs[0].shape[1]
    if any(x.shape[1]!=width for x in xs) or len(time_present)!=len(xs):raise ValueError('Aligned sequence widths and time masks required')
    masks=[]
    for x,mask in zip(xs,time_present):
        a=np.asarray(mask)
        if a.shape!=(len(x),) or a.dtype.kind!='b':raise ValueError('Boolean time-presence masks required')
        masks.append(a)
    population=np.vstack(xs)
    vocabulary=[np.unique(population[~np.isnan(population[:,j]),j]) for j in range(width)]
    result=[]
    for x,mask in zip(xs,masks):
        row=[]
        for column,categories in enumerate(vocabulary):
            for category in categories:
                hot=(x[:,column]==category).astype(float)
                row.extend([hot.mean(),hot.std(ddof=1) if len(hot)>1 else np.nan,hot.sum()])
                if not windowed:row.append(hot[-1])
        for column in range(width):
            values=x[:,column];valid=values[~np.isnan(values)]
            if not windowed:row.append(valid[-1] if len(valid) else np.nan)
            row.append(len(np.unique(valid)))
        row.append(mask.sum());result.append(row)
    return dict(values=np.asarray(result,dtype=float),vocabulary=vocabulary)
