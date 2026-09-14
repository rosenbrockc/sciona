"""Independent orchestration of all nine Amex manual-feature blocks.

Inputs are already denoised numeric values and mapped numeric category codes.
Sequences are ordered customers across the joint training/query population.
Explicit month/time codes and zero-fill column roles replace private schemas.
Blocks retain source internal order; final learner assembly is a later stage.
"""
import numpy as np
from sciona.amex_numeric import _matrix,summarize
from sciona.amex_categorical import summarize_populations
from sciona.amex_ranks import percentile_ranks,latest_window


def build(numerics,categories,times,months,*,zero_fill_columns):
    if type(numerics) is not list or not numerics or not len(numerics)==len(categories)==len(times)==len(months):raise ValueError('Aligned sequence populations required')
    xs=[_matrix(x) for x in numerics];cs=[_matrix(c) for c in categories]
    width=xs[0].shape[1];cat_width=cs[0].shape[1]
    if type(zero_fill_columns) is not list or any(type(i) is not int or not 0<=i<width for i in zero_fill_columns) or len(set(zero_fill_columns))!=len(zero_fill_columns):raise ValueError('Distinct valid fill columns required')
    ts=[];ms=[]
    for x,c,t,m in zip(xs,cs,times,months):
        t=np.asarray(t);m=np.asarray(m)
        if x.shape[1]!=width or c.shape!=(len(x),cat_width) or t.shape!=(len(x),) or m.shape!=t.shape or t.dtype.kind not in 'iuf' or not np.isfinite(t).all() or m.dtype.kind not in 'iu':raise ValueError('Aligned numeric category and time roles required')
        for j in zero_fill_columns:x[np.isnan(x[:,j]),j]=0.
        ts.append(t);ms.append(m)
    lengths=[len(x) for x in xs];groups=np.repeat(np.arange(len(xs)),lengths)
    flat=np.vstack(xs);customer_rank=percentile_ranks(flat,groups);month_rank=percentile_ranks(flat,np.concatenate(ms))
    cuts=np.cumsum(lengths)[:-1]
    blocks=dict(full_categorical=summarize_populations(cs,[np.ones(n,bool) for n in lengths])['values'],
        full_numeric=np.stack([summarize(x) for x in xs]),full_difference=np.stack([summarize(x,differences=True) for x in xs]),
        customer_rank=np.stack([summarize(x,ranked=True) for x in np.split(customer_rank,cuts)]),
        month_rank=np.stack([summarize(x) for x in np.split(month_rank,cuts)]))
    for count in (3,6):
        selected=np.split(latest_window(np.concatenate(ts),groups,count=count),cuts)
        active=[i for i,mask in enumerate(selected) if mask.any()]
        for mode,difference in [('numeric',False)]+([('difference',True)] if count==3 else []):
            out=np.full((len(xs),width*5),np.nan)
            for i in active:
                values=xs[i][selected[i]]
                out[i]=summarize(values,last_window=len(values),differences=difference)
            blocks[f'last{count}_{mode}']=out
        if count==3:
            if active:
                result=summarize_populations([cs[i][selected[i]] for i in active],[np.ones(selected[i].sum(),bool) for i in active],windowed=True)['values']
                out=np.full((len(xs),result.shape[1]),np.nan);out[active]=result
            else:out=np.full((len(xs),cat_width+1),np.nan)
            blocks['last3_categorical']=out
    return blocks
