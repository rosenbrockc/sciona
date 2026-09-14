"""Runtime feeder conversion and ordered window batching for Web Traffic.

MIT source adaptation; see docs/licenses/WebTraffic-MIT.txt. Sampling offsets are
provided explicitly, so TensorFlow RNG/prefetch timing is not implied.
"""
import itertools
import numpy as np
from .webtraffic_windows import cut_window,make_window_features,keep_window,training_offset_bounds


def feeder_arrays(tensors):
    result={}
    for name,value in tensors.items():
        value=value.values if hasattr(value,'values') else value
        value=np.asarray(value)
        result[name]=value.astype(np.float32) if value.dtype==np.float64 else value.copy()
    return result


def fake_split_indices(total_pages, permutation, sampling=1.):
    """Source samples the prefix population before shuffling; caller provides order."""
    n_pages=int(round(total_pages*sampling))
    indices=np.asarray(permutation)
    if indices.dtype.kind not in 'iu' or not np.array_equal(np.sort(indices),np.arange(n_pages)):
        raise ValueError('Explicit permutation of source sampled prefix required')
    return indices.copy(),n_pages,total_pages


def window_batches(tensors,indices,starts,*,data_days,train_window=283,predict_window=63,
                   batch_size=256,epochs=None,training=True,back_offset=0,start_offset=0,
                   completeness=1.):
    """Repeat -> cut -> training-zero filter -> features -> batch, including final tail."""
    if batch_size<1 or (epochs is not None and (type(epochs) is not int or epochs<0)):
        raise ValueError('Positive batch size and optional nonnegative epoch count required')
    if not 0<=completeness<=1:raise ValueError('Completeness must be in [0,1]')
    arrays=feeder_arrays(tensors);offsets=iter(starts);pending=[]
    max_empty=int(round(train_window*(1-completeness)))
    low,high=training_offset_bounds(data_days,train_window,predict_window,back_offset,start_offset)
    for _ in itertools.count() if epochs is None else range(epochs):
        for index in indices:
            if training:
                try:start=next(offsets)
                except StopIteration as error:raise ValueError('Training offset stream exhausted') from error
                if not low<=start<high:raise ValueError('Training offset outside source bounds')
            else:start=data_days-train_window-back_offset
            cut=cut_window(arrays['hits'][index],arrays['dow'],arrays['lagged_ix'],start,train_window,predict_window)
            if not keep_window(cut[0],max_empty):continue
            record=make_window_features(*cut,arrays['pf_agent'][index],arrays['pf_country'][index],
                                        arrays['pf_site'][index],arrays['page_ix'][index],arrays['page_popularity'][index],
                                        arrays['year_autocorr'][index],arrays['quarter_autocorr'][index])
            pending.append(record)
            if len(pending)==batch_size:
                yield tuple(np.stack(column) for column in zip(*pending));pending=[]
    if pending:yield tuple(np.stack(column) for column in zip(*pending))
