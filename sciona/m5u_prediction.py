"""Independent source out-of-sample fold weighting, bag mean and inverse scale."""
import numpy as np
import pandas as pd


def predict(models,frame,scales,quantiles,*,validation=False):
    scales=np.asarray(scales)
    if (not isinstance(frame,pd.DataFrame) or frame.empty or scales.shape!=(len(frame),)
            or scales.dtype.kind not in 'iuf' or not np.isfinite(scales).all() or (scales<=0).any()
            or type(validation) is not bool or not models or not isinstance(quantiles,(list,tuple))
            or not quantiles or len(set(quantiles))!=len(quantiles)):
        raise ValueError('Aligned query features, positive scales and unique quantiles required')
    for q in quantiles:
        if isinstance(q,bool) or not isinstance(q,(int,float)) or not 0<q<1:raise ValueError('Invalid quantile')
    bags=sorted({m.bag for m in models});groups=sorted({m.group for m in models})
    lookup={(m.bag,m.group,m.quantile):m for m in models}
    expected={(b,g,q) for b in bags for g in groups for q in quantiles}
    if len(lookup)!=len(models) or set(lookup)!=expected:raise ValueError('Complete unique bag/group/quantile inventory required')
    weights=np.arange(1,len(groups)+1,dtype=float)
    if validation:weights[:]=0;weights[-1]=1
    weights/=weights.sum()
    output=[]
    for q in quantiles:
        bag_predictions=[]
        for bag in bags:
            folded=np.zeros(len(frame))
            for group,weight in zip(groups,weights):
                if not weight:continue
                model=lookup[(bag,group,q)].model
                if tuple(model.feature_name_)!=tuple(frame.columns):raise ValueError('Query feature order differs from fitted model')
                predicted=np.asarray(model.predict(frame),dtype=float)
                if predicted.shape!=(len(frame),) or not np.isfinite(predicted).all():raise ValueError('Invalid quantile predictions')
                folded+=predicted*weight
            bag_predictions.append(folded)
        output.append(np.mean(bag_predictions,axis=0)*scales)
    result=np.stack(output)
    if not np.isfinite(result).all():raise ValueError('Inverse scaling overflow')
    return result
