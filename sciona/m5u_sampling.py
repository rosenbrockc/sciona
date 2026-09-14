"""Independent weighted replication and quantile-dependent training masks."""
import numpy as np


def sample(weights,levels,level,fraction,*,repeats=1,seed=0):
    weights,levels=np.asarray(weights),np.asarray(levels)
    if (weights.ndim!=1 or not weights.size or weights.dtype.kind not in 'iuf'
            or levels.shape!=weights.shape or levels.dtype.kind not in 'iu'
            or not np.isfinite(weights).all() or (weights<0).any()):
        raise ValueError('Aligned finite nonnegative weights and integer levels required')
    if type(level) is not int or type(repeats) is not int or repeats<1 or type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Invalid level, repeats or seed')
    if isinstance(fraction,bool) or not isinstance(fraction,(int,float)) or not np.isfinite(fraction) or fraction<=0:
        raise ValueError('Positive finite sampling fraction required')
    eligible=levels==level
    if not eligible.any() or weights[eligible].mean()<=0:raise ValueError('Positive eligible weight required')
    rng=np.random.RandomState(seed)
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        ratios=weights.astype(float)/weights[eligible].mean()*fraction
    if not np.isfinite(ratios).all():raise ValueError('Sampling ratios overflow')
    initial=np.flatnonzero((ratios>rng.rand(len(weights)))&eligible)
    if not initial.size:raise ValueError('Sample contains no eligible rows')
    remaining=ratios[initial].copy();pieces=[initial]
    while remaining.max()>1:
        remaining-=1
        pieces.append(initial[remaining>rng.rand(len(initial))])
    rows=np.concatenate(pieces)
    horizon=(rng.randint(0,28,size=len(rows))+1).astype(np.int8)
    return np.tile(rows,repeats),np.tile(horizon,repeats)


def quantile_mask(groups,held_out,weight,level,*,seed=0):
    groups=np.asarray(groups)
    if groups.ndim!=1 or not groups.size or groups.dtype.kind not in 'iu':raise ValueError('Integer groups required')
    if type(held_out) is not int or type(level) is not int or type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Integer controls required')
    if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not np.isfinite(weight) or weight<0:
        raise ValueError('Finite nonnegative quantile weight required')
    chance=weight**(.35 if level>=11 else .25)
    return (groups!=held_out)&(np.random.RandomState(seed).rand(len(groups))<chance)
