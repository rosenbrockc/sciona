"""Source-specific momentum attack updates over explicit normalized arrays.

Derived from dongyp13 non-targeted commit5c68162e05de7afed2e3d33115df43e2a7a1c3da
and targeted commit7da707fe568053a9e2735fcc692c8f0ad7e122a1. Apache-2.0;
see docs/licenses/Adversarial-*-Apache-2.0.txt. Neural gradient computation is separate.
"""
import math
import numpy as np


def configuration(mode, epsilon, *, non_targeted_iterations=10):
    if type(epsilon) not in (float,int) or not math.isfinite(epsilon) or not 0 <= epsilon <= 255:
        raise ValueError('epsilon must be finite pixel units in [0,255]')
    if mode=='non_targeted':
        if type(non_targeted_iterations) is not int or non_targeted_iterations<=0:
            raise ValueError('non-targeted iteration count must be positive')
        return {'branch':'non_targeted','models':8,'iterations':non_targeted_iterations,
                'alpha':(2.*epsilon/255.)/non_targeted_iterations}
    if mode=='targeted':
        large=epsilon>=8
        return {'branch':'targeted_large' if large else 'targeted_small','models':5 if large else 2,
                'iterations':20 if large else 40,'alpha':(2.*epsilon/255.)/(12 if large else 28)}
    raise ValueError('mode must be targeted or non_targeted')


def update(x, gradient, accumulated, lower, upper, *, mode, epsilon, momentum=1.,
           non_targeted_iterations=10):
    """Return projected image and next momentum, without mutating input arrays.

    Values are NHWC float32/64 in normalized [-1,1]. Zero normalization
    denominators explicitly fail instead of silently propagating source NaNs;
    no stabilizing epsilon or substituted update is introduced.
    """
    config=configuration(mode,epsilon,non_targeted_iterations=non_targeted_iterations)
    if type(momentum) not in (float,int) or not math.isfinite(momentum) or momentum<0:
        raise ValueError('momentum must be finite and nonnegative')
    arrays=(x,gradient,accumulated,lower,upper)
    if (not isinstance(x,np.ndarray) or x.ndim!=4 or min(x.shape)==0 or x.shape[-1]!=3
            or x.dtype not in (np.dtype('float32'),np.dtype('float64'))
            or any(not isinstance(a,np.ndarray) or a.shape!=x.shape or a.dtype!=x.dtype
                   or not np.isfinite(a).all() for a in arrays)):
        raise ValueError('update requires matching finite NHWC floating arrays')
    if np.any(lower < -1) or np.any(upper > 1) or np.any(lower>x) or np.any(x>upper):
        raise ValueError('image must lie inside valid normalized clipping bounds')
    if mode=='non_targeted':
        divisor=np.mean(np.abs(gradient),axis=(1,2,3),keepdims=True)
    else:
        divisor=np.std(gradient.reshape(len(x),-1),axis=1).reshape(len(x),1,1,1)
    if np.any(divisor==0) or not np.isfinite(divisor).all():
        raise ValueError('gradient normalization is undefined')
    noise=momentum*accumulated+gradient/divisor
    if mode=='targeted':
        divisor=np.std(noise.reshape(len(x),-1),axis=1).reshape(len(x),1,1,1)
        if np.any(divisor==0) or not np.isfinite(divisor).all():
            raise ValueError('momentum normalization is undefined')
        noise=noise/divisor
        result=x-config['alpha']*np.clip(np.round(noise),-2,2)
    else:
        result=x+config['alpha']*np.sign(noise)
    result=np.clip(result,lower,upper)
    if not np.isfinite(result).all() or not np.isfinite(noise).all():
        raise ValueError('update became nonfinite')
    return result,noise
