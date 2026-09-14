"""Independent final four-family score assembly from the winning writeup."""
import numpy as np


def distribute_unknown(probabilities):
    """Winner mapping: spread CropNet's final unknown mass over five classes."""
    array = np.asarray(probabilities)
    if (array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 6
            or array.dtype.kind not in 'iuf' or not np.isfinite(array).all()
            or (array < 0).any() or (array > 1).any()
            or not np.allclose(array.sum(axis=1), 1., rtol=1e-6, atol=1e-7)):
        raise ValueError('Finite normalized six-class probabilities required')
    array = array.astype(float)
    return array[:, :5] + array[:, 5:6] / 5


def combine(vit,resnext,efficientnet,cropnet):
    """Consume aligned family probabilities; return source-scale scores and labels.

    Upstream fold/TTA averaging is a separate required stage. Conditional on
    normalized five-class inputs, scores sum to three. CropNet's six outputs
    must first pass through distribute_unknown. Scores are not probabilities.
    """
    arrays=[np.asarray(value) for value in (vit,resnext,efficientnet,cropnet)]
    shape=arrays[0].shape
    if len(shape)!=2 or shape[0]<1 or shape[1]!=5:
        raise ValueError('Nonempty five-class probability matrices required')
    for array in arrays:
        if (array.shape!=shape or array.dtype.kind not in 'iuf' or not np.isfinite(array).all()
                or (array<0).any() or (array>1).any()
                or not np.allclose(array.sum(axis=1),1.,rtol=1e-6,atol=1e-7)):
            raise ValueError('Aligned finite normalized family probabilities required')
    a,b,c,d=[array.astype(float) for array in arrays]
    scores=(a+b)*.5+c+d
    return dict(scores=scores,labels=np.argmax(scores,axis=1))
