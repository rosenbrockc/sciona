"""Twenty-member reconstructed inference and explicit teacher statistics.

All members receive the same clipping, correcting the adapted source's
first-member exemption. Teacher moments use un-clipped, reverse-averaged
members; callers explicitly choose population or sample standard deviation.
This inventory identifies the DasLab reconstruction, not the historical blend.
"""
import numpy as np

from sciona.openvaccine_models import validate_inputs

EXPECTED_MEMBERS = frozenset((family, slot) for family in ('lstm','gru','forward','wave') for slot in range(5))


def predict_twenty(members, nodes, adjacency, *, std_ddof):
    nodes, adjacency = validate_inputs(nodes, adjacency)
    if isinstance(std_ddof, bool) or std_ddof not in (0, 1):
        raise ValueError('Explicit standard deviation ddof must be 0 or 1')
    seen = set()
    mean = np.zeros((nodes.shape[0],nodes.shape[1]-2,5),dtype=np.float64)
    moment = np.zeros_like(mean)
    clipped_sum = np.zeros_like(mean)
    reversed_nodes = nodes[:,::-1,:].copy()
    reversed_adjacency = adjacency[:,::-1,::-1,:].copy()
    for family, slot, runtime in members:
        key = (family,slot)
        if isinstance(slot,bool) or key not in EXPECTED_MEMBERS or key in seen or runtime.family != family:
            raise ValueError('Invalid, duplicate or mismatched ensemble member')
        seen.add(key)
        forward = runtime.predict(nodes,adjacency)
        reverse = runtime.predict(reversed_nodes,reversed_adjacency)[:,::-1,:]
        prediction = (forward + reverse)/2
        if prediction.shape != mean.shape or not np.isfinite(prediction).all():
            raise ValueError('Invalid reverse-averaged member prediction')
        value = prediction.astype(np.float64)
        delta = value-mean
        mean += delta/len(seen)
        moment += delta*(value-mean)
        clipped_sum += np.clip(value,-.5,6.)
    if seen != EXPECTED_MEMBERS:
        raise ValueError('Complete four-family five-state ensemble required')
    return dict(prediction=(clipped_sum/20).astype(np.float32),
                teacher_mean=mean.astype(np.float32),
                teacher_std=np.sqrt(np.maximum(moment,0)/(20-std_ddof)).astype(np.float32))
