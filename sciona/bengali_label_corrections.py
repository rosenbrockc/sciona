"""Apply runtime-supplied label corrections by identity before joint encoding.

Matches source DataFrame.update alignment: preserve original row order, ignore
correction-only identities, and leave original components when a correction is
missing. Ambiguous duplicate identities are rejected. No provenance is logged.
"""
import numpy as np

from sciona.bengali_joint_labels import encode_components


def _identities(values, count):
    if not isinstance(values,(list,tuple)) or len(values)!=count:
        raise ValueError('aligned identity sequence required')
    if any(not isinstance(value,str) or not value for value in values) or len(set(values))!=count:
        raise ValueError('unique nonempty string identities required')
    return {value:index for index,value in enumerate(values)}


def corrected_joint_labels(identities, components, correction_identities, corrections):
    base=np.asarray(components)
    encode_components(base)  # Validate bounds before any conversion or mutation.
    positions=_identities(identities,len(base))
    updates=np.asarray(corrections)
    if updates.ndim!=2 or updates.shape[1]!=3 or updates.dtype.kind not in 'iuf':
        raise ValueError('numeric correction triples required; NaN means unchanged')
    _identities(correction_identities,len(updates))
    active=~np.isnan(updates)
    bounds=np.array([168,11,8])
    if np.any(active & (~np.isfinite(updates) | (updates<0) | (updates>=bounds) | (updates!=np.floor(updates)))):
        raise ValueError('corrections must be integral class indices within bounds or NaN')
    result=base.astype(np.int64,copy=True)
    for index,identity in enumerate(correction_identities):
        target=positions.get(identity)
        if target is not None:
            mask=active[index]
            result[target,mask]=updates[index,mask].astype(np.int64)
    return encode_components(result)
