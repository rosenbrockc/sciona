"""Source-order hierarchy aggregation, base partitioning and history preparation."""
import numpy as np
from sciona.m5u_hierarchy import aggregate,normalize_by_outlet
from sciona.m5u_weights import level_factors


def mask_leading_zeros(values):
    """Propagate missingness through following zeros, preserving later sales zeros."""
    values=np.asarray(values)
    if (values.ndim!=2 or not all(values.shape) or values.dtype.kind not in 'iuf'
            or np.isinf(values).any() or (values<0).any()):
        raise ValueError('Nonnegative or missing history required')
    result=values.astype(float,copy=True)
    result[0,result[0]==0]=np.nan
    for day in range(1,len(result)):
        result[day,(result[day]==0)&np.isnan(result[day-1])]=np.nan
    return result


def prepare(units,prices,roles,category_partitions,*,controller_level,max_level=None,
            normalization='source'):
    """Prepare the selected source batch; reject undefined outlet denominators.

    Category-to-partition mapping is explicit runtime configuration. Multipliers
    are computed before splitting and filtering; revenue precedes unit masking.
    'retained_base_reference' explicitly repairs undefined source denominators
    using pre-filter base history at the same outlet, or all base history for
    the aggregate outlet role. This is a reconstruction correction, not a claim
    about the original winning run. Defined source denominators stay unchanged.
    """
    units,prices,roles=np.asarray(units),np.asarray(prices),np.asarray(roles)
    if (units.ndim!=2 or prices.shape!=units.shape or units.dtype.kind not in 'iuf'
            or prices.dtype.kind not in 'iuf' or np.isinf(units).any() or np.isinf(prices).any()
            or (units<0).any() or (prices<0).any() or type(controller_level) is not int
            or (max_level is not None and (type(max_level) is not int or max_level<1))
            or normalization not in ('source','retained_base_reference')):
        raise ValueError('Aligned nonnegative or missing units/prices and integer batch controls required')
    hierarchy=aggregate(units,roles)
    if (not isinstance(category_partitions,dict) or set(category_partitions)!=set(roles[:,2])
            or any(type(v) is not int or v<13 for v in category_partitions.values())
            or len(set(category_partitions.values()))!=len(category_partitions)):
        raise ValueError('Complete unique category partition mapping required')
    revenue=aggregate(units.astype(float)*prices,roles)['values']
    factors=level_factors(hierarchy['levels'])
    levels=hierarchy['levels'].astype(np.int64)
    for category,partition in category_partitions.items():
        levels[(levels==12)&(hierarchy['roles'][:,2]==category)]=partition
    selected=np.flatnonzero(levels<=max_level if max_level is not None else levels==controller_level)
    if not selected.size:raise ValueError('No series in requested source batch')
    history=mask_leading_zeros(hierarchy['values'][:,selected]*factors[selected])
    metadata=hierarchy['roles'][selected]
    if normalization=='source':
        scaled=normalize_by_outlet(history,metadata[:,1],levels[selected]>=12)
    else:
        reference=mask_leading_zeros(units)
        scaled=np.empty_like(history)
        for outlet in np.unique(metadata[:,1]):
            local=(metadata[:,1]==outlet)&(levels[selected]>=12)
            if local.any():
                pool=history[:,local]
            else:
                reference_rows=np.ones(len(roles),dtype=bool) if outlet==-1 else roles[:,1]==outlet
                if not reference_rows.any():raise ValueError('Missing retained base outlet reference')
                pool=reference[:,reference_rows]
            counts=np.sum(~np.isnan(pool),axis=1)
            denominator=np.divide(np.nansum(pool,axis=1),counts,
                                  out=np.ones(len(history)),where=counts>0)
            with np.errstate(divide='ignore',invalid='ignore'):
                scaled[:,metadata[:,1]==outlet]=history[:,metadata[:,1]==outlet]/denominator[:,None]
    return dict(history=history,scaled_history=scaled,revenue=revenue[:,selected]*factors[selected],
                roles=metadata,levels=levels[selected],factors=factors[selected],hierarchy_indices=selected,
                normalization=normalization)
