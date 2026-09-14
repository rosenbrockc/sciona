"""Independent twelve-level aggregation with explicit integer hierarchy roles.

Roles are region, outlet, category, department, product. -1 denotes the aggregate
role. Base-series order is retained; each added grouping is sorted by its keys.
"""
import numpy as np

GROUPS=((11,(0,4)),(10,(4,)),(9,(1,3)),(8,(1,2)),(7,(0,3)),(6,(0,2)),
        (5,(3,)),(4,(2,)),(3,(1,)),(2,(0,)),(1,()))
IMPLIED={4:(3,2),3:(2,),1:(0,)}


def aggregate(values,roles):
    values,roles=np.asarray(values),np.asarray(roles)
    if (values.ndim!=2 or not all(values.shape) or values.dtype.kind not in 'iuf' or np.isinf(values).any()
            or roles.shape!=(values.shape[1],5) or roles.dtype.kind not in 'iu' or (roles<0).any()):
        raise ValueError('Finite-or-missing time-by-series values and complete integer hierarchy roles required')
    if len(np.unique(roles[:,[4,1]],axis=0))!=len(roles):raise ValueError('Base product/outlet pairs must be unique')
    for parent,children in IMPLIED.items():
        for key in np.unique(roles[:,parent]):
            if any(len(np.unique(roles[roles[:,parent]==key,child]))!=1 for child in children):
                raise ValueError('Hierarchy implication is inconsistent')
    matrices=[values.astype(float).copy()];metadata=[roles.astype(np.int64).copy()];levels=[np.full(len(roles),12,dtype=np.int8)]
    for level,group in GROUPS:
        if group:
            _,membership=np.unique(roles[:,group],axis=0,return_inverse=True)
        else:membership=np.zeros(len(roles),dtype=int)
        retained=set(group)
        for role in group:retained.update(IMPLIED.get(role,()))
        result=[];labels=[]
        for key in np.unique(membership):
            selected=np.flatnonzero(membership==key)
            result.append(np.nansum(values[:,selected].astype(float),axis=1))
            label=roles[selected[0]].astype(np.int64).copy()
            for role in range(5):
                if role not in retained:label[role]=-1
            labels.append(label)
        matrices.append(np.column_stack(result));metadata.append(np.array(labels));levels.append(np.full(len(labels),level,dtype=np.int8))
    combined=np.concatenate(matrices,axis=1)
    if np.isinf(combined).any():raise ValueError('Aggregation overflow')
    return dict(values=combined,roles=np.concatenate(metadata),levels=np.concatenate(levels))


def normalize_by_outlet(values,outlets,base_mask):
    """Divide by base-series outlet means; all-missing base groups use one."""
    values,outlets,base_mask=np.asarray(values),np.asarray(outlets),np.asarray(base_mask)
    if (values.ndim!=2 or not all(values.shape) or values.dtype.kind not in 'iuf' or np.isinf(values).any()
            or outlets.shape!=(values.shape[1],) or outlets.dtype.kind not in 'iu'
            or base_mask.shape!=outlets.shape or base_mask.dtype!=np.bool_):
        raise ValueError('Aligned real history, integer outlets and boolean base mask required')
    result=np.empty(values.shape,dtype=float)
    for outlet in np.unique(outlets):
        source=(outlets==outlet)&base_mask
        if not source.any():raise ValueError('Outlet has no base-series denominator')
        selected=values[:,source].astype(float);count=np.sum(~np.isnan(selected),axis=1)
        denominator=np.divide(np.nansum(selected,axis=1),count,out=np.ones(len(values)),where=count>0)
        with np.errstate(divide='ignore',invalid='ignore'):
            result[:,outlets==outlet]=values[:,outlets==outlet]/denominator[:,None]
    return result
