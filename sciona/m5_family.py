"""Source-specific feature order and pool selection for six M5 model families.

Runtime hierarchy roles are region, outlet, category, department, product in
that order. The prepared representation retains categorical vocabularies.
"""
import numpy as np
import pandas as pd
from sciona.m5_preprocessing import GROUPINGS
from sciona.m5_precision import downcast
from sciona.m5_training import parameters


def pool_roles(pooling):
    return {'outlet':(1,),'outlet_category':(1,2),'outlet_department':(1,3)}[pooling]


def pools(prepared,pooling):
    parameters(True,pooling)
    roles=pool_roles(pooling)
    keys=np.column_stack([np.asarray(prepared['categorical'][f'role_{i}']) for i in roles])
    return [tuple(map(int,row)) for row in np.unique(keys,axis=0)]


def frame(prepared, *, recursive, pooling, pool, first_day):
    parameters(recursive,pooling)
    if prepared.get('encoding_groupings')!=GROUPINGS:
        raise ValueError('Encoding grouping order differs from reviewed source')
    if (isinstance(first_day,(bool,np.bool_)) or not isinstance(first_day,(int,np.integer))
            or not isinstance(pool,(list,tuple)) or len(pool)!=len(pool_roles(pooling))
            or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in pool)):
        raise ValueError('Invalid pool or first day')
    categorical=prepared['categorical'];grid=prepared['grid']
    keep=grid['day']>=first_day
    for role,value in zip(pool_roles(pooling),pool):
        keep=keep & (np.asarray(categorical[f'role_{role}'])==value)
    rows=np.flatnonzero(keep)
    if not rows.size:raise ValueError('Empty model pool')
    retained={'outlet':(4,3,2),'outlet_category':(4,3),'outlet_department':(4,)}[pooling]
    columns={f'role_{i}':categorical[f'role_{i}'][rows] for i in retained}
    columns['release']=grid['release'][rows]
    for name,value in prepared['prices'].items():columns['price_'+name]=value[rows]
    for i in range(7):columns[f'calendar_category_{i}']=categorical[f'calendar_{i}'][rows]
    for name,value in prepared['calendar'].items():columns['date_'+name]=value[rows]
    selected= ((2,3,8) if recursive else (7,9)) if pooling=='outlet' else ((7,10) if pooling=='outlet_category' else (10,))
    encoded={f'encoding_{i}_{stat}':prepared['encoding'][rows,2*i+j]
             for i in selected for j,stat in enumerate(('mean','std'))}
    lags={name:value[rows] for name,value in prepared['lags'].items() if recursive or not name.startswith('temporary_')}
    if recursive:columns.update(encoded);columns.update(lags)
    else:columns.update(lags);columns.update(encoded)
    result=pd.DataFrame(columns)
    if not recursive:
        for name in result:
            if not isinstance(result[name].dtype,pd.CategoricalDtype):
                result[name]=downcast(result[name].to_numpy())
    return result,rows
