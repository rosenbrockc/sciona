"""Connect calendar price roles, price arithmetic, precision and grid joins.

The supplied population uses semantic role arrays, never embedded source column
names. Grid pair codes and price pair codes must refer to the same runtime map.
"""
import numpy as np
from sciona.m5_join import first_per_key,left_indices,take_numeric
from sciona.m5_prices import features
from sciona.m5_precision import downcast


def build(grid_pairs, grid_weeks, population, calendar):
    required={'pair','outlet','product','week','value'}
    if not isinstance(population,dict) or set(population)!=required:
        raise ValueError('Price role population differs')
    if not isinstance(calendar,dict) or set(calendar)!={'week','month','year'}:
        raise ValueError('Calendar price roles differ')
    p={k:np.asarray(v) for k,v in population.items()}
    c={k:np.asarray(v) for k,v in calendar.items()}
    n=p['value'].size
    if not n or any(v.shape!=(n,) for v in p.values()):
        raise ValueError('Price roles must be aligned nonempty vectors')
    if any(p[k].dtype.kind not in 'iu' for k in required-{'value'}):
        raise ValueError('Price key roles must be integers')
    if any(v.ndim!=1 or v.shape!=c['week'].shape or v.dtype.kind not in 'iu' for v in c.values()):
        raise ValueError('Calendar roles must be aligned integer vectors')
    # A pair code represents exactly one outlet/product tuple and vice versa.
    mapping={};reverse={}
    for pair,outlet,product in zip(p['pair'],p['outlet'],p['product']):
        key=(int(outlet),int(product));pair=int(pair)
        if mapping.setdefault(pair,key)!=key or reverse.setdefault(key,pair)!=pair:
            raise ValueError('Inconsistent price pair coding')
    first=first_per_key(c['week'][:,None])
    calendar_index=left_indices(p['week'][:,None],c['week'][first,None])
    if (calendar_index<0).any():
        raise ValueError('Price week lacks calendar roles')
    monthly=c['month'][first][calendar_index];yearly=c['year'][first][calendar_index]
    computed=features(p['value'],p['outlet'],p['product'],monthly,yearly)
    grid_pairs,grid_weeks=np.asarray(grid_pairs),np.asarray(grid_weeks)
    if (grid_pairs.ndim!=1 or grid_weeks.shape!=grid_pairs.shape
            or grid_pairs.dtype.kind not in 'iu' or grid_weeks.dtype.kind not in 'iu'):
        raise ValueError('Grid pair/week roles must be aligned integers')
    index=left_indices(np.column_stack((grid_pairs,grid_weeks)),np.column_stack((p['pair'],p['week'])))
    columns={'value':p['value'],**computed}
    # Count features originate as integer columns in the source price table.
    columns['distinct_prices']=columns['distinct_prices'].astype(np.int64)
    if not np.isnan(columns['distinct_products']).any():
        columns['distinct_products']=columns['distinct_products'].astype(np.int64)
    return {name:take_numeric(downcast(column),index) for name,column in columns.items()}
