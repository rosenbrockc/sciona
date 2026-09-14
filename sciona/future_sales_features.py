"""Generic period aggregation and causal demand features on private inputs.

The caller explicitly supplies the active entity universe for every period.
Missing observed-period entity transactions mean zero, including returns netted
within a period. The final period is unobserved and its targets remain NaN.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Panel:
    features: np.ndarray
    targets: np.ndarray
    periods: np.ndarray
    entity_count: int
    feature_names: tuple[str,...]


def build_panel(transactions,entities,boundaries,*,lags,rolling_windows,calendar_cycle,cold_start):
    if not isinstance(boundaries,list) or len(boundaries)<4 or any(type(v) is not int for v in boundaries) or any(b<=a for a,b in zip(boundaries,boundaries[1:])):
        raise ValueError('Strictly increasing integer period boundaries required')
    for offsets in (lags,rolling_windows):
        if not isinstance(offsets,list) or not offsets or any(type(x) is not int or x<1 for x in offsets) or len(set(offsets))!=len(offsets):raise ValueError('Unique positive lag/window offsets required')
    if type(calendar_cycle) is not int or calendar_cycle<1 or type(cold_start) not in (int,float) or not math.isfinite(cold_start) or not 0<=cold_start<=20:
        raise ValueError('Explicit cycle and bounded cold-start mean required')
    if not isinstance(entities,list) or not entities:raise ValueError('Explicit entity population required')
    keys=[];lookup={};categories={}
    for e in entities:
        if not isinstance(e,dict) or set(e)!={'store','item','category'} or any(type(v) is not int or not 0<=v<2**31 for v in e.values()):raise ValueError('Nonnegative bounded integer entity slots required')
        key=e['store'],e['item']
        if key in lookup:raise ValueError('Duplicate entity')
        if e['item'] in categories and categories[e['item']]!=e['category']:raise ValueError('Inconsistent item category')
        categories[e['item']]=e['category'];lookup[key]=len(keys);keys.append((e['store'],e['item'],e['category']))
    if not isinstance(transactions,list):raise ValueError('Explicit transaction list required')
    nperiods=len(boundaries)-1;known=nperiods-1
    # Use finite raw sums, clipping only after aggregation of all rows.
    amounts={}
    for t in transactions:
        if not isinstance(t,dict) or set(t)!={'day','store','item','quantity'}:raise ValueError('Exact transaction fields required')
        if any(type(t[k]) is not int for k in ('day','store','item')) or type(t['quantity']) not in (int,float) or not math.isfinite(t['quantity']):raise ValueError('Finite numeric transaction required')
        if not boundaries[0]<=t['day']<boundaries[-2]:raise ValueError('Transactions must precede the forecast period')
        key=t['store'],t['item']
        if key not in lookup:raise ValueError('Transaction outside explicit entity population')
        period=int(np.searchsorted(boundaries,t['day'],side='right')-1)
        amounts.setdefault((period,lookup[key]),[]).append(t['quantity'])
    y=np.zeros((nperiods,len(keys)),dtype=float)
    for key,values in amounts.items():
        try:value=math.fsum(values)
        except OverflowError:raise ValueError('Aggregate exceeds finite range') from None
        if not math.isfinite(value):raise ValueError('Aggregate exceeds finite range')
        y[key]=np.clip(value,0.,20.)
    y[-1]=np.nan
    names=('store_slot','item_slot','category_slot','period_index','cycle_sin','cycle_cos')+tuple('lag_'+str(v) for v in lags)+tuple('rolling_'+str(v) for v in rolling_windows)+('entity_expanding','store_expanding','item_expanding','category_expanding','global_expanding')
    population=np.asarray(keys);features=[];period_ids=[]
    for period in range(nperiods):
        history=y[:period]
        for index,(store,item,category) in enumerate(keys):
            values=[store,item,category,period,math.sin(2*math.pi*(period%calendar_cycle)/calendar_cycle),math.cos(2*math.pi*(period%calendar_cycle)/calendar_cycle)]
            values += [y[period-offset,index] if period>=offset else np.nan for offset in lags]
            values += [float(y[period-width:period,index].mean()) if period>=width else np.nan for width in rolling_windows]
            if period:
                values += [float(history[:,index].mean())]+[float(history[:,population[:,axis]==value].mean()) for axis,value in enumerate((store,item,category))]+[float(history.mean())]
            else:values += [float(cold_start)]*5
            features.append(values);period_ids.append(period)
    features=np.asarray(features,dtype=float)
    if np.isinf(features).any():raise ValueError('Feature overflow')
    return Panel(features,y.reshape(-1),np.asarray(period_ids),len(keys),names)
