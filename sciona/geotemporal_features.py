"""Causal projected-coordinate joins and spatial/temporal regression features.

Coordinates must share a caller-declared planar metric reference; times are
numeric seconds in one clock. No implicit latitude/longitude conversion.
"""
import math
import numpy as np


def rows(value,extra):
    if type(value) is not list or not value:raise ValueError('Nonempty observation list required')
    fields={'x','y','time'}|set(extra)
    for row in value:
        if type(row) is not dict or set(row)!=fields:raise ValueError('Unexpected observation fields')
        for v in row.values():
            if type(v) not in (int,float):raise ValueError('Finite numeric observations required')
            try:valid=math.isfinite(v)
            except OverflowError:valid=False
            if not valid:raise ValueError('Finite numeric observations required')
    return value


def feature_matrix(points,context,history,*,radius,lookback,period,neighbors):
    """Latest available context per site, then nearest-site mean; strict-past lags.

Context times <= query time are permitted. Target history times must be strictly
less than query time and within lookback. Missing aggregates use zero plus an
explicit availability indicator. Distances use the shared planar coordinates.
"""
    rows(points,());rows(context,('value',))
    if type(history) is not list:raise ValueError('History must be a list')
    if history:rows(history,('target',))
    for value in (radius,lookback,period):
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive finite feature scales required')
    if type(neighbors) is not int or neighbors<1:raise ValueError('Positive neighbor count required')
    for population in (context,history):
        keys=[(r['x'],r['y'],r['time']) for r in population]
        if len(set(keys))!=len(keys):raise ValueError('Ambiguous duplicate space/time observation')
    output=[]
    for point in points:
        latest={}
        for item in context:
            age=point['time']-item['time'];distance=math.hypot(point['x']-item['x'],point['y']-item['y'])
            if not 0<=age<=lookback or distance>radius:continue
            site=(item['x'],item['y'])
            if site not in latest or item['time']>latest[site]['time']:latest[site]=item
        selected=sorted(latest.values(),key=lambda r:(math.hypot(point['x']-r['x'],point['y']-r['y']),r['x'],r['y']))[:neighbors]
        if selected:
            context_value=sum(r['value']/len(selected) for r in selected)
            distance=sum(math.hypot(point['x']-r['x'],point['y']-r['y'])/len(selected) for r in selected)
            age=sum((point['time']-r['time'])/len(selected) for r in selected)
        else:context_value=distance=age=0.
        past=[r for r in history if 0<point['time']-r['time']<=lookback and math.hypot(point['x']-r['x'],point['y']-r['y'])<=radius]
        lag=sum(r['target']/len(past) for r in past) if past else 0.
        phase=2*math.pi*((point['time']%period)/period)
        output.append([point['x'],point['y'],math.sin(phase),math.cos(phase),context_value,distance,age,float(bool(selected)),lag,float(bool(past)),float(len(past))])
    result=np.asarray(output,dtype=np.float64)
    if not np.isfinite(result).all():raise ValueError('Nonfinite joined features')
    return result
