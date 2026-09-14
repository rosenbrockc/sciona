"""Explicit runtime-role raw preprocessing for the independent Amex pipeline."""
import numpy as np
from sciona.amex_numeric import denoise_numeric


def fields(value,names):
    if type(value) is not dict or set(value)!=set(names):raise ValueError('Invalid input fields')


def preprocess(population,rules):
    fields(population,('numerics','categories','times','months'))
    if type(rules) is not list or not rules:raise ValueError('Category rules required')
    for rule in rules:
        if rule is None:continue
        fields(rule,('values','missing'))
        if type(rule['values']) is not dict or not rule['values'] or any(type(k) is not str or type(v) is not int or not -128<=v<=127 for k,v in rule['values'].items()):raise ValueError('Explicit string-to-integer mappings required')
        if rule['missing'] is not None and (type(rule['missing']) is not int or not -128<=rule['missing']<=127):raise ValueError('Valid missing-category code required')
    if any(type(v) is not list for v in population.values()) or not population['numerics'] or len({len(v) for v in population.values()})!=1:raise ValueError('Aligned sequence populations required')
    out={k:[] for k in population};width=None
    for numeric,categorical,times,months in zip(*(population[k] for k in ('numerics','categories','times','months'))):
        if type(numeric) is not list or not 1<=len(numeric)<=13 or any(type(r) is not list or not r or any(v is not None and type(v) not in (int,float) for v in r) for r in numeric):raise ValueError('Numeric sequences with explicit null missing values required')
        x=denoise_numeric(np.asarray(numeric,dtype=float));width=x.shape[1] if width is None else width
        if x.shape[1]!=width:raise ValueError('Aligned numeric widths required')
        if type(categorical) is not list or len(categorical)!=len(x) or any(type(r) is not list or len(r)!=len(rules) for r in categorical):raise ValueError('Aligned categorical rows required')
        c=np.empty((len(x),len(rules)))
        for i,row in enumerate(categorical):
            for j,(value,rule) in enumerate(zip(row,rules)):
                if rule is None:
                    if value is not None and type(value) not in (int,float):raise ValueError('Numeric category or null required')
                    c[i,j]=np.nan if value is None else np.floor(float(value)*100.)
                else:
                    if value is None:
                        if rule['missing'] is None:raise ValueError('Missing category has no mapping')
                        c[i,j]=rule['missing']
                    else:
                        if type(value) is not str or value not in rule['values']:raise ValueError('Unknown mapped category')
                        c[i,j]=rule['values'][value]
        if np.isinf(c).any():raise ValueError('Category arithmetic overflow')
        if type(times) is not list or len(times)!=len(x) or any(type(v) not in (int,float) or not np.isfinite(v) for v in times):raise ValueError('Finite aligned time-order codes required')
        if type(months) is not list or len(months)!=len(x) or any(type(v) is not int for v in months):raise ValueError('Aligned integer month-group codes required')
        out['numerics'].append(x);out['categories'].append(c);out['times'].append(times.copy());out['months'].append(months.copy())
    return out
