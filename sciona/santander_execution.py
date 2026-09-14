"""Private canonical input boundary for the complete Santander realization."""
from dataclasses import dataclass,field
import hashlib
import json
import numpy as np


@dataclass(frozen=True,repr=False)
class Prepared:
    configuration: str=field(repr=False)
    fingerprint: str=field(repr=False)


def fields(value,expected):
    if type(value) is not dict or set(value)!=set(expected):raise ValueError('Invalid execution fields')


def prepare(payload):
    fields(payload,('version','training','query','controls'))
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported execution version')
    fields(payload['training'],('values','labels','identities'));fields(payload['query'],('values','identities'))
    c=payload['controls'];fields(c,('initial_neural','neural','tree','reference_policy'))
    if c['reference_policy']!='retain_selected':raise ValueError('Explicit retained-query reference policy required')
    for branch in ('initial_neural','neural'):
        fields(c[branch],('folds','seeds','fold_seed','epochs','batch_size','maximum_lr'))
        if type(c[branch]['epochs']) is not int or c[branch]['epochs']!=15:raise ValueError('Fifteen neural epochs required')
        if type(c[branch]['batch_size']) is not int or c[branch]['batch_size']<2:raise ValueError('Invalid batch size')
        rate=c[branch]['maximum_lr']
        if type(rate) not in (int,float) or not np.isfinite(rate) or not 0<rate<=1:raise ValueError('Invalid neural rate')
    fields(c['tree'],('folds','seeds','fold_seed','controls'))
    fields(c['tree']['controls'],('num_leaves','learning_rate','feature_fraction','bagging_fraction','bagging_freq','min_data_in_leaf','max_rounds','stopping_rounds','categorical'))
    for branch in ('initial_neural','neural','tree'):
        b=c[branch]
        if type(b['folds']) is not int or b['folds']!=10:raise ValueError('Ten folds required for every stage')
        if type(b['fold_seed']) is not int or not 0<=b['fold_seed']<2**31:raise ValueError('Invalid fold seed')
        if type(b['seeds']) not in (list,tuple) or not b['seeds'] or any(type(s) is not int or not 0<=s<2**31 for s in b['seeds']) or len(set(b['seeds']))!=len(b['seeds']):raise ValueError('Distinct bounded seeds required')
    try:encoded=json.dumps(payload,allow_nan=False,sort_keys=True,separators=(',',':'))
    except (TypeError,ValueError,OverflowError):raise ValueError('Finite JSON input required') from None
    if len(encoded.encode())>64*1024*1024:raise ValueError('Execution input exceeds byte limit')
    p=json.loads(encoded);t=p['training'];q=p['query'];matrices=[];seen=set()
    for population in (t,q):
        try:values=np.asarray(population['values'])
        except ValueError:raise ValueError('Rectangular numeric population required') from None
        if values.ndim!=2 or not all(values.shape) or values.dtype.kind not in 'iuf' or not np.isfinite(values).all():raise ValueError('Finite real matrices required')
        ids=population['identities']
        if type(ids) is not list or len(ids)!=len(values) or any(type(i) is not str or not i for i in ids) or len(set(ids))!=len(ids) or seen.intersection(ids):raise ValueError('Disjoint aligned opaque identities required')
        seen.update(ids);matrices.append(values)
    x,v=matrices
    if x.shape[1]!=v.shape[1] or len(v)<8000:raise ValueError('Aligned populations and full default pseudo-label capacity required')
    y=t['labels']
    if type(y) is not list or len(y)!=len(x) or any(type(i) is not int or i not in (0,1) for i in y) or min(y.count(0),y.count(1))<10:raise ValueError('Ten or more original examples per class required')
    return Prepared(encoded,hashlib.sha256(encoded.encode()).hexdigest())


def execute(prepared):
    if type(prepared) is not Prepared or type(prepared.configuration) is not str or hashlib.sha256(prepared.configuration.encode()).hexdigest()!=prepared.fingerprint:raise ValueError('Unchanged prepared input required')
    try:p=json.loads(prepared.configuration)
    except (ValueError,TypeError):raise ValueError('Invalid prepared JSON') from None
    if prepare(p)!=prepared:raise ValueError('Canonical prepared input required')
    from sciona.santander_lifecycle import fit
    c=p['controls']
    result=fit(np.asarray(p['training']['values'],dtype=float),p['training']['labels'],np.asarray(p['query']['values'],dtype=float),
        initial_neural_controls=c['initial_neural'],neural_controls=c['neural'],tree_controls=c['tree'],reference_policy=c['reference_policy'])
    output=dict(scores=result['scores'].tolist(),selection_counts=result['selection_counts'],
        model_counts={'initial_neural':len(result['initial_neural']['models']),**{name:len(b['models']) for name,b in result['branches'].items()}},
        score_kind='rank_blend',reference_policy=result['reference_policy'],validation_scope=result['validation_scope'])
    json.dumps(output,allow_nan=False)
    return output
