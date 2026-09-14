"""Private JSON boundary for the complete independent Otto CPU realization.

Population and control structure are checked before fitting. Individual native
learners validate their detailed controls. Inputs and prepared fingerprints
must never be included in published evidence or diagnostics.
"""
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np
from sciona.otto_preprocessing import representation


@dataclass(frozen=True, repr=False)
class Prepared:
    configuration: str = field(repr=False)
    fingerprint: str = field(repr=False)


def _fields(value, expected):
    if type(value) is not dict or set(value)!=set(expected):
        raise ValueError('Invalid execution fields')


def _first_controls(serialized):
    controls={int(k):dict(v) for k,v in serialized.items()}
    for entry in (12,13):
        triples=controls[entry].get('triples')
        if type(triples) is not list or not triples:
            raise ValueError('Nonempty interaction triples required')
        converted=[];seen=set()
        for triple in triples:
            if type(triple) is not list or len(triple)!=3 or any(type(i) is not int or not 0<=i<13 for i in triple) or len(set(triple))!=3:
                raise ValueError('Three distinct integer feature ranks required')
            key=tuple(sorted(triple))
            if key in seen:raise ValueError('Duplicate interaction triple')
            seen.add(key);converted.append(tuple(triple))
        controls[entry]['triples']=converted
    return controls


def prepare(payload):
    _fields(payload,('version','training','query','controls'))
    if type(payload['version']) is not int or payload['version']!=1:
        raise ValueError('Unsupported execution version')
    _fields(payload['training'],('values','labels','identities','first_folds','meta_folds'))
    _fields(payload['query'],('values','identities'))
    controls=payload['controls']
    _fields(controls,('seed','embedding','first_level','supplemental','meta_candidates','tsne_interpretation'))
    if type(controls['seed']) is not int or not 0<=controls['seed']<2**31:
        raise ValueError('Signed integer seed required')
    if controls['tsne_interpretation']!='five_features':
        raise ValueError('Explicit five-feature interpretation required')
    _fields(controls['embedding'],('perplexity','learning_rate','max_iter'))
    _fields(controls['first_level'],(str(i) for i in range(1,34)))
    if any(type(c) is not dict for c in controls['first_level'].values()):
        raise ValueError('Explicit first-level control objects required')
    _fields(controls['supplemental'],('raw_metrics','tfidf_metrics','embedding_metrics','tfidf_controls','cluster_controls'))
    _fields(controls['meta_candidates'],('xgboost','lasagne_neural','adaboost_extratrees'))
    if any(type(cs) is not list or not cs or any(type(c) is not dict for c in cs)
           for cs in controls['meta_candidates'].values()):
        raise ValueError('Nonempty meta candidate lists required')
    try:
        encoded=json.dumps(payload,allow_nan=False,sort_keys=True,separators=(',',':'))
    except (ValueError,TypeError,OverflowError):
        raise ValueError('Finite JSON execution input required') from None
    if len(encoded.encode())>64*1024*1024:
        raise ValueError('Execution input exceeds byte limit')
    # Validate the canonical representation that execute will consume.
    p=json.loads(encoded);t=p['training'];q=p['query']
    _first_controls(p['controls']['first_level'])
    x=representation(t['values'],kind='raw');v=representation(q['values'],kind='raw')
    if x.shape[1]<13 or v.shape[1]!=x.shape[1]:
        raise ValueError('Aligned matrices with at least thirteen features required')
    labels=t['labels']
    if type(labels) is not list or len(labels)!=len(x) or any(type(i) is not int for i in labels) or set(labels)!=set(range(9)):
        raise ValueError('Nine aligned integer classes required')
    y=np.asarray(labels);seen=set()
    for ids,size in ((t['identities'],len(x)),(q['identities'],len(v))):
        if type(ids) is not list or len(ids)!=size or any(type(i) is not str or not i for i in ids):
            raise ValueError('Aligned opaque identities required')
        if len(set(ids))!=len(ids) or seen.intersection(ids):
            raise ValueError('Duplicate or overlapping identities')
        seen.update(ids)
    for name,count in (('first_folds',5),('meta_folds',4)):
        folds=t[name]
        if type(folds) is not list or len(folds)!=len(x) or any(type(i) is not int for i in folds) or set(folds)!=set(range(count)):
            raise ValueError('Invalid aligned fold partition')
        f=np.asarray(folds)
        for fold in range(count):
            fit=f!=fold
            if set(y[fit].tolist())!=set(range(9)):
                raise ValueError('Each fitting fold requires all nine classes')
            if count==5 and (fit.sum()<1024 or any(np.sum(y[fit]==i)<4 for i in range(9))):
                raise ValueError('Each first-level fitting fold needs 1024 references and four per class')
    embedding=p['controls']['embedding']
    for name in ('perplexity','learning_rate'):
        if type(embedding[name]) not in (int,float) or embedding[name]<=0:
            raise ValueError('Positive embedding controls required')
    if embedding['perplexity']>=len(x)+len(v) or type(embedding['max_iter']) is not int or embedding['max_iter']<300:
        raise ValueError('Invalid embedding population or iteration controls')
    return Prepared(encoded,hashlib.sha256(encoded.encode()).hexdigest())


def execute(prepared):
    if type(prepared) is not Prepared or type(prepared.configuration) is not str:
        raise ValueError('Prepared execution required')
    if hashlib.sha256(prepared.configuration.encode()).hexdigest()!=prepared.fingerprint:
        raise ValueError('Prepared execution changed')
    try:payload=json.loads(prepared.configuration)
    except (ValueError,TypeError):raise ValueError('Invalid prepared JSON') from None
    if prepare(payload)!=prepared:
        raise ValueError('Prepared execution is not canonical')
    from sciona.otto_tsne import fit_population
    from sciona.otto_first_level import build
    from sciona.otto_supplemental import build as supplemental
    from sciona.otto_meta_layout import assemble
    from sciona.otto_meta_stage import fit
    t=payload['training'];q=payload['query'];c=payload['controls'];seed=c['seed']
    x=np.asarray(t['values']);v=np.asarray(q['values'])
    embedding=fit_population(np.vstack((x,v)),t['identities']+q['identities'],seed=seed,**c['embedding'])
    args=(x,t['labels'],t['first_folds'],t['identities'],v,q['identities'])
    first=build(*args,embedding=embedding,seed=seed,controls=_first_controls(c['first_level']))
    extra=supplemental(*args,embedding=embedding,seed=seed,**c['supplemental'])
    layout=assemble(first,extra['supplemental'],extra['raw_neural'],t['identities'],q['identities'],t['first_folds'],tsne_interpretation=c['tsne_interpretation'])
    result=fit(layout,t['labels'],t['meta_folds'],t['identities'],seed=seed,candidates=c['meta_candidates'])
    json.dumps(result,allow_nan=False)
    return result
