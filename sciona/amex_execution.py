"""Canonical private raw-sequence boundary for the full independent Amex pipeline."""
from dataclasses import dataclass,field
import hashlib
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sciona.amex_raw import fields,preprocess


@dataclass(frozen=True,repr=False)
class Prepared:
    configuration:str=field(repr=False)
    fingerprint:str=field(repr=False)


def prepare(payload):
    fields(payload,('version','training','query','controls'))
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported version')
    fields(payload['training'],('sequences','labels','identities'));fields(payload['query'],('sequences','identities'))
    fields(payload['controls'],('category_rules','zero_fill_columns','seed'))
    try:encoded=json.dumps(payload,allow_nan=False,sort_keys=True,separators=(',',':'))
    except (ValueError,TypeError,OverflowError):raise ValueError('Finite JSON input required') from None
    if len(encoded.encode())>64*1024*1024:raise ValueError('Execution input exceeds byte limit')
    p=json.loads(encoded);controls=p['controls'];seen=set();counts=[];widths=[]
    for pop in (p['training'],p['query']):
        processed=preprocess(pop['sequences'],controls['category_rules']);n=len(processed['numerics']);counts.append(n);widths.append(processed['numerics'][0].shape[1])
        ids=pop['identities']
        if type(ids) is not list or len(ids)!=n or any(type(i) is not str or not i for i in ids) or len(set(ids))!=len(ids) or seen.intersection(ids):raise ValueError('Unique disjoint opaque customer identities required')
        seen.update(ids)
    fill=controls['zero_fill_columns']
    if widths[0]!=widths[1] or type(fill) is not list or any(type(i) is not int or not 0<=i<widths[0] for i in fill) or len(set(fill))!=len(fill):raise ValueError('Aligned numeric widths and valid fill roles required')
    y=p['training']['labels'];seed=controls['seed']
    if type(y) is not list or len(y)!=counts[0] or any(type(v) is not int or v not in (0,1) for v in y) or min(y.count(0),y.count(1))<5:raise ValueError('Aligned binary customer labels required')
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Bounded integer seed required')
    plans=StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)
    if any(len(tr)<256 for tr,va in plans):raise ValueError('Full neural batches required in every fold')
    return Prepared(encoded,hashlib.sha256(encoded.encode()).hexdigest())


def execute(prepared):
    if type(prepared) is not Prepared or type(prepared.configuration) is not str or hashlib.sha256(prepared.configuration.encode()).hexdigest()!=prepared.fingerprint:raise ValueError('Unchanged prepared input required')
    p=json.loads(prepared.configuration)
    if prepare(p)!=prepared:raise ValueError('Canonical prepared input required')
    from sciona.amex_pipeline import fit
    controls=p['controls'];training=preprocess(p['training']['sequences'],controls['category_rules']);query=preprocess(p['query']['sequences'],controls['category_rules'])
    result=fit(training,query,p['training']['labels'],zero_fill_columns=controls['zero_fill_columns'],seed=controls['seed'])
    out=dict(scores=result['scores'].tolist(),models=result['models'],model_counts=result['model_counts'],score_kind=result['score_kind'],scope=result['scope'])
    json.dumps(out,allow_nan=False)
    return out
