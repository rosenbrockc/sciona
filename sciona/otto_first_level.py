"""All 33 source entries, using explicit independent CPU/runtime realizations."""
import importlib
import json
import numpy as np
from sciona.otto_preprocessing import representation
from sciona.otto_meta_layout import AlignedBlock
from sciona.otto_tsne import PopulationEmbedding
from sciona.otto_tsne_clusters import fit_embedding_clusters


def build(training,labels,folds,training_ids,query,query_ids,*,embedding,seed,controls,progress=None):
    x=representation(training,kind='raw');q=representation(query,kind='raw');y=np.asarray(labels);f=np.asarray(folds)
    if x.shape[1]<13 or q.shape[1]!=x.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!=set(range(9)):raise ValueError('Aligned nine-class populations with at least thirteen features required')
    if f.shape!=(len(x),) or f.dtype.kind not in 'iu' or set(f.tolist())!=set(range(5)):raise ValueError('Five aligned folds required')
    seen=set()
    for ids,size in ((training_ids,len(x)),(query_ids,len(q))):
        if type(ids) is not list or len(ids)!=size or any(type(i) is not str or not i for i in ids):raise ValueError('Aligned opaque identities required')
        for identity in ids:
            if identity in seen:raise ValueError('Duplicate or overlapping identities')
            seen.add(identity)
    for fold in range(5):
        if np.sum(f!=fold)<1024 or any(np.sum((f!=fold)&(y==label))<4 for label in range(9)):raise ValueError('Every fold needs 1024 references and four of each class')
    if not isinstance(embedding,PopulationEmbedding):raise ValueError('Fixed embedding required')
    embedding.lookup(training_ids);embedding.lookup(query_ids)
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Common signed integer seed required')
    if type(controls) is not dict or any(type(i) is not int for i in controls) or set(controls)!=set(range(1,34)) or any(type(c) is not dict for c in controls.values()):raise ValueError('Explicit controls for all 33 entries required')
    args=(x,y,f,training_ids,q,query_ids);blocks={};knn_cache={}
    def call(name,function,**kwargs):return getattr(importlib.import_module('sciona.otto_'+name),function)(*args,**kwargs)
    for entry in range(1,34):
        options=controls[entry]
        if entry==10:
            model=fit_embedding_clusters(embedding,training_ids,seed=seed,**options)
            a=model.lookup(training_ids);b=model.lookup(query_ids)
        else:
            if entry==1:r=call('random_forest_crossfit','crossfit_forest',seed=seed,**options)
            elif entry in (2,3,7):r=call('classical_models','crossfit',family={2:'logistic',3:'extra_trees',7:'multinomial_nb'}[entry],seed=seed,controls=options)
            elif entry in (4,22,23):r=call('variant_knn','crossfit',variant={4:'scaled_log',22:'raw_zero',23:'raw_zero_log'}[entry],**options)
            elif entry==5:r=call('libfm_crossfit','crossfit_libfm',seed=seed,**options)
            elif entry==6:r=call('h2o_crossfit','crossfit_h2o',seed=seed,controls=options)
            elif entry in (8,9):r=call('lasagne_crossfit','crossfit_lasagne',variant='pair' if entry==8 else 'six_log',seed=seed,controls=options)
            elif entry==11:r=call('sofia_crossfit','crossfit_sofia',seed=seed,**options)
            elif entry in (12,13):r=call('sofia_interactions_crossfit' if entry==12 else 'sofia_roc_crossfit','crossfit_sofia',embedding=embedding,seed=seed,controls=options)
            elif entry in (14,15,21):r=call({14:'ovr_crossfit',15:'augmented_crossfit',21:'raw_boosting_crossfit'}[entry],'crossfit_boosting',seed=seed,controls=options)
            elif entry in (16,17,18):r=call('embedding_crossfit','crossfit_boosting',embedding=embedding,variant={16:'raw',17:'log1p',18:'scaled_raw'}[entry],seed=seed,controls=options)
            elif entry in (19,20):r=call('lasagne_120_crossfit','crossfit_lasagne',variant='two_hidden' if entry==19 else 'three_hidden',seed=seed,controls=options)
            else:
                key=json.dumps(options,sort_keys=True)
                if key not in knn_cache:knn_cache[key]=call('knn_models','crossfit',**options)
                raw=knn_cache[key];column=entry-24
                r=dict(oof=raw['oof'][:,column,:],query=raw['query'][:,column,:])
            a=np.asarray(r['oof']);b=np.asarray(r['query'])
        width=5 if entry==10 else 9
        if a.shape!=(len(x),width) or b.shape!=(len(q),width) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Producer emitted invalid first-level block')
        blocks[entry]=AlignedBlock(a,b,tuple(training_ids),tuple(query_ids),None if entry==10 else f.copy())
        if progress is not None:progress(entry)
    return blocks
