"""Execute all seven shared Otto supplemental features plus neural raw values.

Label-dependent distances exclude each row's fold. Clusters fit full training
without labels in one common coordinate frame. The supplied embedding retains
its declared (possibly transductive) scope. Historical multiplicities unresolved.
"""
import numpy as np
from sciona.otto_preprocessing import representation
from sciona.otto_neighbor_crossfit import crossfit_distances
from sciona.otto_tfidf_crossfit import crossfit_tfidf_distances
from sciona.otto_clustering import fit_clusters
from sciona.otto_meta_layout import AlignedBlock
from sciona.otto_tsne import PopulationEmbedding


def build(training,labels,folds,training_ids,query,query_ids,*,embedding,seed,raw_metrics,tfidf_metrics,embedding_metrics,tfidf_controls,cluster_controls):
    x=representation(training,kind='raw');q=representation(query,kind='raw');f=np.asarray(folds)
    if not isinstance(embedding,PopulationEmbedding):raise ValueError('Fixed population embedding required')
    for metrics,allowed in [(raw_metrics,('euclidean','cityblock','braycurtis')),(tfidf_metrics,('euclidean','cityblock','braycurtis')),(embedding_metrics,('euclidean','cityblock'))]:
        if type(metrics) is not list or not metrics or any(type(m) is not str or m not in allowed for m in metrics) or len(set(metrics))!=len(metrics):raise ValueError('Explicit distinct supported metric list required')
    if type(tfidf_controls) is not dict or set(tfidf_controls)!={'smooth_idf','sublinear_tf','norm'}:raise ValueError('Explicit TF-IDF controls required')
    if type(cluster_controls) is not dict or set(cluster_controls)!={'kind','standardize','ddof','cluster_counts','n_init','max_iter'}:raise ValueError('Explicit clustering controls required')
    z=embedding.lookup(training_ids);v=embedding.lookup(query_ids)
    if z.shape!=(len(x),3) or v.shape!=(len(q),3):raise ValueError('Aligned three-dimensional coordinates required')
    def block(a,b,supervised):
        return AlignedBlock(a,b,tuple(training_ids),tuple(query_ids),f.copy() if supervised else None)
    raw=[crossfit_distances(x,labels,f,training_ids,q,query_ids,metric=metric) for metric in raw_metrics]
    supplemental={entry:block(np.column_stack([r['oof'][:,entry-1,:] for r in raw]),np.column_stack([r['query'][:,entry-1,:] for r in raw]),True) for entry in (1,2,3)}
    tfidf=[crossfit_tfidf_distances(x,labels,f,training_ids,q,query_ids,metric=metric,**tfidf_controls) for metric in tfidf_metrics]
    supplemental[4]=block(np.column_stack([r['oof'] for r in tfidf]),np.column_stack([r['query'] for r in tfidf]),True)
    embedded=[crossfit_distances(z,labels,f,training_ids,v,query_ids,metric=metric) for metric in embedding_metrics]
    supplemental[5]=block(np.column_stack([r['oof'][:,0,:] for r in embedded]),np.column_stack([r['query'][:,0,:] for r in embedded]),True)
    clusters=fit_clusters(x,seed=seed,**cluster_controls)
    supplemental[6]=block(clusters.transform(x)['assignments'],clusters.transform(q)['assignments'],False)
    supplemental[7]=block(np.count_nonzero(x,axis=1)[:,None],np.count_nonzero(q,axis=1)[:,None],False)
    return dict(supplemental=supplemental,raw_neural=block(x,q,False))
