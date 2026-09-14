"""Two explicit k-means assignment features in a fixed t-SNE coordinate frame.

Five output columns are features, not nine class probabilities. Historical
cluster counts, encoding and reference-population choices remain unresolved.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from sciona.otto_tsne import PopulationEmbedding


@dataclass(frozen=True,repr=False)
class EmbeddingClusters:
    embedding: PopulationEmbedding
    centers: tuple

    def lookup(self, identities):
        coordinates=self.embedding.lookup(identities)
        assignments=[]
        for centers in self.centers:
            distances=np.square(coordinates[:,None,:]-centers[None,:,:]).sum(axis=2)
            if not np.isfinite(distances).all():raise ValueError('Nonfinite cluster distances')
            assignments.append(distances.argmin(axis=1))
        return np.column_stack((coordinates,*assignments))


def fit_embedding_clusters(embedding, reference_ids, *, cluster_counts, seed, n_init, max_iter):
    if not isinstance(embedding,PopulationEmbedding):raise ValueError('Fitted population embedding required')
    if type(cluster_counts) is not list or len(cluster_counts)!=2 or any(type(k) is not int or k<2 for k in cluster_counts) or len(set(cluster_counts))!=2:
        raise ValueError('Exactly two distinct cluster counts required')
    if type(seed) is not int or not 0<=seed<2**32 or any(type(v) is not int or v<1 for v in (n_init,max_iter)):
        raise ValueError('Valid seed and positive fitting controls required')
    x=embedding.lookup(reference_ids)
    if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all() or len(np.unique(x,axis=0))<max(cluster_counts):
        raise ValueError('Finite distinct three-dimensional reference points required')
    centers=[]
    for count in cluster_counts:
        model=KMeans(n_clusters=count,init='k-means++',n_init=n_init,max_iter=max_iter,tol=1e-4,random_state=seed,algorithm='lloyd')
        with warnings.catch_warnings(),threadpool_limits(limits=1):
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(x)
        center=model.cluster_centers_.copy()
        if not np.isfinite(center).all():raise ValueError('Nonfinite fitted centers')
        center.setflags(write=False);centers.append(center)
    return EmbeddingClusters(embedding,tuple(centers))
