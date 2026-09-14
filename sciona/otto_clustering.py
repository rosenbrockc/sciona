"""Explicit-reference k-means feature banks for Otto numeric representations."""
from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sciona.otto_preprocessing import representation,fit_scaling


@dataclass(repr=False)
class ClusterBank:
    kind: str
    scaling: object
    models: list

    def transform(self, values):
        x=self.scaling.transform(values) if self.scaling is not None else representation(values,kind=self.kind)
        assignments=[];distances=[]
        for model in self.models:
            if x.shape[1]!=model.n_features_in_:raise ValueError('Cluster feature width differs')
            distances.append(model.transform(x))
            assignments.append(model.predict(x))
        if not all(np.isfinite(d).all() for d in distances):raise ValueError('Nonfinite cluster distances')
        return dict(assignments=np.column_stack(assignments),distances=np.concatenate(distances,axis=1))


def fit_clusters(reference,*,kind,standardize,ddof,cluster_counts,seed,n_init,max_iter):
    if kind not in ('raw','log1p') or type(standardize) is not bool:
        raise ValueError('Explicit raw/log1p representation and scaling flag required')
    if type(ddof) is not int or ddof not in (0,1):raise ValueError('Explicit scaling convention required')
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Unsigned seed required')
    if any(type(v) is not int or v<1 for v in (n_init,max_iter)):
        raise ValueError('Positive fit iteration controls required')
    if type(cluster_counts) is not list or not cluster_counts or any(type(k) is not int or k<2 for k in cluster_counts) or len(set(cluster_counts))!=len(cluster_counts):
        raise ValueError('Distinct cluster counts at least two required')
    scaling=fit_scaling(reference,kind=kind,ddof=ddof) if standardize else None
    x=scaling.transform(reference) if scaling is not None else representation(reference,kind=kind)
    if max(cluster_counts)>len(x):raise ValueError('More clusters than fitting rows')
    if max(cluster_counts)>len(np.unique(x,axis=0)):raise ValueError('Insufficient distinct fitting points')
    models=[]
    for k in cluster_counts:
        model=KMeans(n_clusters=k,init='k-means++',n_init=n_init,max_iter=max_iter,tol=1e-4,random_state=seed,algorithm='lloyd')
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(x)
        if not np.isfinite(model.cluster_centers_).all():raise ValueError('Nonfinite cluster centers')
        models.append(model)
    return ClusterBank(kind,scaling,models)
