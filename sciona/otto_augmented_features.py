"""Explicit realization of Otto model-15 raw/cluster/row-statistic features.

Seven cluster assignments are used as seven numeric columns. The historical
cluster counts, encoding and fitting population are not fully specified.
"""
from dataclasses import dataclass
import numpy as np
from sciona.otto_preprocessing import representation,fit_scaling
from sciona.otto_clustering import fit_clusters


@dataclass(repr=False)
class Augmentation:
    scaling: object
    clusters: object

    def transform(self,values):
        raw=representation(values,kind='raw')
        standardized=self.scaling.transform(values)
        cluster_ids=self.clusters.transform(values)['assignments']
        statistics=np.column_stack(((raw==0).sum(axis=1),(standardized>.5).sum(axis=1),(standardized<-.5).sum(axis=1)))
        result=np.concatenate((raw,cluster_ids,statistics),axis=1)
        if not np.isfinite(result).all():raise ValueError('Nonfinite augmented features')
        return result


def fit_augmentation(reference,*,cluster_counts,ddof,seed,n_init,max_iter):
    if type(cluster_counts) is not list or len(cluster_counts)!=7:
        raise ValueError('Exactly seven cluster counts required')
    scaling=fit_scaling(reference,kind='raw',ddof=ddof)
    clusters=fit_clusters(reference,kind='raw',standardize=False,ddof=ddof,
                          cluster_counts=cluster_counts,seed=seed,n_init=n_init,max_iter=max_iter)
    return Augmentation(scaling,clusters)
