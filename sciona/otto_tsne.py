"""Fixed-population three-dimensional log1p t-SNE feature realization.

This is transductive when the declared population contains future query rows.
There is no out-of-sample transform or inferred historical population scope.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
from sciona.otto_preprocessing import representation


@dataclass(frozen=True,repr=False)
class PopulationEmbedding:
    identities: tuple
    coordinates: np.ndarray
    kl_divergence: float

    def lookup(self, identities):
        if type(identities) is not list or not identities or any(type(i) is not str for i in identities) or len(set(identities))!=len(identities):
            raise ValueError('Unique nonempty row identity list required')
        positions={identity:i for i,identity in enumerate(self.identities)}
        if any(i not in positions for i in identities):
            raise ValueError('Row outside fitted population; a new embedding is required')
        return self.coordinates[[positions[i] for i in identities]].copy()


def fit_population(values, identities, *, seed, perplexity, learning_rate, max_iter):
    x=representation(values,kind='log1p')
    if type(identities) is not list or len(identities)!=len(x) or any(type(i) is not str or not i for i in identities) or len(set(identities))!=len(identities):
        raise ValueError('Aligned unique opaque identities required')
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Valid integer seed required')
    if type(perplexity) not in (int,float) or not np.isfinite(perplexity) or not 0<perplexity<len(x):
        raise ValueError('Perplexity must be positive and below population size')
    if type(learning_rate) not in (int,float) or not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('Explicit finite positive learning rate required')
    if type(max_iter) is not int or max_iter<300:
        raise ValueError('At least 300 iterations including post-exaggeration optimization required')
    model=TSNE(n_components=3,perplexity=perplexity,learning_rate=learning_rate,max_iter=max_iter,
               init='random',random_state=seed,method='barnes_hut',angle=.5,metric='euclidean',
               early_exaggeration=12.,n_iter_without_progress=300,min_grad_norm=1e-7,n_jobs=1)
    with threadpool_limits(limits=1):coordinates=model.fit_transform(x)
    if coordinates.shape!=(len(x),3) or not np.isfinite(coordinates).all() or not np.isfinite(model.kl_divergence_):
        raise ValueError('Invalid embedding optimization result')
    coordinates.setflags(write=False)
    return PopulationEmbedding(tuple(identities),coordinates,float(model.kl_divergence_))
