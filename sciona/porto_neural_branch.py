"""Unlabeled joint-population DAE followed by supervised hidden-feature learning."""
import numpy as np
from sciona.porto_transforms import _matrix
from sciona.porto_autoencoder import fit_population
from sciona.porto_neural import fit_predict


def fit_branch(reference,labels,query,*,seed,dae_controls,neural_controls):
    x=_matrix(reference);q=_matrix(query);y=np.asarray(labels)
    if x.shape[1]!=q.shape[1] or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or set(y.tolist())!={0,1}:
        raise ValueError('Aligned binary reference and query populations required')
    # No labels enter the DAE. The caller has already prepared the joint feature
    # population; query participation is explicitly transductive.
    encoder=fit_population(np.vstack((x,q)),seed=seed,controls=dae_controls)
    train_features=encoder.transform(x);query_features=encoder.transform(q)
    result=fit_predict(train_features,y,query_features,seed=seed,controls=neural_controls)
    result.update(learned_width=train_features.shape[1],dae_population_rows=len(x)+len(q),
        dae_initial_clean_mse=encoder.clean_mse[0],dae_final_clean_mse=encoder.clean_mse[-1],
        dae_epochs=len(encoder.noisy_training_mse),reference_rows=len(x),query_rows=len(q))
    return result
