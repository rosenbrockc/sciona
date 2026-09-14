"""Independent outer-group quantile bags with nested grouped model selection."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sciona.m5u_sampling import quantile_mask
from sciona.m5u_search import search
from sciona.m5u_loss import pinball


@dataclass(repr=False)
class BagModel:
    bag: int
    group: int
    quantile: float
    training_rows: np.ndarray
    holdout_rows: np.ndarray
    model: object
    report: dict


def train(bags,quantiles,quantile_weights,level,*,single_fold=False,fast=False,iterations=4,seed=0):
    """Each bag supplies independently sampled/scaled features, targets and groups."""
    if (not isinstance(bags,(list,tuple)) or not bags or not isinstance(quantiles,(list,tuple)) or not quantiles
            or len(set(quantiles))!=len(quantiles) or len(quantile_weights)!=len(quantiles)
            or type(single_fold) is not bool or type(seed) is not int or not 0<=seed<2**31):
        raise ValueError('Invalid bags, quantile inventory or controls')
    for q in quantiles:pinball([0.],[0.],q)
    rng=np.random.RandomState(seed);plans=[]
    for bag,data in enumerate(bags):
        frame=data['features'];y=np.asarray(data['targets']);groups=np.asarray(data['groups'])
        if (not isinstance(frame,pd.DataFrame) or frame.empty or y.shape!=(len(frame),)
                or groups.shape!=y.shape or groups.dtype.kind not in 'iu' or len(np.unique(groups))<3):
            raise ValueError('Each bag needs aligned data and three groups for nested holdouts')
        pinball(y,y,.5)
        heldout=np.unique(groups)[-1:] if single_fold else np.unique(groups)
        for group in heldout:
            validation=np.flatnonzero(groups==group)
            for q,weight in zip(quantiles,quantile_weights):
                mask_seed=int(rng.randint(0,2**31));model_seed=int(rng.randint(0,2**31))
                training=np.flatnonzero(quantile_mask(groups,int(group),weight,level,seed=mask_seed))
                if len(np.unique(groups[training]))<2:
                    raise ValueError('Quantile subsample lacks two inner-search groups')
                assert not np.intersect1d(training,validation).size
                plans.append((bag,int(group),float(q),training,validation,model_seed))
    result=[]
    for bag,group,q,training,validation,model_seed in plans:
        data=bags[bag];y=np.asarray(data['targets']);groups=np.asarray(data['groups'])
        learner,report=search(data['features'].iloc[training],y[training],groups[training],q,
                              fast=fast,iterations=iterations,seed=model_seed)
        predictions=learner.predict(data['features'].iloc[validation])
        report=dict(report,outer_holdout_loss=pinball(y[validation],predictions,q),
                    outer_holdout_rows=len(validation),outer_training_rows=len(training),outer_group_excluded=True)
        result.append(BagModel(bag,group,q,training,validation,learner,report))
    return result
