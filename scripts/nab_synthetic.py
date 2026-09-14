"""Generated numeric stream; no benchmark or real records."""
import numpy as np


def payload():
    rng=np.random.default_rng(31);x=rng.normal(size=100);x[65:69]+=10
    return dict(version=1,values=x.tolist(),detector=dict(window_size=3,fit_history=20,min_fit_windows=8,
        threshold_history=12,min_threshold_scores=5,n_estimators=31,max_samples=20,seed=82,std_ddof=0),
        windows=[[64,70]],probation_percent=.15,costs=dict(tpWeight=1.,fpWeight=.11,fnWeight=1.))
