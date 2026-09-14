import math
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sciona.nab_scoring import evaluate
from sciona.nab_streaming import detect

COSTS=dict(tpWeight=1.,fpWeight=.11,fnWeight=1.)
CONFIG=dict(window_size=3,fit_history=8,min_fit_windows=3,threshold_history=7,min_threshold_scores=3,n_estimators=11,max_samples=8,seed=42,std_ddof=0)


def test_threshold_and_singleton_corrections():
    low=evaluate([.8]*8,[(2,4)],threshold=.2,probation_percent=0.,costs=COSTS)
    assert low['counts']==dict(tp=3,tn=0,fp=5,fn=0)
    single=evaluate([0,0,1,1],[(2,2)],threshold=.5,probation_percent=0.,costs=COSTS)
    assert single['raw_score']==pytest.approx(1-.11*math.tanh(2.5))
    assert single['normalized_score']==pytest.approx(100*(single['raw_score']+1)/2)


def test_no_windows_has_no_normalization():
    r=evaluate([0,1],[],threshold=.5,probation_percent=0.,costs=COSTS)
    assert r['raw_score']==-.11 and r['normalized_score'] is None


def test_probation_preserves_source_normalization():
    r=evaluate([1.]*10,[(0,1),(3,5)],threshold=.5,probation_percent=.4,costs=COSTS)
    assert r['scorable_windows']==1 and r['perfect_score']==2 and r['null_score']==-1 and r['scored_points']==6


@pytest.mark.parametrize('windows',[[(2,3),(3,4)],[(4,5),(1,2)],[(-1,2)],[(1,8)],[(1.,2)]])
def test_bad_windows(windows):
    with pytest.raises(ValueError):evaluate([0.]*8,windows,threshold=.5,probation_percent=0.,costs=COSTS)


def test_prefix_and_future_mutation_invariance():
    rng=np.random.default_rng(111);x=rng.normal(size=34);x[27]=12
    full=detect(x,CONFIG);prefix=detect(x[:25],CONFIG)
    modified=x.copy();modified[25:]=rng.normal(500,20,size=9)
    future=detect(modified,CONFIG)
    for key in ['anomaly_scores','thresholds','detections']:
        assert full[key][:25]==prefix[key]==future[key][:25]
    assert full['models_fitted']==29


def test_training_excludes_current(monkeypatch):
    from sklearn.ensemble import IsolationForest
    original=IsolationForest.fit;seen=[]
    def fit(self,x,*args,**kwargs):seen.append(x.copy());return original(self,x,*args,**kwargs)
    monkeypatch.setattr(IsolationForest,'fit',fit)
    detect([0.,1.,2.,3.,4.,999.],CONFIG)
    assert len(seen)==1 and not np.any(seen[0]==999.)
    assert np.array_equal(seen[0],[[0,1,2],[1,2,3],[2,3,4]])
