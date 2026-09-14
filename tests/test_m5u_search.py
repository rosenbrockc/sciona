import numpy as np
import pandas as pd
import pytest
from sciona.m5u_search import search


def test_native_grouped_search_and_full_refit_reload():
    rng=np.random.default_rng(612);x=rng.normal(size=(240,3));groups=np.repeat([1,2,3],80)
    frame=pd.DataFrame(x,columns=['a','b','c']);y=2+x[:,0]+rng.normal(0,.1,240)
    model,report=search(frame,y,groups,.25,seed=17)
    assert report['candidates']==4 and report['candidate_fits']==12 and report['refits']==1
    assert report['best_loss']==min(report['candidate_losses'])
    assert report['best_index']==int(np.argmin(report['candidate_losses']))
    assert model.get_params()['objective']=='quantile' and model.get_params()['alpha']==.25
    import lightgbm as lgb
    reloaded=lgb.Booster(model_str=model.booster_.model_to_string())
    np.testing.assert_allclose(model.predict(frame),reloaded.predict(frame),rtol=0,atol=0)


def test_insufficient_group_population_rejects():
    with pytest.raises(ValueError,match='two groups'):
        search(pd.DataFrame({'x':[1.,2.]}),[1.,2.],[1,1],.5)
