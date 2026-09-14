import numpy as np
import pandas as pd
import pytest
from sciona.m5_training import fit,parameters


def test_source_family_settings():
    for recursive in (False,True):
        for pooling in ('outlet','outlet_category','outlet_department'):
            p=parameters(recursive,pooling)
            assert p['seed']==(42 if recursive else 1995)
            assert p['objective']=='tweedie'
            assert p['num_leaves']==(2047 if recursive and pooling=='outlet' else 255)
            assert p['min_data_in_leaf']==(4095 if recursive and pooling=='outlet' else 255)


@pytest.mark.parametrize('recursive',[False,True])
def test_native_full_rounds_diagnostic_masks_and_reload(recursive):
    rng=np.random.default_rng(883)
    days=np.repeat(np.arange(710,778),20)
    x=rng.normal(size=len(days))
    frame=pd.DataFrame({'x':x,'category':pd.Categorical(np.arange(len(days))%3)})
    y=np.maximum(.2,2+x*.4)
    if not recursive:y[days>749]=np.nan
    model,report=fit(frame,y,days,749,recursive=recursive,pooling='outlet_category',first_day=710)
    assert report['evaluated_rounds']==3000 and report['training_rows']==800
    assert report['overlapping_rows']==(560 if recursive else 0)
    assert report['missing_diagnostic_targets']==(0 if recursive else 560)
    import lightgbm as lgb
    reloaded=lgb.Booster(model_str=model.model_to_string())
    pred=model.predict(frame.iloc[:20]);actual=reloaded.predict(frame.iloc[:20])
    np.testing.assert_equal(pred,actual)
    assert np.isfinite(pred).all() and (pred>0).all()


def test_missing_training_labels_reject():
    with pytest.raises(ValueError,match='training targets'):
        fit(pd.DataFrame({'x':[1.,2.]}),[np.nan,1.],[0,1],1,recursive=True,pooling='outlet',first_day=0)
