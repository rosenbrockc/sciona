import copy
import numpy as np
import pytest
from sciona.future_sales_features import build_panel
from sciona.future_sales_training import train_forecast


def inputs():
    entities=[dict(store=i//4,item=i%4,category=(i%4)//2) for i in range(8)]
    rows=[]
    for period in range(9):
        for i,e in enumerate(entities):
            rows.append(dict(day=10*period+1,store=e['store'],item=e['item'],quantity=(i*3+period*2)%21))
    panel=build_panel(rows,entities,list(range(0,101,10)),lags=[1,2],rolling_windows=[2],calendar_cycle=12,cold_start=0.)
    config=dict(train_start=2,validation_start=7,trials=3,startup_trials=1,max_rounds=20,stopping_rounds=3,seed=92,
        search_space=dict(num_leaves=[3,8],min_data_in_leaf=[2,5],learning_rate=[.05,.2]))
    return panel,config


def test_complete_bayesian_early_stopping_and_refit():
    panel,config=inputs();r=train_forecast(panel,config)
    assert len(r['trials'])==3 and 1<=r['selected_rounds']<=20
    assert (r['training_rows'],r['validation_rows'],r['refit_rows'],r['forecast_rows'])==(40,16,56,8)
    assert np.isfinite(r['predictions']).all() and np.all((r['predictions']>=0)&(r['predictions']<=20))
    repeated=train_forecast(panel,config)
    assert np.array_equal(r['predictions'],repeated['predictions']) and r['trials']==repeated['trials']


@pytest.mark.parametrize('case',['no_bayesian','overlap','early_stopping','bad_search','forecast_target'])
def test_invalid_training(case):
    panel,config=inputs();config=copy.deepcopy(config)
    if case=='no_bayesian':config['trials']=1
    elif case=='overlap':config['validation_start']=2
    elif case=='early_stopping':config['stopping_rounds']=20
    elif case=='bad_search':config['search_space']['num_leaves']=[1,8]
    else:panel.targets[-1]=1.
    with pytest.raises(ValueError):train_forecast(panel,config)


def test_tpe_model_based_sampling_is_exercised(monkeypatch):
    from optuna.samplers import TPESampler
    original=TPESampler._sample;calls=[]
    def sample(self,*args,**kwargs):
        calls.append(1);return original(self,*args,**kwargs)
    monkeypatch.setattr(TPESampler,'_sample',sample)
    panel,config=inputs();train_forecast(panel,config)
    assert len(calls)>0
