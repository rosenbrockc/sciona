"""Generated period transactions only; no competition records or identities."""


def payload():
    entities=[dict(store=i//4,item=i%4,category=(i%4)//2) for i in range(8)]
    rows=[]
    for period in range(9):
        for i,e in enumerate(entities):
            rows.append(dict(day=10*period+1,store=e['store'],item=e['item'],quantity=(i*3+period*2)%21))
    return dict(version=1,transactions=rows,entities=entities,boundaries=list(range(0,101,10)),
        feature_controls=dict(lags=[1,2],rolling_windows=[2],calendar_cycle=12,cold_start=0.),
        training_controls=dict(train_start=2,validation_start=7,trials=3,startup_trials=1,max_rounds=20,stopping_rounds=3,seed=92,
            search_space=dict(num_leaves=[3,8],min_data_in_leaf=[2,5],learning_rate=[.05,.2])))
