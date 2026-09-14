"""Generated synthetic replay fixture; no external data."""
import numpy as np

def replay(identity, seed, rounds=80, constant=None):
    rng=np.random.default_rng(seed)
    thresholds=rng.integers(1,101,size=100) if constant is None else np.full(100,constant)
    counts=np.zeros(100,dtype=int);actions=[];rewards=[]
    for _ in range(rounds):
        pair=rng.integers(0,100,size=2).tolist()
        outcomes=[int(rng.integers(0,101)<thresholds[a]*.97**counts[a]) for a in pair]
        actions.append(pair);rewards.append(outcomes)
        for a in pair:counts[a]+=1
    return dict(episode_id=identity,thresholds=thresholds.tolist(),actions=actions,rewards=rewards,perspectives=[0,1])


def payload():
    return dict(version=1,training=[replay('synthetic-train',1)],validation=[replay('synthetic-valid',2)],
                query=dict(episode_id='synthetic-query',history=[dict(own_action=0,opponent_action=1,reward=1)]),
                controls=dict(seed=42,decision_seed=12,max_rows=10000,num_boost_round=24,stopping_rounds=4,num_leaves=15))

