"""Synthetic-only raw Amex population with runtime category and column roles."""
import numpy as np


def payload():
    rng=np.random.default_rng(119);labels=[0]*800+[1]*800
    def population(y):
        return dict(numerics=[rng.normal(loc=2*v,scale=.3,size=(13,2)).tolist() for v in y],
            categories=[[[int(rng.integers(0,2)),str(rng.choice(['alpha','beta']))] for _ in range(13)] for v in y],
            times=[list(range(13)) for v in y],months=[list(range(13)) for v in y])
    return dict(version=1,training=dict(sequences=population(labels),labels=labels,identities=[f'r{i}' for i in range(len(labels))]),
        query=dict(sequences=population([0,1]*4),identities=[f'q{i}' for i in range(8)]),
        controls=dict(category_rules=[None,dict(values={'alpha':0,'beta':1},missing=-1)],zero_fill_columns=[0],seed=42))
