"""Synthetic-only raw-signal fixture for the full VSB execution boundary."""
import numpy as np


def payload():
    rng=np.random.default_rng(101);labels=np.zeros((200,3),int);labels[100:,0]=1
    training=rng.normal(size=(512,600));query=rng.normal(size=(512,24))
    return dict(version=1,training=dict(signals=training.tolist(),labels=labels.tolist(),identities=[f'r{i}' for i in range(200)]),query=dict(signals=query.tolist(),identities=[f'q{i}' for i in range(8)]))
