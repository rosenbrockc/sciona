"""Synthetic-only complete Cornell graph input."""
import numpy as np


def payload(length=960001):
    records=[dict(key=f'synthetic-{i}',waveform=(.2*np.sin(np.arange(length)*(.021+i*.007))).astype(np.float32),sample_rate=32000,primary=i,secondary=[]) for i in range(5)]
    populations=[]
    for group,count in [('four',4),('five',5)]:
        for fold in range(count):populations.append(dict(group=group,fold=fold,training=[r for i,r in enumerate(records[:count]) if i!=fold],validation=[records[fold]]))
    noise=(.15*np.sin(np.arange(32000)*.13)+.04).astype(np.float32)
    return dict(version=1,populations=populations,background=[noise],short_noises=[noise],
        inference=dict(sample_rate=32000,waveform=records[0]['waveform'][:960000]),epochs=2,batch_size=2,seed=13,initialization='synthetic')
