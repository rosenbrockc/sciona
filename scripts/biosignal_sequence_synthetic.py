"""Entirely synthetic periodic signals for execution validation."""
import math

def payload():
    def population(prefix,count,phase,labeled=True):
        labels=[i%2 for i in range(count)]
        signals=[[[math.sin(2*math.pi*(4 if y==0 else 12)*j/64+phase+.2*i) for j in range(64+16*i)]] for i,y in enumerate(labels)]
        result=dict(signals=signals,subjects=[f'{prefix}{i}' for i in range(count)])
        if labeled:result['labels']=labels
        return result
    return dict(version=1,training=population('train',4,0.),calibration=population('cal',4,.1),query=population('query',2,.3,False),
        controls=dict(sample_rate=64.,window_size=32,stride=16,n_fft=16,hop=8,seed=12,width=4,epochs=40,batch_size=3,learning_rate=.02,max_frequency=1,max_time=1))
