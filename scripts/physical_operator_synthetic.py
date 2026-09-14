"""Analytic periodic heat-equation fixtures; no observed data."""
import math


def payload():
    grid=[i/16 for i in range(16)]
    def population(start,count,labels=True):
        states=[];targets=[];diff=[];elapsed=[]
        for i in range(start,start+count):
            mean=1.+.03*(i%4);amplitude=.2+.01*(i%3);phase=i*.37
            tau=.005+.002*(i%5)
            states.append([mean+amplitude*math.cos(2*math.pi*x+phase) for x in grid])
            targets.append([mean+amplitude*math.exp(-4*math.pi**2*tau)*math.cos(2*math.pi*x+phase) for x in grid])
            diff.append(.1);elapsed.append(tau/.1)
        result=dict(states=states,diffusivity=diff,elapsed=elapsed,groups=[f'synthetic{i}' for i in range(start,start+count)])
        if labels:result['targets']=targets
        return result
    return dict(version=1,units=dict(length='m',time='s',diffusivity='m^2/s',state='dimensionless'),length=1.,grid=grid,
        training=population(0,8),validation=population(8,4),query=population(12,2,False),
        controls=dict(seed=12,width=8,modes=4,depth=2,epochs=60,learning_rate=.01,smoothing=.1))
