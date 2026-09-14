"""Bounded streaming temporal features with fixed-width categorical hashing."""
from collections import deque
import math
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction import FeatureHasher


def validate_event(event,*,labeled):
    fields={'entity','time','value','category'}|({'target'} if labeled else set())
    if type(event) is not dict or set(event)!=fields:raise ValueError('Invalid event fields')
    if any(type(event[k]) is not str or not event[k] for k in ('entity','category')):raise ValueError('Nonempty entity/category required')
    for key in ('time','value')+ (('target',) if labeled else ()):
        v=event[key]
        if type(v) not in (int,float):raise ValueError('Finite numeric event required')
        try:valid=math.isfinite(v)
        except OverflowError:valid=False
        if not valid:raise ValueError('Finite numeric event required')


class TemporalSparseEncoder:
    """One-pass features; labels become available only after their timestamp.

Same-time pending labels survive output chunk boundaries. Explicit entity,
per-entity history and pending-label limits fail closed rather than evict data.
Targets are assumed available immediately after their observation timestamp.
"""
    def __init__(self,*,hash_size=128,lookback=100.,period=24.,max_entities=1000,max_history=1000,max_pending=1000):
        for v in (hash_size,max_entities,max_history,max_pending):
            if type(v) is not int or v<1:raise ValueError('Positive integer capacity required')
        for v in (lookback,period):
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0:raise ValueError('Positive finite time scale required')
        self.hasher=FeatureHasher(n_features=hash_size,input_type='dict',alternate_sign=False)
        self.lookback=lookback;self.period=period;self.max_entities=max_entities
        self.max_history=max_history;self.max_pending=max_pending
        self.history={};self.pending=[];self.time=None

    def _advance(self,time):
        if self.time is not None and time<self.time:raise ValueError('Events must be globally ordered by time')
        if self.time is not None and time>self.time:
            # Validate capacities before changing history.
            additions={}
            for entity,_,_ in self.pending:additions[entity]=additions.get(entity,0)+1
            if len(set(self.history)|set(additions))>self.max_entities:raise ValueError('Entity capacity exceeded')
            for entity,count in additions.items():
                retained=sum(t>=time-self.lookback for t,_ in self.history.get(entity,()))
                incoming=sum(t>=time-self.lookback for e,t,_ in self.pending if e==entity)
                if retained+incoming>self.max_history:raise ValueError('Per-entity history capacity exceeded')
            for history in self.history.values():
                while history and history[0][0]<time-self.lookback:history.popleft()
            for entity,t,target in self.pending:
                history=self.history.setdefault(entity,deque())
                if t>=time-self.lookback:history.append((t,target))
            self.pending=[]
        self.time=time

    def encode(self,events,*,chunk_size,labeled):
        if type(chunk_size) is not int or chunk_size<1 or type(labeled) is not bool:raise ValueError('Invalid streaming controls')
        dense=[];categories=[];targets=[]
        for event in events:
            validate_event(event,labeled=labeled)
            self._advance(event['time'])
            if labeled and len(self.pending)>=self.max_pending:raise ValueError('Same-time pending capacity exceeded')
            history=self.history.get(event['entity'],())
            mean=sum(v/len(history) for _,v in history) if history else 0.
            latest=[v for t,v in history if t==history[-1][0]] if history else []
            last=sum(v/len(latest) for v in latest) if latest else 0.
            age=event['time']-history[-1][0] if history else 0.
            phase=2*math.pi*((event['time']%self.period)/self.period)
            dense.append([event['value'],math.sin(phase),math.cos(phase),mean,last,age,float(len(history)),float(bool(history))])
            categories.append({'entity:'+event['entity']:1.,'category:'+event['category']:1.})
            if labeled:
                self.pending.append((event['entity'],event['time'],event['target']));targets.append(event['target'])
            if len(dense)==chunk_size:
                yield self._matrix(dense,categories),np.asarray(targets,dtype=np.float64) if labeled else None
                dense=[];categories=[];targets=[]
        if dense:yield self._matrix(dense,categories),np.asarray(targets,dtype=np.float64) if labeled else None

    def _matrix(self,dense,categories):
        values=np.asarray(dense,dtype=np.float64)
        if not np.isfinite(values).all():raise ValueError('Nonfinite temporal features')
        return hstack((csr_matrix(values),self.hasher.transform(categories)),format='csr')
