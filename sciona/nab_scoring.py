"""Independent NAB-compatible row-index scoring with explicit boundary fixes.

Singleton windows use one row as their post-window penalty scale, where the
source divides by zero. Direct threshold reduction supports thresholds below
all scores. No detector or historical NAB corpus is embedded.
"""
import math
import numpy as np


def evaluate(scores,windows,*,threshold,probation_percent,costs):
    scores=np.asarray(scores)
    if scores.ndim!=1 or scores.size<1 or scores.dtype.kind not in 'fi' or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ValueError('Finite one-dimensional scores in [0,1] required')
    for value in (threshold,probation_percent):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Finite scalar controls required')
    if not 0<=threshold<=1 or not 0<=probation_percent<1:raise ValueError('Invalid scoring controls')
    if not isinstance(costs,dict) or set(costs)!={'tpWeight','fpWeight','fnWeight'}:raise ValueError('Exact cost weights required')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in costs.values()):raise ValueError('Positive finite cost weights required')
    if not isinstance(windows,(list,tuple)):raise ValueError('Explicit anomaly windows required')
    previous=-1;checked=[]
    for w in windows:
        if not isinstance(w,(list,tuple)) or len(w)!=2 or any(type(i) is not int for i in w):raise ValueError('Integer window bounds required')
        start,end=w
        if not previous<start<=end<len(scores):raise ValueError('Sorted disjoint in-range inclusive windows required')
        checked.append((start,end));previous=end
    probation=min(math.floor(probation_percent*len(scores)),probation_percent*5000)
    score_parts={};fp_score=0.;counts=dict(tp=0,tn=0,fp=0,fn=0)
    current=0;previous_window=None
    for i,value in enumerate(scores):
        while current<len(checked) and i>checked[current][1]:previous_window=checked[current];current+=1
        inside=current<len(checked) and checked[current][0]<=i<=checked[current][1]
        if i<probation:continue
        active=value>=threshold
        if inside:
            start,end=checked[current];score_parts.setdefault(current,-costs['fnWeight'])
            reward=-math.tanh(2.5*(-(end-i+1)/(end-start+1)))/math.tanh(2.5)*costs['tpWeight']
            if active:score_parts[current]=max(score_parts[current],reward)
            counts['tp' if active else 'fn']+=1
        else:
            if previous_window is None:penalty=-costs['fpWeight']
            else:
                start,end=previous_window;position=(i-end)/max(1,end-start)
                penalty=(-1. if position>3 else -math.tanh(2.5*position))*costs['fpWeight']
            if active:fp_score+=penalty
            counts['fp' if active else 'tn']+=1
    raw=math.fsum(score_parts.values())+fp_score
    baseline=-len(score_parts)*costs['fnWeight']
    # Source normalization counts all supplied labeled windows, including those
    # wholly inside probation. Preserve this distinction from the null score.
    perfect=len(checked)*costs['tpWeight'];denominator=perfect-baseline
    normalized=100*(raw-baseline)/denominator if denominator>0 else None
    if not all(math.isfinite(v) for v in [raw,baseline,perfect]) or (normalized is not None and not math.isfinite(normalized)):
        raise ValueError('Scoring overflow')
    return dict(raw_score=raw,null_score=baseline,perfect_score=perfect,normalized_score=normalized,
                normalization_defined=denominator>0,counts=counts,scored_points=sum(counts.values()),scorable_windows=len(score_parts))
