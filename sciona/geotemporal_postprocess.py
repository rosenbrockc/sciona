"""Nonnegative clipping then simultaneous-time geographic neighbor smoothing."""
import math
import numpy as np
from sciona.geotemporal_features import rows


def smooth(points,predictions,*,radius,strength):
    rows(points,())
    values=np.asarray(predictions,dtype=np.float64)
    if values.shape!=(len(points),) or not np.isfinite(values).all():raise ValueError('Aligned finite predictions required')
    if type(radius) not in (int,float) or not math.isfinite(radius) or radius<=0:raise ValueError('Positive radius required')
    if type(strength) not in (int,float) or not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Strength must lie in [0,1]')
    values=np.maximum(values,0.);result=[]
    for i,point in enumerate(points):
        neighbors=[j for j,other in enumerate(points) if j!=i and point['time']==other['time'] and math.hypot(point['x']-other['x'],point['y']-other['y'])<=radius]
        average=sum(values[j]/len(neighbors) for j in neighbors) if neighbors else values[i]
        result.append(float((1-strength)*values[i]+strength*average))
    if not np.isfinite(result).all():raise ValueError('Nonfinite smoothed predictions')
    return result
