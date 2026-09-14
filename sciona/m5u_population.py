"""Align history features, hierarchy categories and sampling weights in day order."""
import numpy as np
import pandas as pd
from sciona.m5u_rolling import blocks
from sciona.m5u_cleaning import retained_rows,missing_sentinel
from sciona.m5u_weights import statistics


def build(history,scaled_history,revenue,roles,levels,calendar_years,calendar_months,minimum_day,*,reduced=False):
    history,roles,levels=np.asarray(history),np.asarray(roles),np.asarray(levels)
    if (history.ndim!=2 or roles.shape!=(history.shape[1],5) or roles.dtype.kind not in 'iu'
            or levels.shape!=(history.shape[1],) or levels.dtype.kind not in 'iu'
            or len(calendar_years)<len(history)):
        raise ValueError('Aligned history, hierarchy, levels and complete calendar required')
    history_blocks=blocks(history,scaled_history,reduced=reduced)
    weight_blocks=statistics(history,revenue,levels)
    selected=retained_rows(np.arange(len(history)),calendar_years,calendar_months,minimum_day,stacked=False)
    # Stacked weight cleaning always applies the boundary. Detect the source
    # pivot/stack asymmetry rather than silently joining mismatched populations.
    weight_selected=retained_rows(np.arange(len(history)),calendar_years,calendar_months,minimum_day)
    if not np.array_equal(selected,weight_selected):raise ValueError('History and weight cleaning populations differ')
    if not selected.size:raise ValueError('Calendar cleaning removed all history rows')
    series_count=history.shape[1]
    days=np.repeat(selected,series_count);series=np.tile(np.arange(series_count),len(selected))
    frame=pd.DataFrame({name:missing_sentinel(value[selected]).ravel() for name,value in history_blocks.items()})
    for role in (3,2,1,0):
        frame[f'role_{role}']=pd.Categorical(roles[series,role],categories=np.unique(roles[:,role]))
    weights=weight_blocks['weights'].copy()
    weights[weight_blocks['observations']<30]=0
    return dict(features=frame,days=days,series=series,levels=levels[series].copy(),
        weights=weights[selected].ravel(),volumes=weight_blocks['volatility'][selected].ravel(),
        scaled_columns=tuple(name for name in history_blocks if name.startswith(('raw_','scaled_','lag_'))),
        ewm_columns=tuple(name for name in history_blocks if name.startswith('raw_ewm_')))
