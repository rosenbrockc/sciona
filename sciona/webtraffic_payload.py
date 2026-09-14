"""Versioned runtime-only payload codec; no dataset or checkpoint persistence."""
from dataclasses import fields
import datetime
import numpy as np
import pandas as pd
from .webtraffic_runtime import RuntimeConfig,prepare_runtime,train_runtime,forecast_runtime


def decode_payload(payload):
    if not isinstance(payload,dict) or set(payload)!={'version','pages','start_day','counts','config'} or type(payload['version']) is not int or payload['version']!=1:
        raise ValueError('Version1 Web Traffic payload required')
    pages=payload['pages'];rows=payload['counts'];options=payload['config']
    if (not isinstance(pages,list) or not pages or any(not isinstance(p,str) or not p for p in pages)
        or len(set(pages))!=len(pages) or pages!=sorted(pages)):
        raise ValueError('Unique sorted runtime page strings required')
    if not isinstance(rows,list) or len(rows)!=len(pages) or any(not isinstance(r,list) for r in rows):
        raise ValueError('One numeric row per runtime page required')
    days=len(rows[0])
    if days==0 or any(len(row)!=days for row in rows):raise ValueError('Rectangular daily count rows required')
    if any(v is not None and (type(v) not in (int,float) or not np.isfinite(v) or v<0) for row in rows for v in row):
        raise ValueError('Nonnegative finite counts or null required')
    if not isinstance(options,dict) or set(options)-{f.name for f in fields(RuntimeConfig)}:
        raise ValueError('Unknown runtime configuration field')
    day=payload['start_day']
    if not isinstance(day,str) or datetime.date.fromisoformat(day).isoformat()!=day:
        raise ValueError('ISO calendar start day required')
    with np.errstate(over='ignore',invalid='ignore'):values=np.array(rows,dtype=np.float32)
    if np.isinf(values).any():raise ValueError('Count magnitude exceeds float32 representation')
    frame=pd.DataFrame(values,index=np.array(pages,dtype=object),columns=pd.date_range(day,periods=days))
    return prepare_runtime(frame,RuntimeConfig(**options))


def train_payload(prepared):
    return dict(prepared=prepared,trained=train_runtime(prepared))


def encode_forecast(completed):
    frame=forecast_runtime(completed['prepared'],completed['trained'])
    return dict(version=1,pages=frame.index.tolist(),days=[day.date().isoformat() for day in frame.columns],
                forecasts=frame.to_numpy().tolist())


def execute_payload(payload):
    return encode_forecast(train_payload(decode_payload(payload)))
