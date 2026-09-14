"""Strict private JSON boundary for the generic one-period demand pipeline."""
import copy
from dataclasses import dataclass
import json
import math
import os
import subprocess
import sys
from sciona.future_sales_features import build_panel
_FIELDS={'version','transactions','entities','boundaries','feature_controls','training_controls'}
_FEATURES={'lags','rolling_windows','calendar_cycle','cold_start'}


@dataclass(frozen=True)
class Prepared:
    payload: dict


def prepare(payload):
    if not isinstance(payload,dict) or set(payload)!=_FIELDS:raise ValueError('Exact Future Sales payload fields required')
    try:json.dumps(payload,allow_nan=False)
    except (ValueError,TypeError,OverflowError):raise ValueError('Finite JSON payload required') from None
    if type(payload['version']) is not int or payload['version']!=1:raise ValueError('Unsupported payload version')
    if not isinstance(payload['feature_controls'],dict) or set(payload['feature_controls'])!=_FEATURES:raise ValueError('Exact feature controls required')
    if not isinstance(payload['training_controls'],dict):raise ValueError('Explicit training controls required')
    build_panel(payload['transactions'],payload['entities'],payload['boundaries'],**payload['feature_controls'])
    return Prepared(copy.deepcopy(payload))


def execute(prepared):
    if not isinstance(prepared,Prepared):raise ValueError('Prepared Future Sales input required')
    p=prepare(prepared.payload).payload
    try:
        timeout=float(os.environ.get('SCIONA_FUTURE_SALES_TIMEOUT_SECONDS','3600'))
        if not math.isfinite(timeout) or timeout<=0:raise ValueError()
    except (ValueError,TypeError):raise ValueError('Positive finite worker timeout required') from None
    try:
        completed=subprocess.run([sys.executable,'-m','sciona.future_sales_worker'],input=json.dumps(p,allow_nan=False),
            text=True,capture_output=True,timeout=timeout,check=False)
    except (OSError,subprocess.TimeoutExpired):raise RuntimeError('Isolated Future Sales worker failed or timed out') from None
    if completed.returncode!=0:raise RuntimeError('Isolated Future Sales worker failed')
    try:result=json.loads(completed.stdout)
    except (ValueError,TypeError):raise RuntimeError('Invalid Future Sales worker response') from None
    fields={'predictions','trials','selected_trial','selected_rounds','training_rows','validation_rows','refit_rows','forecast_rows'}
    if not isinstance(result,dict) or set(result)!=fields:raise RuntimeError('Invalid Future Sales worker fields')
    predictions=result['predictions']
    if (not isinstance(predictions,list) or len(predictions)!=len(p['entities'])
            or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=20 for v in predictions)):
        raise RuntimeError('Invalid Future Sales worker predictions')
    return result
