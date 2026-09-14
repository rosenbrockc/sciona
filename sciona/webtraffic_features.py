"""Web Traffic numerical features; MIT source adaptation.

Derived from Arturus/kaggle-web-traffic; see docs/licenses/WebTraffic-MIT.txt.
Source numerical bodies run with NumPy; legacy np.NaN spelling uses np.nan.
"""
import numpy as np
import pandas as pd

def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0

def batch_autocorr(data, lag, starts, ends, threshold, backoffset=0):
    """
    Calculate autocorrelation for batch (many time series at once)
    :param data: Time series, shape [n_pages, n_days]
    :param lag: Autocorrelation lag
    :param starts: Start index for each series
    :param ends: End index for each series
    :param threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
    :param backoffset: Offset from the series end, days.
    :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
    autocorrelation value is NaN
    """
    n_series = data.shape[0]
    n_days = data.shape[1]
    max_end = n_days - backoffset
    corr = np.empty(n_series, dtype=np.float64)
    support = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        series = data[i]
        end = min(ends[i], max_end)
        real_len = end - starts[i]
        support[i] = real_len / lag
        if support[i] > threshold:
            series = series[starts[i]:end]
            c_365 = single_autocorr(series, lag)
            c_364 = single_autocorr(series, lag - 1)
            c_366 = single_autocorr(series, lag + 1)
            corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
        else:
            corr[i] = np.nan
    return corr

def find_start_end(data: np.ndarray):
    """
    Calculates start and end of real traffic data. Start is an index of first non-zero, non-NaN value,
     end is index of last non-zero, non-NaN value
    :param data: Time series, shape [n_pages, n_days]
    :return:
    """
    n_pages = data.shape[0]
    n_days = data.shape[1]
    start_idx = np.full(n_pages, -1, dtype=np.int32)
    end_idx = np.full(n_pages, -1, dtype=np.int32)
    for page in range(n_pages):
        for day in range(n_days):
            if not np.isnan(data[page, day]) and data[page, day] > 0:
                start_idx[page] = day
                break
        for day in range(n_days - 1, -1, -1):
            if not np.isnan(data[page, day]) and data[page, day] > 0:
                end_idx[page] = day
                break
    return (start_idx, end_idx)


def prepare_series(frame, valid_threshold):
    """Source prepare_data numerical path over an explicit runtime table."""
    if not isinstance(frame,pd.DataFrame) or frame.shape[1]==0:
        raise ValueError('Runtime series table with at least one time step required')
    values=frame.to_numpy()
    if values.dtype.kind not in 'if' or np.isinf(values).any() or (values<0).any():
        raise ValueError('Nonnegative numeric values or NaN required')
    if not np.isfinite(valid_threshold) or not 0<=valid_threshold<=1:
        raise ValueError('Validity threshold must be in [0,1]')
    starts,ends=find_start_end(values)
    keep=~((ends-starts)/frame.shape[1]<valid_threshold)
    selected=frame.loc[keep]
    return np.log1p(selected.fillna(0)),pd.isnull(selected),starts[keep],ends[keep]
