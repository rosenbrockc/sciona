"""Source Web Traffic feature assembly over explicit runtime inputs.

MIT adaptation of Arturus/kaggle-web-traffic make_features.run.
See docs/licenses/WebTraffic-MIT.txt. No filesystem dataset reader or writer.
"""
import numpy as np
import pandas as pd
from .webtraffic_features import prepare_series, batch_autocorr
from .webtraffic_calendar import lag_indexes
from .webtraffic_pages import uniq_page_map, make_page_features, encode_page_features


def normalize(values):
    return (values - values.mean()) / np.std(values)


def assemble_features(frame, valid_threshold=0.0, add_days=64, corr_backoffset=0):
    """Assemble the source tensors; caller supplies sorted, daily runtime series."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Nonempty runtime series table required")
    if not isinstance(frame.columns, pd.DatetimeIndex) or frame.columns.tz is not None:
        raise ValueError("Timezone-naive daily date columns required")
    if not frame.columns.equals(pd.date_range(frame.columns[0], periods=len(frame.columns))):
        raise ValueError("Contiguous increasing daily columns required")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Unique sorted page strings required")
    for value in (add_days, corr_backoffset):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("Nonnegative integer day offsets required")
    if len(frame.columns) + add_days > 32768 or corr_backoffset >= len(frame.columns):
        raise ValueError("Offsets exceed supported source index range")
    df, nans, starts, ends = prepare_series(frame, valid_threshold)
    if df.empty:
        raise ValueError("No series survive validity filtering")
    data_start, data_end = (df.columns[0], df.columns[-1])
    features_end = data_end + pd.Timedelta(add_days, unit='D')
    assert df.index.is_monotonic_increasing
    page_map = uniq_page_map(df.index.values)
    raw_year_autocorr = batch_autocorr(df.values, 365, starts, ends, 1.5, corr_backoffset)
    year_unknown_pct = np.sum(np.isnan(raw_year_autocorr)) / len(raw_year_autocorr)
    raw_quarter_autocorr = batch_autocorr(df.values, int(round(365.25 / 4)), starts, ends, 2, corr_backoffset)
    quarter_unknown_pct = np.sum(np.isnan(raw_quarter_autocorr)) / len(raw_quarter_autocorr)
    year_autocorr = normalize(np.nan_to_num(raw_year_autocorr))
    quarter_autocorr = normalize(np.nan_to_num(raw_quarter_autocorr))
    page_features = make_page_features(df.index.values)
    encoded_page_features = encode_page_features(page_features)
    features_days = pd.date_range(data_start, features_end)
    week_period = 7 / (2 * np.pi)
    dow_norm = features_days.dayofweek.values / week_period
    dow = np.stack([np.cos(dow_norm), np.sin(dow_norm)], axis=-1)
    lagged_ix = np.stack(lag_indexes(data_start, features_end), axis=-1)
    page_popularity = df.median(axis=1)
    page_popularity = (page_popularity - page_popularity.mean()) / page_popularity.std()
    df[nans] = np.nan
    tensors = dict(hits=df, lagged_ix=lagged_ix, page_map=page_map, page_ix=df.index.values, pf_agent=encoded_page_features['agent'], pf_country=encoded_page_features['country'], pf_site=encoded_page_features['site'], page_popularity=page_popularity, year_autocorr=year_autocorr, quarter_autocorr=quarter_autocorr, dow=dow)
    plain = dict(features_days=len(features_days), data_days=len(df.columns), n_pages=len(df), data_start=data_start, data_end=data_end, features_end=features_end)
    return tensors, plain
