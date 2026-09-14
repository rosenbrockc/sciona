"""NumPy window numerics adapted from Web Traffic InputPipe (MIT).

See docs/licenses/WebTraffic-MIT.txt. TensorFlow dataset scheduling is not included.
"""
import numpy as np


def cut_window(hits, dow, lagged_ix, start, train_window, predict_window):
    """Preserve source padding, missing-lag masking and target NaNs."""
    end = start + train_window + predict_window
    padded = np.concatenate([hits, np.full(predict_window, np.nan, dtype=hits.dtype)])
    cropped = padded[start:end]
    lags = lagged_ix[start:end].astype(np.int32)
    lag_mask = lags < 0
    lagged = padded[np.maximum(lags, 0)]
    lagged = np.where(lag_mask | np.isnan(lagged), np.zeros_like(lagged), lagged)
    x, y = np.split(cropped, [train_window])
    x = np.where(np.isnan(x), np.zeros_like(x), x)
    return x, y, dow[start:end], lagged


def make_window_features(x_hits, y_hits, dow, lagged_hits, pf_agent, pf_country,
                         pf_site, page_ix, page_popularity, year_autocorr,
                         quarter_autocorr):
    """Source feature order and population normalization, including zero-std NaNs."""
    train_window, predict_window = len(x_hits), len(y_hits)
    x_dow, y_dow = np.split(dow, [train_window])
    mean = np.mean(x_hits)
    std = np.sqrt(np.mean(np.square(x_hits - mean)))
    norm_x = (x_hits - mean) / std
    norm_y = (y_hits - mean) / std
    norm_lags = (lagged_hits - mean) / std
    x_lags, y_lags = np.split(norm_lags, [train_window])
    scalars = np.stack([page_popularity, quarter_autocorr, year_autocorr])
    flat = np.concatenate([pf_agent, pf_country, pf_site, scalars])
    page = np.expand_dims(flat, 0)
    x_features = np.concatenate([norm_x[:,None], x_dow, x_lags,
                                 np.tile(page, [train_window,1])], axis=1)
    y_features = np.concatenate([y_dow, y_lags, np.tile(page,[predict_window,1])], axis=1)
    return (x_hits, x_features, norm_x, x_lags, y_hits, y_features,
            norm_y, mean, std, flat, page_ix)


def keep_window(x_hits, max_train_empty):
    """Source rejection considers zero training values only."""
    return np.sum(x_hits == 0) <= max_train_empty


def training_offset_bounds(data_days, train_window, predict_window, back_offset, start_offset):
    """Source integer random_uniform bounds: upper endpoint is excluded."""
    return start_offset, data_days - predict_window - train_window - back_offset - start_offset
