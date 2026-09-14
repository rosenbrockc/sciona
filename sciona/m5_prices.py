"""Independent price features before source memory downcasting and grid joins.

All rows participate in group moments, including known future prices. The caller
supplies ordered price rows and calendar roles already joined to each row.
"""
import numpy as np


def features(prices, outlet, product, month, year):
    prices = np.asarray(prices)
    roles = [np.asarray(v) for v in (outlet, product, month, year)]
    if (prices.ndim != 1 or not prices.size or prices.dtype.kind not in 'iuf'
            or any(r.shape != prices.shape or r.dtype.kind not in 'iu' for r in roles)):
        raise ValueError('Expected real prices and aligned integer role codes')
    prices = prices.astype(np.float64)
    if np.isinf(prices).any() or (prices < 0).any():
        raise ValueError('Prices must be nonnegative finite values or NaN')
    outlet, product, month, year = roles
    names = ('maximum', 'minimum', 'std', 'mean', 'normalized', 'distinct_prices',
             'distinct_products', 'previous_ratio', 'month_ratio', 'year_ratio')
    out = {name: np.full(prices.shape, np.nan) for name in names}
    _, membership = np.unique(np.column_stack((outlet, product)), axis=0, return_inverse=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        for group in np.unique(membership):
            positions = np.flatnonzero(membership == group)
            values = prices[positions]
            valid = values[~np.isnan(values)]
            out['distinct_prices'][positions] = len(np.unique(valid))
            if valid.size:
                out['maximum'][positions] = np.max(valid)
                out['minimum'][positions] = np.min(valid)
                out['mean'][positions] = np.mean(valid)
                if valid.size > 1:
                    out['std'][positions] = np.std(valid, ddof=1)
                out['normalized'][positions] = values / np.max(valid)
            out['previous_ratio'][positions[1:]] = values[1:] / values[:-1]
            for role, name in ((month, 'month_ratio'), (year, 'year_ratio')):
                for period in np.unique(role[positions]):
                    subset = positions[role[positions] == period]
                    history = prices[subset]
                    valid_period = history[~np.isnan(history)]
                    if valid_period.size:
                        out[name][subset] = history / np.mean(valid_period)
        # Count unique products sharing the same outlet and exact price, across
        # all periods. Missing-price groups are excluded, as in source groupby.
        for group in np.unique(outlet):
            for price in np.unique(prices[outlet == group]):
                if np.isnan(price):
                    continue
                mask = (outlet == group) & (prices == price)
                out['distinct_products'][mask] = len(np.unique(product[mask]))
    return out
