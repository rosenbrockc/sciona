"""Web Traffic count-space checkpoint/model averaging (MIT source adaptation).

Derived from trainer.predict and final notebook; see docs/licenses/WebTraffic-MIT.txt.
Inputs are runtime predictions, not filesystem checkpoints or competition records.
"""
import numpy as np
import pandas as pd


def average_checkpoint_predictions(log_predictions):
    """Average after expm1, matching trainer.predict accumulation order."""
    predictions = None
    count = 0
    for frame in log_predictions:
        current = np.expm1(frame)
        if predictions is None:
            predictions = current
        else:
            predictions += current
        count += 1
    if count == 0:
        raise ValueError('At least one checkpoint prediction required')
    predictions /= count
    return predictions


def finalize_predictions(model_predictions, runtime_pages):
    """Three-model notebook mean, missing-page fill, threshold and ties-to-even round."""
    if len(model_predictions) != 3:
        raise ValueError('Source final ensemble requires three models')
    preds = sum(model_predictions) / 3
    missing_pages = runtime_pages.difference(preds.index)
    missing = pd.DataFrame(index=missing_pages,
                           data=np.tile(0,(len(preds.columns),len(missing_pages))).T,
                           columns=preds.columns)
    result = pd.concat([preds, missing]).sort_index()
    result[result < 0.5] = 0
    return np.round(result).astype(np.int64)
