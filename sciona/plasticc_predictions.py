"""Source task-specific probability adjustment, not general novelty detection.

Copyright (c) 2019 Kyle Boone. MIT: docs/licenses/Avocado-MIT.txt.
Source kboone/avocado cd7809db4d7eb92860b178d8f5a53ab16f03ad91.
No file export or persistence. Input class probabilities are runtime supplied.
"""
import numpy as np

def create_kaggle_predictions(dataset, predictions=None):
    """Add predictions for unknown objects for the Kaggle PLAsTiCC challenge
    using a predefined formula.

    This formula was tuned to the PLAsTiCC dataset, and is not a real method of
    identifying new objects in a dataset.

    Parameters
    ----------
    dataset : :class:`Dataset`
        The dataset to create predictions for.
    predictions : :class:`pandas.DataFrame` (optional)
        The original predictions for each class. If not specified, the
        dataset's predictions will be used.

    Returns
    -------
    kaggle_predictions : :class:`pandas.DataFrame`
        A pandas DataFrame with the predictions for each class, with class 99
        predictions added.
    """
    if predictions is None:
        predictions = dataset.predictions
    predictions = predictions.copy()
    if 99 in predictions:
        predictions[99] = 0.0
        norm = np.sum(predictions, axis=1)
        predictions = predictions.div(norm, axis=0)
    galactic_classes = [6, 16, 53, 65, 92]
    for class_name in predictions.columns:
        if class_name in galactic_classes:
            mask = ~dataset.metadata['galactic']
        else:
            mask = dataset.metadata['galactic']
        predictions.loc[mask, class_name] = 0.0
    predictions[99] = 1.0 * predictions[42] + 0.6 * predictions[62] + 0.2 * predictions[52] + 0.2 * predictions[95]
    predictions.loc[dataset.metadata['galactic'], 99] = 0.04
    norm = np.sum(predictions, axis=1)
    predictions = predictions.div(norm, axis=0)
    return predictions
