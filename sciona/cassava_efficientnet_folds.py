"""Shared fivefold EfficientNet CV with explicit best-checkpoint evaluation."""
import gc
from pathlib import Path

from sciona.cassava_fold_contract import members
from sciona.cassava_fold_dispatch import train_planned_fold


def train_five_folds(plan, encoded_images, *, weights, expected_sha256, batch_size, output_directory):
    """Return CV histories, selected-checkpoint scores and private checkpoints.

CV models are evaluated on their held-out populations. The final submission
uses a separate full-population refit, not an average of these CV models.
"""
    from tensorflow import keras
    import tensorflow as tf
    from sciona.cassava_efficientnet_training import keras_focal_loss
    from sciona.cassava_training_stream import validation_stream, epoch_steps
    members(plan, 0)
    if (not isinstance(encoded_images, (list, tuple)) or len(encoded_images) != len(plan.keys)
            or not all(isinstance(image, bytes) and image for image in encoded_images)):
        raise ValueError('Encoded population must align with the validated plan')
    for fold in range(5):
        training, _ = members(plan, fold)
        epoch_steps(len(training), batch_size)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    histories, evaluations, checkpoints = {}, {}, {}
    for fold in range(5):
        keras.backend.clear_session()
        model, history, checkpoint = train_planned_fold('efficientnet', plan, encoded_images, fold,
            weights=weights, expected_sha256=expected_sha256, weight_format='upstream',
            batch_size=batch_size, checkpoint_directory=directory / f'fold-{fold}')
        del model
        gc.collect()
        # Always load the selected artifact, including runs reaching epoch 20.
        selected = keras.models.load_model(checkpoint, custom_objects={'keras_focal_loss': keras_focal_loss})
        _, valid = members(plan, fold)
        records = tf.data.Dataset.from_tensor_slices(
            ([encoded_images[i] for i in valid], [plan.labels[i] for i in valid]))
        score = selected.evaluate(validation_stream(records, batch_size=batch_size), verbose=0, return_dict=True)
        histories[fold], evaluations[fold], checkpoints[fold] = history, score, checkpoint
        del selected
        keras.backend.clear_session()
        gc.collect()
    return histories, evaluations, checkpoints
