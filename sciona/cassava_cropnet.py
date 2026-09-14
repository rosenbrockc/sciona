"""Hash-verified official frozen CropNet branch in the reference TF runtime."""
import hashlib
from pathlib import Path

import numpy as np

from sciona.cassava_ensemble import distribute_unknown
from sciona.cassava_image_views import cropnet_view


def load_model(directory, *, expected_hashes):
    """Load only an exact reviewed model tree; manifest must be caller-trusted."""
    import tensorflow as tf
    root = Path(directory)
    if not isinstance(expected_hashes, dict) or not expected_hashes or root.is_symlink():
        raise ValueError('Explicit model hash manifest and nonsymlink directory required')
    files = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Model tree must not contain symlinks')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if files != expected_hashes:
        raise ValueError('Model files differ from reviewed manifest')
    return tf.saved_model.load(str(root))


def predict_images(model, images):
    """Apply source crop, frozen classifier and uniform unknown-mass mapping."""
    if not isinstance(images, (list, tuple)) or not images:
        raise ValueError('Nonempty decoded-image sequence required')
    output = []
    for image in images:
        raw = model(cropnet_view(image), training=False)
        output.append(distribute_unknown(np.asarray(raw))[0])
    return np.stack(output)
