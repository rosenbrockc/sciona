"""In-memory training preparation for both final Contrails branches.

Adapted from Jun Koda, commit 08a15beb36f9cbed4c3990e74c625c1332b61fe8.
MIT: docs/licenses/Contrails-MIT.txt. File loading is replaced by explicit arrays.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize

from sciona.contrails_grid import create_grid
from sciona.contrails_preprocessing import ash_color
from sciona.contrails_temporal import create_four_panels


def augmentation(mode='rotation'):
    import albumentations as A
    transforms = [A.RandomRotate90(p=1), A.HorizontalFlip(p=.5)]
    if mode == 'rotation':
        transforms.append(A.ShiftScaleRotate(rotate_limit=30, scale_limit=.2, p=.75))
    elif mode != 'd4':
        raise ValueError('Unknown source augmentation')
    return A.Compose(transforms)


def prepare_training_example(thermal, label, annotation_mean, *, branch,
                             annotation_mode=True, augment=None,
                             augment_probability=.95, random_draw=None):
    """Prepare source-sized arrays; augmentation jointly transforms image/shifted label.

    Original targets remain untransformed and receive weight zero on augmented
    examples. All input arrays are explicit runtime inputs; no files are accessed.
    """
    if branch not in ('single', 'temporal'):
        raise ValueError('Unknown branch')
    thermal = np.asarray(thermal)
    label = np.asarray(label, dtype=np.float32)
    mean = np.asarray(annotation_mean, dtype=np.float32)
    if thermal.shape != (4, 3, 256, 256) or label.shape != (1, 256, 256) or mean.shape != label.shape:
        raise ValueError('Unexpected source-sized input shapes')
    if not np.issubdtype(thermal.dtype, np.floating) or not all(np.isfinite(a).all() for a in (thermal, label, mean)):
        raise ValueError('Expected finite floating point inputs')
    if not (0 <= augment_probability <= 1) or any(np.any((a < 0) | (a > 1)) for a in (label, mean)):
        raise ValueError('Invalid augmentation probability or label range')
    if annotation_mode == 'mix':
        target = .5 * (mean + label)
    elif annotation_mode is True:
        target = mean
    elif annotation_mode is False:
        target = label
    else:
        raise ValueError('Unknown annotation mode')
    y = torch.from_numpy(target.copy())
    x = torch.from_numpy(thermal.copy())
    x = ash_color(x[3] if branch == 'single' else x)
    x = Resize(1024 if branch == 'single' else 512, antialias=False)(x)
    y_sym = F.grid_sample(y.unsqueeze(0), create_grid(512, offset=.5),
                          mode='bilinear', padding_mode='border', align_corners=False).squeeze(0)
    weight = np.float32(1)
    if augment is not None:
        draw = np.random.random() if random_draw is None else random_draw
        if not np.isfinite(draw) or not 0 <= draw < 1:
            raise ValueError('Expected random draw in [0,1)')
        if draw < augment_probability:
            weight = np.float32(0)
            if branch == 'temporal':
                x = x.reshape(12, 512, 512)
            result = augment(image=x.permute(1, 2, 0).numpy(), mask=y_sym.permute(1, 2, 0).numpy())
            x = torch.from_numpy(result['image'].transpose(2, 0, 1))
            y_sym = torch.from_numpy(result['mask'].transpose(2, 0, 1))
            if branch == 'temporal':
                x = x.reshape(4, 3, 512, 512)
    if branch == 'temporal':
        x = create_four_panels(x.unsqueeze(0)).squeeze(0)
    return dict(x=x, y=y, y_sym=y_sym, w=torch.tensor(weight), label=torch.from_numpy(label.copy()))
