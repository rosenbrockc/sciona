"""Explicit APTOS training augmentation reference, not recovered winner code.

Published ranges are retained. Underspecified semantics are resolved as:
independent uniform draws; additive brightness/saturation in byte units; hue
in degrees; contrast multiplier; shear slope converted to degrees; independent
horizontal/vertical mirrors. Blur/sharpen is an equal-probability choice of
Pillow GaussianBlur(radius=1) or SHARPEN. Operation order is photometric,
filter, affine, mirrors. Affine uses bilinear interpolation and black fill.
These choices require conditional Tier 3 review alongside the full workflow.
"""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class Parameters:
    contrast: float = 1.
    brightness: float = 0.
    hue_degrees: float = 0.
    saturation: float = 0.
    angle: float = 0.
    scale: float = 1.
    shear_slope: float = 0.
    shift_x: float = 0.
    shift_y: float = 0.
    mirror_x: bool = False
    mirror_y: bool = False
    filter: str = 'none'


def draw_parameters(rng):
    """Draw from a caller-owned generator; never consume a global RNG."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError('Explicit numpy Generator required')
    return Parameters(contrast=float(rng.uniform(.8, 1.2)),
        brightness=float(rng.uniform(-20., 20.)), hue_degrees=float(rng.uniform(-10., 10.)),
        saturation=float(rng.uniform(-20., 20.)), angle=float(rng.uniform(-180., 180.)),
        scale=float(rng.uniform(.8, 1.2)), shear_slope=float(rng.uniform(-.2, .2)),
        shift_x=float(rng.uniform(-.2, .2)), shift_y=float(rng.uniform(-.2, .2)),
        mirror_x=bool(rng.integers(2)), mirror_y=bool(rng.integers(2)),
        filter='blur' if rng.integers(2) else 'sharpen')


def apply_parameters(rgb, parameters):
    """Apply replayable reference parameters to RGB uint8 before normalization."""
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1):
        raise ValueError('Nonempty RGB uint8 HWC image required')
    if not isinstance(parameters, Parameters):
        raise ValueError('Explicit augmentation Parameters required')
    p = parameters
    bounds = [(p.contrast, .8, 1.2), (p.brightness, -20., 20.),
        (p.hue_degrees, -10., 10.), (p.saturation, -20., 20.), (p.angle, -180., 180.),
        (p.scale, .8, 1.2), (p.shear_slope, -.2, .2), (p.shift_x, -.2, .2), (p.shift_y, -.2, .2)]
    if (any(not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or not lo <= value <= hi for value, lo, hi in bounds)
            or type(p.mirror_x) is not bool or type(p.mirror_y) is not bool
            or p.filter not in ('none', 'blur', 'sharpen')):
        raise ValueError('Augmentation parameters outside reference ranges')
    image = Image.fromarray(rgb)
    if p.contrast != 1.:
        image = ImageEnhance.Contrast(image).enhance(p.contrast)
    if p.brightness:
        values = np.asarray(image, dtype=np.float64) + p.brightness
        image = Image.fromarray(np.rint(values).clip(0, 255).astype(np.uint8))
    if p.hue_degrees or p.saturation:
        hsv = np.asarray(image.convert('HSV')).copy()
        hsv[..., 0] = np.rint(hsv[..., 0].astype(np.float64) + p.hue_degrees * 256. / 360.).astype(np.int64) % 256
        hsv[..., 1] = np.rint(hsv[..., 1].astype(np.float64) + p.saturation).clip(0, 255).astype(np.uint8)
        image = Image.frombytes('HSV', image.size, hsv.tobytes()).convert('RGB')
    if p.filter != 'none':
        image = image.filter(ImageFilter.GaussianBlur(radius=1.) if p.filter == 'blur' else ImageFilter.SHARPEN)
    image = TF.affine(image, angle=p.angle,
        translate=[round(p.shift_x * image.width), round(p.shift_y * image.height)],
        scale=p.scale, shear=[math.degrees(math.atan(p.shear_slope)), 0.],
        interpolation=InterpolationMode.BILINEAR, fill=0)
    if p.mirror_x:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if p.mirror_y:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return np.array(image, dtype=np.uint8, copy=True)
