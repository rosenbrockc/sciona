"""DeepVoltaire SVHN policy with an explicitly owned Python random stream.

Derived from DeepVoltaire/AutoAugment revision
17d718251f25c0d9413bf30f91b523907924f33a (MIT, Philip Popien, 2018).
See docs/licenses/deepvoltaire-autoaugment-LICENSE.txt.
Pillow raster behavior is that of the installed version, not a claim about the
winner's unspecified historical environment. No resize/crop is added here.
"""
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


POLICIES = (
    (.9, 'shearX', 4, .2, 'invert', 3),
    (.9, 'shearY', 8, .7, 'invert', 5),
    (.6, 'equalize', 5, .6, 'solarize', 6),
    (.9, 'invert', 3, .6, 'equalize', 3),
    (.6, 'equalize', 1, .9, 'rotate', 3),
    (.9, 'shearX', 4, .8, 'autocontrast', 3),
    (.9, 'shearY', 8, .4, 'invert', 5),
    (.9, 'shearY', 5, .2, 'solarize', 6),
    (.9, 'invert', 6, .8, 'autocontrast', 1),
    (.6, 'equalize', 3, .9, 'rotate', 3),
    (.9, 'shearX', 4, .3, 'solarize', 3),
    (.8, 'shearY', 8, .7, 'invert', 4),
    (.9, 'equalize', 5, .6, 'translateY', 6),
    (.9, 'invert', 4, .6, 'equalize', 7),
    (.3, 'contrast', 3, .8, 'rotate', 4),
    (.8, 'invert', 5, .0, 'translateY', 2),
    (.7, 'shearY', 6, .4, 'solarize', 8),
    (.6, 'invert', 4, .8, 'rotate', 4),
    (.3, 'shearY', 7, .9, 'translateX', 3),
    (.1, 'shearX', 6, .6, 'invert', 5),
    (.7, 'solarize', 2, .6, 'translateY', 7),
    (.8, 'shearY', 4, .8, 'invert', 8),
    (.7, 'shearX', 9, .8, 'translateY', 3),
    (.8, 'shearY', 5, .7, 'autocontrast', 3),
    (.7, 'shearX', 2, .1, 'invert', 5),
)


class SVHNAutoAugment:
    def __init__(self, *, seed, fillcolor=(128, 128, 128)):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('explicit uint32 augmentation seed required')
        if (not isinstance(fillcolor, tuple) or len(fillcolor) != 3
                or any(type(v) is not int or not 0 <= v <= 255 for v in fillcolor)):
            raise ValueError('RGB byte fill tuple required')
        self.rng = random.Random(seed)
        self.fillcolor = fillcolor
        self.ranges = {
            'shearX': np.linspace(0, .3, 10), 'shearY': np.linspace(0, .3, 10),
            'translateX': np.linspace(0, 150 / 331, 10), 'translateY': np.linspace(0, 150 / 331, 10),
            'rotate': np.linspace(0, 30, 10), 'solarize': np.linspace(256, 0, 10),
            'contrast': np.linspace(0, .9, 10),
            'invert': [0] * 10, 'equalize': [0] * 10, 'autocontrast': [0] * 10,
        }

    def _operation(self, image, operation, magnitude):
        if operation in ('shearX', 'shearY'):
            signed = magnitude * self.rng.choice([-1, 1])
            matrix = (1, signed, 0, 0, 1, 0) if operation == 'shearX' else (1, 0, 0, signed, 1, 0)
            return image.transform(image.size, Image.Transform.AFFINE, matrix,
                                   Image.Resampling.BICUBIC, fillcolor=self.fillcolor)
        if operation in ('translateX', 'translateY'):
            axis = 0 if operation == 'translateX' else 1
            offset = magnitude * image.size[axis] * self.rng.choice([-1, 1])
            matrix = (1, 0, offset, 0, 1, 0) if axis == 0 else (1, 0, 0, 0, 1, offset)
            return image.transform(image.size, Image.Transform.AFFINE, matrix, fillcolor=self.fillcolor)
        if operation == 'rotate':
            # Source rotation has no sign draw and ignores custom fillcolor.
            rotated = image.convert('RGBA').rotate(magnitude)
            return Image.composite(rotated, Image.new('RGBA', rotated.size, (128,) * 4), rotated).convert(image.mode)
        if operation == 'contrast':
            return ImageEnhance.Contrast(image).enhance(1 + magnitude * self.rng.choice([-1, 1]))
        if operation == 'solarize':
            return ImageOps.solarize(image, magnitude)
        return {'equalize': ImageOps.equalize, 'invert': ImageOps.invert,
                'autocontrast': ImageOps.autocontrast}[operation](image)

    def _subpolicy(self, image, policy):
        p1, op1, index1, p2, op2, index2 = policy
        if self.rng.random() < p1:
            image = self._operation(image, op1, self.ranges[op1][index1])
        if self.rng.random() < p2:
            image = self._operation(image, op2, self.ranges[op2][index2])
        return image

    def __call__(self, image):
        if not isinstance(image, Image.Image) or image.mode != 'RGB' or min(image.size) < 1:
            raise ValueError('nonempty Pillow RGB image required')
        index = self.rng.randint(0, len(POLICIES) - 1)
        return self._subpolicy(image, POLICIES[index])
