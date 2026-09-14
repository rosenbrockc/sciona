"""Historical candidate font-wrapper draws with explicit owned RNG streams.

Seeds are reconstruction inputs, not a claim to reproduce notebook worker
seeding. Backend selection must be part of the recorded execution contract.
"""
import random
import numpy as np

from sciona.bengali_font_augmentation import FontTransform
from sciona.bengali_shear_sampling import draw_shear


class FontParameterSampler:
    def __init__(self, *, python_seed, image_seed, backend):
        if any(type(seed) is not int or not 0<=seed<2**32 for seed in (python_seed,image_seed)):
            raise ValueError('explicit uint32 augmentation seeds required')
        if backend not in ('legacy','sfc64'):
            raise ValueError('explicit legacy or sfc64 backend required')
        self.python=random.Random(python_seed)
        self.image=(np.random.RandomState(image_seed) if backend=='legacy'
                    else np.random.Generator(np.random.SFC64(image_seed)))
        self.backend=backend

    def draw(self):
        return FontTransform(*self._draw_geometry(5))

    def _draw_geometry(self,limit):
        # Compose, CenterCrop, Resize, PadIfNeeded, IAAAffine each make a
        # probability draw even though all five always run for this pipeline.
        for _ in range(5):self.python.random()
        if self.backend=='legacy':
            child=np.random.RandomState(self.image.randint(0,2**31-1))
        else:
            seed=self.image.integers(0,2**31-1,dtype='int32',size=(2,))[-1]
            child=np.random.Generator(np.random.SFC64(np.random.SeedSequence(seed).spawn(1)[0]))
        shear=draw_shear(child,limit=limit)
        self.python.random()  # ShiftScaleRotate decision.
        angle=self.python.uniform(-limit,limit)
        scale=self.python.uniform(.9,1.1)
        dx=self.python.uniform(-.0625,.0625)
        dy=self.python.uniform(-.0625,.0625)
        self.python.random()  # RandomCrop decision.
        return shear,angle,scale,dx,dy,self.python.random(),self.python.random()
