"""Independent 50-image replay pool for the Bengali CycleGAN discriminators."""
import random
import torch


class ImageReplayPool:
    def __init__(self, *, seed):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('explicit uint32 replay seed required')
        self.rng = random.Random(seed)
        self.images = []
        self.shape = None

    def query(self, images):
        if (not isinstance(images, torch.Tensor) or images.device.type != 'cpu' or images.dtype != torch.float32
                or images.ndim != 4 or any(n < 1 for n in images.shape) or not torch.isfinite(images).all()
                or self.shape is not None and tuple(images.shape[1:]) != self.shape):
            raise ValueError('aligned finite float32 CPU NCHW replay images required')
        self.shape = tuple(images.shape[1:])
        result = []
        for image in images:
            current = image.detach().clone()
            if len(self.images) < 50:
                self.images.append(current)
                result.append(current)
            elif self.rng.uniform(0, 1) > .5:
                index = self.rng.randint(0, 49)
                result.append(self.images[index])
                self.images[index] = current
            else:
                result.append(current)
        return torch.stack(result)

    def snapshot(self):
        return dict(images=[image.clone() for image in self.images], shape=self.shape, random_state=self.rng.getstate())

    def restore(self, state):
        images, shape = state['images'], state['shape']
        if not isinstance(images, list) or len(images) > 50:
            raise ValueError('invalid replay history')
        if images:
            if not isinstance(shape, tuple) or len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError('invalid replay shape')
            for image in images:
                if (not isinstance(image, torch.Tensor) or image.device.type != 'cpu' or image.dtype != torch.float32
                        or tuple(image.shape) != shape or not torch.isfinite(image).all()):
                    raise ValueError('invalid replay image')
        elif shape is not None:
            raise ValueError('empty replay history requires no shape')
        rng = random.Random()
        rng.setstate(state['random_state'])
        self.images, self.shape, self.rng = [image.detach().clone() for image in images], shape, rng
