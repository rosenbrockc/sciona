"""Corrected Bengali CycleGAN update ordering with explicitly supplied networks.

Network architecture, initialization, population construction and winning fit
budgets must be qualified by the caller. This is one training-step lifecycle,
not the full competition pipeline.
"""
import itertools

import torch
from torch import nn

from sciona.bengali_font_guidance import FrozenFontGuidance
from sciona.bengali_gan_losses import generator_objective, discriminator_objective
from sciona.bengali_image_pool import ImageReplayPool


class CycleGANTraining:
    def __init__(self, *, generator_a, generator_b, discriminator_a, discriminator_b,
                 classifier, classifier_weight, total_steps, replay_seed_a, replay_seed_b):
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError('positive explicit total training steps required')
        modules = [generator_a, generator_b, discriminator_a, discriminator_b, classifier]
        if any(not isinstance(m, nn.Module) for m in modules):
            raise ValueError('five separately owned network modules required')
        parameters = [list(m.parameters()) for m in modules]
        flat = list(itertools.chain.from_iterable(parameters))
        if any(not p for p in parameters) or len({id(p) for p in flat}) != len(flat):
            raise ValueError('networks must have parameters and cannot share them')
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 or not torch.isfinite(p).all() for p in flat):
            raise ValueError('finite float32 CPU network parameters required')
        if any(not p.requires_grad for p in itertools.chain.from_iterable(parameters[:4])):
            raise ValueError('generator and discriminator parameters must be trainable')
        self.pool_a = ImageReplayPool(seed=replay_seed_a)
        self.pool_b = ImageReplayPool(seed=replay_seed_b)
        self.generator_a, self.generator_b = generator_a, generator_b
        self.discriminator_a, self.discriminator_b = discriminator_a, discriminator_b
        self.guidance = FrozenFontGuidance(classifier, weight=classifier_weight)
        self.generator_parameters = parameters[0] + parameters[1]
        self.discriminator_parameters = parameters[2] + parameters[3]
        self.generator_optimizer = torch.optim.Adam(self.generator_parameters, lr=.0002, betas=(.5,.999))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator_parameters, lr=.0002, betas=(.5,.999))
        self.total_steps, self.completed_steps = total_steps, 0
        self.generator_scheduler = torch.optim.lr_scheduler.LambdaLR(self.generator_optimizer, self._rate)
        self.discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(self.discriminator_optimizer, self._rate)

    def _rate(self, step):
        halfway = self.total_steps * .5
        return 1. if step < halfway else (self.total_steps - step) / halfway

    def step(self, real_a, real_b, labels_a):
        if self.completed_steps >= self.total_steps:
            raise ValueError('declared training budget exhausted')
        for images in (real_a, real_b):
            if (not isinstance(images, torch.Tensor) or images.dtype != torch.float32
                    or images.device.type != 'cpu' or images.ndim != 4
                    or images.shape[0] < 1 or images.shape[1:] != (3,224,224)
                    or not torch.isfinite(images).all()):
                raise ValueError('finite float32 CPU N3x224x224 images required')
        if (not isinstance(labels_a, torch.Tensor) or labels_a.dtype != torch.int64
                or labels_a.device.type != 'cpu' or labels_a.shape != (len(real_a),)
                or (labels_a < 0).any() or (labels_a >= 14784).any()):
            raise ValueError('aligned 14784-way integer labels required')
        for model in (self.generator_a, self.generator_b, self.discriminator_a, self.discriminator_b):
            model.train()
        self.guidance.train()  # Preserves frozen classifier eval despite parent train mode.
        rate = self.generator_optimizer.param_groups[0]['lr']
        fake_a, fake_b = self.generator_a(real_b), self.generator_b(real_a)
        rec_a, rec_b = self.generator_a(fake_b), self.generator_b(fake_a)
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        for parameter in self.discriminator_parameters:
            parameter.requires_grad_(False)
        try:
            generator = generator_objective(fake_a_prediction=self.discriminator_a(fake_a),
                fake_b_prediction=self.discriminator_b(fake_b), reconstructed_a=rec_a, real_a=real_a,
                reconstructed_b=rec_b, real_b=real_b, weighted_font_loss=self.guidance(fake_b, labels_a))
            generator['total'].backward()
            self.generator_optimizer.step()
        finally:
            for parameter in self.discriminator_parameters:
                parameter.requires_grad_(True)
        # Source reuses pre-generator-update fakes, and the replay pool detaches
        # them. Recomputing with the new generator would change the algorithm.
        discriminator = discriminator_objective(real_a_prediction=self.discriminator_a(real_a),
            replay_a_prediction=self.discriminator_a(self.pool_a.query(fake_a)),
            real_b_prediction=self.discriminator_b(real_b),
            replay_b_prediction=self.discriminator_b(self.pool_b.query(fake_b)))
        discriminator['total'].backward()
        self.discriminator_optimizer.step()
        self.generator_scheduler.step()
        self.discriminator_scheduler.step()
        self.completed_steps += 1
        return dict(step=self.completed_steps, learning_rate=rate,
                    generator={k:float(v.detach()) for k,v in generator.items()},
                    discriminator={k:float(v.detach()) for k,v in discriminator.items()})
