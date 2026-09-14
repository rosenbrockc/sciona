"""Source replacement-sampling budget with explicit, separate domain RNGs.

The notebook creates loaders once, outside the epoch loop. Epoch boundaries do
not reset sampling. Only full batches consumed by that loop are yielded here.
"""
from dataclasses import dataclass
import itertools

import torch


@dataclass(frozen=True)
class SamplingBudget:
    hand_count: int
    font_count: int
    batch_size: int
    epochs: int

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.hand_count,self.font_count,self.batch_size,self.epochs)):
            raise ValueError('positive integer population counts, batch size and epochs required')
        if self.steps_per_epoch < 1:
            raise ValueError('source population cannot supply a full training batch')

    @property
    def requested_samples_per_domain(self):
        return max(self.hand_count,self.font_count)*self.epochs

    @property
    def steps_per_epoch(self):
        return (self.requested_samples_per_domain//self.batch_size)//self.epochs

    @property
    def total_steps(self):
        return self.steps_per_epoch*self.epochs

    @property
    def unused_samples_per_domain(self):
        return self.requested_samples_per_domain-self.total_steps*self.batch_size


def paired_batches(budget, *, hand_seed, font_seed):
    if not isinstance(budget,SamplingBudget):
        raise ValueError('validated sampling budget required')
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in (hand_seed,font_seed)):
        raise ValueError('explicit uint32 domain seeds required')
    samplers=[iter(torch.utils.data.RandomSampler(range(count),replacement=True,
        num_samples=budget.requested_samples_per_domain,
        generator=torch.Generator().manual_seed(seed)))
        for count,seed in [(budget.hand_count,hand_seed),(budget.font_count,font_seed)]]
    for step in range(budget.total_steps):
        batches=[torch.tensor(list(itertools.islice(sampler,budget.batch_size)),dtype=torch.int64) for sampler in samplers]
        if any(len(batch)!=budget.batch_size for batch in batches):
            raise RuntimeError('replacement sampler ended before declared budget')
        yield dict(epoch=step//budget.steps_per_epoch,step=step,
                   hand_indices=batches[0],font_indices=batches[1])
