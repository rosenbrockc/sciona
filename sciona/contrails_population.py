"""Source fold selection and in-memory training loaders for Contrails.

Jun Koda's MIT-licensed source uses shuffled ten-fold KFold, random_state=42,
then selects branch-specific folds. Inputs retain caller-provided order.
"""
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from sciona.contrails_preparation import augmentation, prepare_training_example


FOLDS = {'v43': {'single': (3, 4), 'temporal': (5, 7)},
         'v47': {'single': (0, 2, 3, 4, 5), 'temporal': (5, 6, 7, 8, 9)}}


def select_folds(population_size, *, branch, variant='v47'):
    if type(population_size) is not int or population_size < 10:
        raise ValueError('Source ten-fold split requires at least ten examples')
    if variant not in FOLDS or branch not in FOLDS[variant]:
        raise ValueError('Unknown final branch or variant')
    selected = FOLDS[variant][branch]
    return [(fold, train, validation) for fold, (train, validation) in
            enumerate(KFold(n_splits=10, shuffle=True, random_state=42).split(np.arange(population_size)))
            if fold in selected]


class PreparedDataset(Dataset):
    """Prepare each example on access so augmentation is resampled each epoch."""
    def __init__(self, examples, indices, *, branch, training):
        self.examples = examples
        self.indices = tuple(int(i) for i in indices)
        if not self.indices or any(i < 0 or i >= len(examples) for i in self.indices):
            raise ValueError('Expected nonempty valid population indices')
        if branch not in ('single', 'temporal'):
            raise ValueError('Unknown branch')
        self.branch = branch
        self.augment = augmentation() if training else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        example = self.examples[self.indices[index]]
        return prepare_training_example(example['thermal'], example['label'], example['annotation_mean'],
                                        branch=self.branch, augment=self.augment)


def make_loader(examples, indices, *, branch, training, batch_size=4, num_workers=2, generator=None):
    """Source shuffle/drop-last for training; ordered complete validation batches."""
    if type(batch_size) is not int or batch_size < 1 or type(num_workers) is not int or num_workers < 0:
        raise ValueError('Invalid loader sizing')
    dataset = PreparedDataset(examples, indices, branch=branch, training=training)
    if training and len(dataset) < batch_size:
        raise ValueError('Training population cannot form one complete batch')
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=training, drop_last=training, generator=generator)
