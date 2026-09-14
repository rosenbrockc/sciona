"""DFDC per-access sample preparation and CPU DataLoader integration.

Source MIT2020SelimSeferbekov,89c6290490bac96b29193a4061b3db9dd3933e36;
see docs/licenses/DFDC-MIT.txt. Default workers0 is an explicit local execution
adaptation from source workers6; source shuffle/drop-last/batch ratios retained.
"""
import numpy as np
from torch.utils.data import Dataset,DataLoader

from sciona.dfdc_population import select_population
from sciona.dfdc_preparation import prepare_sample
from sciona.dfdc_transforms import create_train_transforms,create_val_transforms


class PreparedDataset(Dataset):
    def __init__(self, records, *, mode, detector, predictor, fold=0):
        if mode not in ('train','val'):
            raise ValueError('mode must be train or val')
        self.records=tuple(records)
        self.mode=mode;self.fold=fold
        self.detector=detector;self.predictor=predictor
        self.transforms=(create_train_transforms if mode=='train' else create_val_transforms)(380)
        self.indices=None

    def reset(self, epoch, seed):
        """Reset source population and current NumPy stream for augmentation.

        The outer lifecycle must isolate/restore process RNG state around a run.
        """
        selection=select_population([r['label'] for r in self.records],
            [r['fold'] for r in self.records],[r['frame'] for r in self.records],
            mode=self.mode,fold=self.fold,epoch=epoch,seed=seed)
        self.indices=selection.indices
        np.random.set_state(selection.numpy_state_after_shuffle)

    def __len__(self):
        if self.indices is None:
            raise RuntimeError('dataset must be reset before use')
        return len(self.indices)

    def __getitem__(self,index):
        if self.indices is None:
            raise RuntimeError('dataset must be reset before use')
        row=self.records[int(self.indices[index])]
        return prepare_sample(row['image'],row['label'],mode=self.mode,
            transforms=self.transforms,detector=self.detector,predictor=self.predictor,
            mask=row['mask'],landmarks=row['landmarks'])


def make_loader(dataset, *, batch_size=12, num_workers=0, generator=None):
    """Source train batch12/shuffle/drop-last; validation batch24/ordered/full."""
    if type(batch_size) is not int or batch_size<=0:
        raise ValueError('batch_size must be a positive integer')
    if type(num_workers) is not int or num_workers<0:
        raise ValueError('num_workers must be a nonnegative integer')
    training=dataset.mode=='train'
    size=batch_size if training else batch_size*2
    if len(dataset)==0 or (training and len(dataset)<size):
        raise ValueError('population cannot produce a usable loader batch')
    return DataLoader(dataset,batch_size=size,num_workers=num_workers,
        shuffle=training,drop_last=training,pin_memory=False,generator=generator)
