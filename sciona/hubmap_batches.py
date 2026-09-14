"""Connect source population balancing, Torch shuffle and runtime augmentation."""
import torch
from sciona.hubmap_sampling import balanced_epoch_indices
from sciona.hubmap_loader import ordered_batches


def training_batches(images_bgr,rles,present,bins,*,maximum_bin,batch_size,input_side,
                     python_rng,numpy_rng,torch_generator):
    if not isinstance(torch_generator,torch.Generator) or torch_generator.device.type!='cpu':
        raise ValueError('Explicit CPU Torch generator required')
    if len(images_bgr)!=len(rles) or len(images_bgr)!=len(present) or len(images_bgr)!=len(bins):
        raise ValueError('Aligned runtime collections required')
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<2:
        raise ValueError('Training batch size at least two required')
    indices=balanced_epoch_indices(present,bins,maximum_bin=maximum_bin,rng=numpy_rng)
    # Use real DataLoader sampling to preserve both its base-seed draw and
    # RandomSampler's complete generator consumption, including the dropped tail.
    loader=torch.utils.data.DataLoader(indices.tolist(),batch_size=batch_size,shuffle=True,
        drop_last=True,num_workers=0,generator=torch_generator)
    chunks=list(loader)
    if not chunks:
        raise ValueError('Balanced population does not contain a full minibatch')
    order=torch.cat(chunks).numpy()
    batches=ordered_batches(images_bgr,rles,order,batch_size=batch_size,input_side=input_side,training=True,
        python_rng=python_rng,numpy_rng=numpy_rng)
    return dict(batches=batches,population_size=len(indices))
