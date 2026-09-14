"""Private in-memory populations and paired CycleGAN training batches.

Inputs must already align image order with identity order. No identities,
images or per-record provenance are persisted by this module.
"""
import numpy as np
import torch

from sciona.bengali_font_augmentation import augment_font
from sciona.bengali_font_sampling import FontParameterSampler
from sciona.bengali_joint_labels import decode_joint_classes
from sciona.bengali_label_corrections import corrected_joint_labels
from sciona.bengali_preprocessing import crop_resize,normalize_rgb,prepare_handwriting
from sciona.bengali_sampling import SamplingBudget,paired_batches


class ImagePopulation:
    def __init__(self,images,joint_labels):
        images=np.asarray(images)
        labels=np.asarray(joint_labels)
        if images.dtype!=np.uint8 or images.ndim!=3 or not len(images) or images.shape[1:]!=(137,236):
            raise ValueError('nonempty uint8 grayscale population required')
        decode_joint_classes(labels)
        if len(labels)!=len(images):
            raise ValueError('image and label populations must align')
        self.images=images.copy()
        self.labels=labels.astype(np.int64,copy=True)
        self.images.flags.writeable=False
        self.labels.flags.writeable=False

    @classmethod
    def corrected(cls,images,identities,components,correction_identities,corrections):
        return cls(images,corrected_joint_labels(identities,components,correction_identities,corrections))


def training_batches(hand,font,budget,*,hand_seed,font_seed,font_sampler):
    if not isinstance(hand,ImagePopulation) or not isinstance(font,ImagePopulation):
        raise ValueError('validated image populations required')
    if (not isinstance(budget,SamplingBudget) or budget.hand_count!=len(hand.images)
            or budget.font_count!=len(font.images)):
        raise ValueError('training budget must match both populations')
    if not isinstance(font_sampler,FontParameterSampler):
        raise ValueError('explicit font parameter sampler required')
    for pair in paired_batches(budget,hand_seed=hand_seed,font_seed=font_seed):
        hand_indices=pair['hand_indices'].numpy()
        font_indices=pair['font_indices'].numpy()
        images_a=prepare_handwriting(hand.images[hand_indices])
        font_rgb=crop_resize(font.images[font_indices])
        images_b=normalize_rgb(np.stack([augment_font(image,font_sampler.draw()) for image in font_rgb]))
        yield dict(epoch=pair['epoch'],step=pair['step'],images_a=images_a,images_b=images_b,
            labels_a=torch.from_numpy(hand.labels[hand_indices].copy()),
            labels_b=torch.from_numpy(font.labels[font_indices].copy()))
