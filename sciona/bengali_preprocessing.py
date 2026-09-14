"""Deterministic CycleGAN preprocessing from the pinned Bengali notebook.

Font augmentation belongs between crop/resize and tensor normalization. The
main seen/OOD classifiers have separate preprocessing contracts.
"""
import cv2
import numpy as np
import torch


def crop_resize(images):
    images=np.asarray(images)
    if images.dtype!=np.uint8 or images.ndim!=3 or not len(images) or images.shape[1:]!=(137,236):
        raise ValueError('nonempty uint8 grayscale N137x236 images required')
    result=[]
    for image in images:
        rgb=np.repeat(image[:,:,None],3,axis=2)
        # Source center crop removes six columns from either side.
        result.append(cv2.resize(rgb[:,6:230],(224,224),interpolation=cv2.INTER_LINEAR))
    return np.stack(result)


def normalize_rgb(images):
    images=np.asarray(images)
    if images.dtype!=np.uint8 or images.ndim!=4 or not len(images) or images.shape[1:]!=(224,224,3):
        raise ValueError('nonempty uint8 N224x224x3 images required')
    tensor=torch.from_numpy(np.ascontiguousarray(images.transpose(0,3,1,2))).to(dtype=torch.float32)
    # Match ToTensor then Normalize rounding; do not replace with x/127.5-1.
    return tensor.div_(255.).sub_(.5).div_(.5)


def prepare_handwriting(images):
    return normalize_rgb(crop_resize(images))
