"""Checkpoint-backed source inference with batch-bounded tile preparation."""
import numpy as np
import torch
from sciona.hubmap_checkpoint import restore_checkpoint
from sciona.hubmap_inference_network import InferenceNetwork
from sciona.hubmap_inference_tiles import InferenceTiles
from sciona.hubmap_inference import predict_batch


def load_inference_model(payload,*,input_resolution=320,classifier_threshold=.5):
    model=InferenceNetwork((input_resolution,input_resolution),classifier_threshold=classifier_threshold)
    restore_checkpoint(model,payload)
    return model


def predict_slide(models,image_rgb,*,python_rng,torch_generator,resolution=1024,
                  input_resolution=320,pad_size=256,batch_size=12,tta=4,threshold=.5):
    """Preserve source batch boundaries and loader generator consumption.

    Only the decoded source slide, output mask and current prepared batch are
    retained. Model ensemble order and batch-wide classifier shortcut matter.
    """
    for value in [batch_size,tta]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError('Positive integer batch size and TTA required')
    if not np.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('Finite probability threshold required')
    if not isinstance(torch_generator,torch.Generator) or torch_generator.device.type!='cpu':
        raise ValueError('Explicit CPU Torch generator required')
    models=list(models)
    if not models or any(model.training for model in models):
        raise ValueError('Nonempty evaluation model ensemble required')
    tiles=InferenceTiles(image_rgb,resolution=resolution,input_resolution=input_resolution,
                         pad_size=pad_size,python_rng=python_rng)
    def collate(samples):
        return dict(img=torch.stack([s['img'] for s in samples]),p=[s['p'] for s in samples],q=[s['q'] for s in samples])
    loader=torch.utils.data.DataLoader(tiles,batch_size=batch_size,num_workers=0,shuffle=False,
                                      collate_fn=collate,generator=torch_generator)
    result=np.zeros((tiles.h,tiles.w),dtype=np.uint8)
    for data in loader:
        binary=predict_batch(models,data['img'],resolution=resolution,tta=tta,threshold=threshold)
        for index,(p,q) in enumerate(zip(data['p'],data['q'])):
            py0,py1,px0,px1=p;qy0,qy1,qx0,qx1=q
            # Equivalent to source padded grid transpose/crop, without the
            # intermediate full collection of padded binary prediction tiles.
            result[py0:py1,px0:px1]=binary[index,py0-qy0:py1-qy0,px0-qx0:px1-qx0]
    return result
