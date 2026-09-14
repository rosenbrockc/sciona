"""Source ordered model/flip averaging, resize, threshold and tile stitching."""
import cv2
import numpy as np
import torch


def predict_batch(models,batch,*,resolution,tta,threshold):
    """Source model-major flip averaging and resize before thresholding."""
    probabilities=0
    for model in models:
        with torch.no_grad():
            for axes in [(),(-1,),(-2,),(-1,-2)][:min(tta,4)]:
                logits=model(batch.flip(axes) if axes else batch)
                if not isinstance(logits,torch.Tensor) or logits.shape!=(len(batch),1,batch.shape[2],batch.shape[3]) or not torch.isfinite(logits).all():
                    raise ValueError('Model must return finite N1SS logits')
                values=torch.sigmoid(logits).detach().cpu().numpy()[:,0,:,:]
                if -2 in axes:values=values[:,::-1,:]
                if -1 in axes:values=values[:,:,::-1]
                probabilities+=values
    probabilities=probabilities/min(tta,4)/len(models)
    resized=np.vstack([cv2.resize(value.astype(np.float32),(resolution,resolution))[None] for value in probabilities])
    binary=(resized>threshold).astype(np.uint8)
    return binary


def predict_tiles(models, images, interiors, windows, *, height, width, resolution,
                  pad_size, batch_size=12, tta=4, threshold=.5):
    """Consume normalized CPU tiles and source raster coordinates.

    Models must return plain N1HW logits. Model construction, classifier shortcut
    policy, raw raster IO and image normalization belong to the calling stage.
    """
    for value in [height,width,resolution,batch_size,tta]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError('Positive integer geometry, batch size and TTA required')
    if isinstance(pad_size,bool) or not isinstance(pad_size,int) or not 0<=pad_size<resolution/2:
        raise ValueError('Padding must leave a positive tile interior')
    if not np.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('Finite probability threshold required')
    models=list(models)
    if not models or any(model.training for model in models):
        raise ValueError('Nonempty evaluation-mode model ensemble required')
    side=resolution-2*pad_size
    nh=height//side+1;nw=width//side+1
    if not isinstance(images,torch.Tensor) or images.device.type!='cpu' or images.dtype!=torch.float32 or images.ndim!=4 or images.shape[1]!=3 or images.shape[2]!=images.shape[3] or not torch.isfinite(images).all():
        raise ValueError('Finite normalized float32 CPU N3SS images required')
    if len(images)!=nh*nw or len(interiors)!=len(images) or len(windows)!=len(images):
        raise ValueError('Complete ordered tile grid required')
    for index,(p,q) in enumerate(zip(interiors,windows)):
        y=index//nw*side;x=index%nw*side
        expected_p=[y,min(y+side,height),x,min(x+side,width)]
        expected_q=[max(0,y-pad_size),min(y+side+pad_size,height),max(0,x-pad_size),min(x+side+pad_size,width)]
        if list(p)!=expected_p or list(q)!=expected_q:
            raise ValueError('Coordinates must follow source grid order')
    result=np.zeros((len(images),side,side),dtype=np.uint8)
    for start in range(0,len(images),batch_size):
        batch=images[start:start+batch_size]
        binary=predict_batch(models,batch,resolution=resolution,tta=tta,threshold=threshold)
        for j in range(len(batch)):
            py0,py1,px0,px1=interiors[start+j]
            qy0,qy1,qx0,qx1=windows[start+j]
            result[start+j,:py1-py0,:px1-px0]=binary[j,py0-qy0:py1-qy0,px0-qx0:px1-qx0]
    return result.reshape(nh,nw,side,side).transpose(0,2,1,3).reshape(nh*side,nw*side)[:height,:width]
