"""HuBMAP's pinned Albumentations 0.5.2 recipe with explicit RNG streams."""
import random
import cv2
import numpy as np
import torch


def augment(image,mask,*,training,python_rng,numpy_rng):
    """Transform RGB uint8 images and binary int8 masks, preserving source draws.

    The old ToTensor receives an int8 mask deliberately: uint8 would divide the
    mask by255. Output is float32 CHW image, 1HW mask and post-augmentation label.
    Current OpenCV is used; original historical binary-library parity is excluded.
    """
    if not isinstance(training,bool) or not isinstance(python_rng,random.Random) or not isinstance(numpy_rng,np.random.RandomState):
        raise ValueError('Boolean mode and explicit Python/NumPy RNG streams required')
    image=np.asarray(image);mask=np.asarray(mask)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or image.shape[0]!=image.shape[1] or image.shape[0]<16:
        raise ValueError('Square RGB uint8 image at least16pixels required')
    if mask.dtype!=np.int8 or mask.shape!=image.shape[:2] or not np.isin(mask,[0,1]).all():
        raise ValueError('Aligned binary int8 mask required')
    image=image.copy();mask=mask.copy();height,width=mask.shape
    r=python_rng
    r.random()  # Compose(p=1)
    if training:
        r.random()  # RandomRotate90(p=1)
        factor=r.randint(0,3)
        image=np.ascontiguousarray(np.rot90(image,factor));mask=np.ascontiguousarray(np.rot90(mask,factor))
        if r.random()<.5:
            image=cv2.flip(image,1);mask=np.ascontiguousarray(mask[:,::-1])
        if r.random()<.5:
            angle=r.uniform(-30,30);scale=r.uniform(.8,1.2)
            dx=r.uniform(0,0);dy=r.uniform(0,0)
            matrix=cv2.getRotationMatrix2D((width/2,height/2),angle,scale)
            matrix[0,2]+=dx*width;matrix[1,2]+=dy*height
            image=cv2.warpAffine(image,matrix,(width,height),flags=cv2.INTER_LINEAR,borderMode=0,borderValue=(0,0,0))
            mask=cv2.warpAffine(mask,matrix,(width,height),flags=cv2.INTER_NEAREST,borderMode=0,borderValue=None)
        if r.random()<.5:
            sigma=r.uniform(0,50.)**.5
            noise_rng=np.random.RandomState(r.randint(0,2**32-1))
            noise=noise_rng.normal(0,sigma,image.shape)
            image=np.clip(image.astype('float32')+noise,0,255).astype(np.uint8)
        if r.random()<.5:
            size=numpy_rng.randint(3,8)
            if size!=0 and size%2!=1:size=(size+1)%8
            sigma=r.uniform(0,0)
            image=cv2.GaussianBlur(image,(size,size),sigmaX=sigma)
        if r.random()<.5:
            alpha=1.+r.uniform(-.5,.5);beta=r.uniform(-.35,.35)
            lut=np.arange(256).astype('float32')
            if alpha!=1:lut*=alpha
            if beta!=0:lut+=beta*255
            image=cv2.LUT(image,np.clip(lut,0,255).astype(np.uint8))
        if r.random()<.5:
            hue=r.uniform(-30,30);saturation=r.uniform(-30,30);value=r.uniform(0,0)
            hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV);h,s,v=cv2.split(hsv)
            if hue!=0:h=cv2.LUT(h,np.mod(np.arange(256,dtype=np.int16)+hue,180).astype(np.uint8))
            if saturation!=0:s=cv2.LUT(s,np.clip(np.arange(256,dtype=np.int16)+saturation,0,255).astype(np.uint8))
            if value!=0:v=cv2.LUT(v,np.clip(np.arange(256,dtype=np.int16)+value,0,255).astype(np.uint8))
            image=cv2.cvtColor(cv2.merge((h,s,v)).astype(np.uint8),cv2.COLOR_HSV2RGB)
        if r.random()<.5:
            for _ in range(r.randint(1,2)):
                hole_height=r.randint(height//16,height//4);hole_width=r.randint(width//16,width//4)
                y=r.randint(0,height-hole_height);x=r.randint(0,width-hole_width)
                image[y:y+hole_height,x:x+hole_width]=0
                mask[y:y+hole_height,x:x+hole_width]=0
    r.random()  # Normalize(p=1); ToTensor.__call__ consumes no draw.
    mean=np.array([.485,.456,.406],dtype=np.float32)*255.
    std=np.array([.229,.224,.225],dtype=np.float32)*255.
    normalized=image.astype(np.float32)
    normalized-=mean;normalized*=np.reciprocal(std,dtype=np.float32)
    tensor=torch.from_numpy(np.moveaxis(normalized/1,-1,0).astype(np.float32))
    target=torch.from_numpy(np.expand_dims(mask/1,0).astype(np.float32))
    return tensor,target,(target.sum()>0).float()
