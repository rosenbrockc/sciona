"""Source inference windows and normalized tensors from decoded RGB slides."""
import random
import cv2
import numpy as np
import torch


class InferenceTiles:
    def __init__(self,image_rgb,*,resolution=1024,input_resolution=320,pad_size=256,python_rng):
        image_rgb=np.asarray(image_rgb)
        if image_rgb.dtype!=np.uint8 or image_rgb.ndim!=3 or image_rgb.shape[2]!=3 or not image_rgb.size:
            raise ValueError('Nonempty RGB uint8 slide required')
        for value in [resolution,input_resolution]:
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError('Positive integer resolutions required')
        if isinstance(pad_size,bool) or not isinstance(pad_size,int) or not 0<=pad_size<resolution/2:
            raise ValueError('Padding must leave a positive tile interior')
        if not isinstance(python_rng,random.Random):
            raise ValueError('Explicit Python Random required')
        self.image=image_rgb
        self.h,self.w=image_rgb.shape[:2]
        self.sz=resolution;self.input_sz=input_resolution;self.pad_sz=pad_size
        self.pred_sz=resolution-2*pad_size
        self.num_h=self.h//self.pred_sz+1;self.num_w=self.w//self.pred_sz+1
        self.rng=python_rng

    def __len__(self):return self.num_h*self.num_w

    def __getitem__(self,index):
        if isinstance(index,bool) or not isinstance(index,(int,np.integer)) or not 0<=index<len(self):
            raise IndexError('Tile index outside the source grid')
        y=index//self.num_w*self.pred_sz;x=index%self.num_w*self.pred_sz
        p=[y,min(y+self.pred_sz,self.h),x,min(x+self.pred_sz,self.w)]
        q=[max(0,y-self.pad_sz),min(y+self.pred_sz+self.pad_sz,self.h),
           max(0,x-self.pad_sz),min(x+self.pred_sz+self.pad_sz,self.w)]
        qy0,qy1,qx0,qx1=q
        image=np.zeros((self.sz,self.sz,3),dtype=np.uint8)
        # Source places the clipped window at the tile's top-left, even at
        # top/left slide boundaries; it does not center missing context.
        image[:qy1-qy0,:qx1-qx0]=self.image[qy0:qy1,qx0:qx1]
        if self.sz!=self.input_sz:
            image=cv2.resize(image,(self.input_sz,self.input_sz),interpolation=cv2.INTER_AREA)
        self.rng.random()  # Compose
        self.rng.random()  # Normalize
        mean=np.array([.485,.456,.406],dtype=np.float32)*255.
        std=np.array([.229,.224,.225],dtype=np.float32)*255.
        normalized=image.astype(np.float32)
        normalized-=mean;normalized*=np.reciprocal(std,dtype=np.float32)
        self.rng.random()  # ToTensorV2 BasicTransform (unlike old ToTensor)
        return dict(img=torch.from_numpy(normalized.transpose(2,0,1)),p=p,q=q)
