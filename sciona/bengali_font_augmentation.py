"""Explicit-parameter reconstruction of the historical font augmentation.

Geometry follows inspected Albumentations 0.4.x/imgaug 0.3.0 candidates.
Historical environment identity and stochastic draw equivalence are separate
qualification requirements; this module does not infer sampled parameters.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class FontTransform:
    shear: float
    angle: float
    scale: float
    dx: float
    dy: float
    h_start: float
    w_start: float

    def __post_init__(self):
        values=[self.shear,self.angle,self.scale,self.dx,self.dy,self.h_start,self.w_start]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
            raise ValueError('finite numeric font-transform parameters required')
        if (abs(self.shear)>5 or abs(self.angle)>5 or not .9<=self.scale<=1.1
                or abs(self.dx)>.0625 or abs(self.dy)>.0625
                or not 0<=self.h_start<1 or not 0<=self.w_start<1):
            raise ValueError('font-transform parameters exceed historical ranges')


def shear_matrix(degrees):
    # imgaug Uniform returns float32, including the constant zero rotation.
    # Its joint deg2rad conversion therefore rounds in float32 as well.
    radians=float(np.deg2rad(np.float32(degrees)))
    affine=np.array([[1.,-math.sin(radians),0.],[0.,math.cos(radians),0.],[0.,0.,1.]])
    center=127.5
    before=np.array([[1.,0.,-center],[0.,1.,-center],[0.,0.,1.]])
    after=np.array([[1.,0.,center],[0.,1.,center],[0.,0.,1.]])
    return after @ affine @ before


def augment_font(image, parameters):
    image=np.asarray(image)
    if image.dtype!=np.uint8 or image.shape!=(224,224,3):
        raise ValueError('uint8 224x224 RGB image required')
    if not isinstance(parameters,FontTransform):
        raise ValueError('validated explicit font-transform parameters required')
    padded=cv2.copyMakeBorder(image,16,16,16,16,cv2.BORDER_CONSTANT,value=(255,255,255))
    matrix=shear_matrix(parameters.shear)
    # imgaug skips matrices close to identity before calling its warp backend.
    if np.average(np.abs(matrix-np.eye(3,dtype=np.float32)))>1e-4:
        padded=cv2.warpAffine(padded,matrix[:2],(256,256),flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    rotation=cv2.getRotationMatrix2D((128.,128.),parameters.angle,parameters.scale)
    rotation[0,2]+=parameters.dx*256
    rotation[1,2]+=parameters.dy*256
    warped=cv2.warpAffine(padded,rotation,(256,256),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    top,left=int(32*parameters.h_start),int(32*parameters.w_start)
    return warped[top:top+224,left:left+224].copy()
