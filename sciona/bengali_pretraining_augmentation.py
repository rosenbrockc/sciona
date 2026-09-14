"""Font-classifier pretraining transform from the recovered winner notebook."""
from dataclasses import dataclass
import math

from sciona.bengali_font_augmentation import FontTransform,augment_font


@dataclass(frozen=True)
class PretrainingTransform(FontTransform):
    cutout_y: int
    cutout_x: int

    def __post_init__(self):
        # Same geometry as CycleGAN but a source-confirmed wider angle range.
        for angle in (self.shear,self.angle):
            if isinstance(angle,bool) or not isinstance(angle,(int,float)) or not math.isfinite(angle):
                raise ValueError('finite pretraining angles required')
        FontTransform(self.shear/4,self.angle/4,self.scale,self.dx,self.dy,self.h_start,self.w_start)
        if any(type(value) is not int or not 0<=value<=224 for value in (self.cutout_y,self.cutout_x)):
            raise ValueError('integer cutout centers within inclusive0..224 required')


def augment_pretraining(image,parameters):
    if not isinstance(parameters,PretrainingTransform):
        raise ValueError('explicit pretraining parameters required')
    result=augment_font(image,parameters)
    # Historical Cutout samples an inclusive image-boundary center; hole size
    # is fixed before clipping, rather than independently sampled up to112.
    top,bottom=max(0,parameters.cutout_y-56),min(224,parameters.cutout_y+56)
    left,right=max(0,parameters.cutout_x-56),min(224,parameters.cutout_x+56)
    result[top:bottom,left:right]=128
    return result
