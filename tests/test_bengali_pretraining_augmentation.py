import numpy as np
import pytest
from sciona.bengali_pretraining_augmentation import PretrainingTransform,augment_pretraining


@pytest.mark.parametrize('center,area',[(112,112**2),(0,56**2),(224,56**2)])
def test_source_fixed_cutout_size_clips_at_boundaries(center,area):
    image=np.zeros((224,224,3),dtype=np.uint8)
    result=augment_pretraining(image,PretrainingTransform(0,0,1,0,0,.5,.5,center,center))
    assert (result[:,:,0]==128).sum()==area
    assert not image.any()


def test_pretraining_accepts_source_wider_angles_and_preserves_shape():
    image=np.random.default_rng(951).integers(0,256,(224,224,3),dtype=np.uint8)
    result=augment_pretraining(image,PretrainingTransform(20,-20,1,0,0,.5,.5,112,112))
    assert result.dtype==np.uint8 and result.shape==image.shape
    assert (result[56:168,56:168]==128).all()


@pytest.mark.parametrize('arguments',[(21,0,112,112),(0,-21,112,112),(0,0,225,112),(0,0,112,-1)])
def test_invalid_pretraining_parameters_rejected(arguments):
    shear,angle,y,x=arguments
    with pytest.raises(ValueError):PretrainingTransform(shear,angle,1,0,0,.5,.5,y,x)
