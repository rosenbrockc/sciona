import numpy as np
import pytest

from sciona.bengali_font_augmentation import FontTransform,augment_font,shear_matrix


def test_identity_center_crop_and_no_input_alias():
    image=np.random.default_rng(39).integers(0,256,(224,224,3),dtype=np.uint8)
    result=augment_font(image,FontTransform(0,0,1,0,0,.5,.5))
    np.testing.assert_array_equal(result,image)
    assert not np.shares_memory(result,image)


def test_historical_crop_never_reaches_last_integer_offset():
    image=np.zeros((224,224,3),dtype=np.uint8)
    result=augment_font(image,FontTransform(0,0,1,0,0,np.nextafter(1.,0.),np.nextafter(1.,0.)))
    assert (result[:209,:209]==0).all()
    assert (result[209:]==255).all() and (result[:,209:]==255).all()


def test_shear_preserves_half_pixel_center():
    center=np.array([127.5,127.5,1.])
    for angle in [-5.,0.,5.]:
        np.testing.assert_allclose(shear_matrix(angle)@center,center,rtol=0,atol=3e-14)


@pytest.mark.parametrize('values',[(6,0,1,0,0,.5,.5),(0,6,1,0,0,.5,.5),(0,0,1.2,0,0,.5,.5),(0,0,1,0,0,1,.5),(0,0,1,0,0,float('nan'),.5)])
def test_invalid_transform_rejected(values):
    with pytest.raises(ValueError):FontTransform(*values)
