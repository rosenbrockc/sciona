import numpy as np
import pytest

from sciona.dfdc_difference_mask import difference_mask


def test_identical_crops_have_zero_mask():
    image=np.random.default_rng(718).integers(0,256,(17,19,3),dtype=np.uint8)
    np.testing.assert_array_equal(difference_mask(image,image),np.zeros((17,19),dtype=np.uint8))


def test_constant_offset_has_independent_ssim_oracle():
    a=np.full((11,13,3),100,dtype=np.uint8)
    b=np.full_like(a,140)
    similarity=(2*100*140+(0.01*255)**2)/(100**2+140**2+(0.01*255)**2)
    expected=int((1-similarity)*255)
    np.testing.assert_array_equal(difference_mask(a,b),np.full((11,13),expected,dtype=np.uint8))


def test_negative_similarity_wraps_instead_of_clipping():
    board=(np.indices((15,15)).sum(axis=0)%2*255).astype(np.uint8)
    a=np.repeat(board[:,:,None],3,axis=2)
    got=difference_mask(a,255-a)
    # Interior 7x7 checkerboard: 25 white and 24 black, inverse pairing.
    mean_a,mean_b=255*25/49,255*24/49
    variance=(25*(255-mean_a)**2+24*mean_a**2)/48
    similarity=((2*mean_a*mean_b+2.55**2)*(-2*variance+7.65**2))/((mean_a**2+mean_b**2+2.55**2)*(2*variance+7.65**2))
    assert similarity<0
    assert got[7,7]==int((1-similarity)*255)%256
    assert got[7,7]!=255


@pytest.mark.parametrize('a,b', [
    (np.zeros((6,8,3),dtype=np.uint8),np.zeros((6,8,3),dtype=np.uint8)),
    (np.zeros((8,8,3),dtype=np.uint8),np.zeros((8,9,3),dtype=np.uint8)),
])
def test_uncomputable_pair_is_missing_mask(a,b):
    assert difference_mask(a,b) is None


def test_nonuint8_rejected():
    with pytest.raises(ValueError):difference_mask(np.zeros((8,8,3)),np.zeros((8,8,3)))
