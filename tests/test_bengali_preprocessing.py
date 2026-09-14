import cv2
import numpy as np
import pytest
import torch
from torchvision import transforms

from sciona.bengali_preprocessing import crop_resize,normalize_rgb,prepare_handwriting


def test_center_crop_removes_borders_without_intensity_inversion():
    image=np.zeros((1,137,236),dtype=np.uint8)
    image[:,:,:6]=255
    image[:,:,230:]=255
    image[:,:,6:230]=np.arange(224,dtype=np.uint8)
    actual=crop_resize(image)
    expected=np.broadcast_to(np.arange(224,dtype=np.uint8)[None,None,:,None],(1,224,224,3))
    np.testing.assert_array_equal(actual,expected)
    assert actual[0,0,0,0]==0 and actual[0,-1,-1,0]==223


def test_matches_source_tensor_normalization_and_uint8_resize_order():
    images=np.random.default_rng(827).integers(0,256,size=(2,137,236),dtype=np.uint8)
    before=images.copy()
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    expected=[]
    for image in images:
        rgb=np.array([image,image,image]).transpose(1,2,0)
        resized=cv2.resize(rgb[:,6:230],(224,224),interpolation=cv2.INTER_LINEAR)
        expected.append(transform(resized))
    torch.testing.assert_close(prepare_handwriting(images),torch.stack(expected),rtol=0,atol=0)
    np.testing.assert_array_equal(images,before)


def test_black_and_white_endpoints():
    images=np.stack([np.zeros((224,224,3),dtype=np.uint8),np.full((224,224,3),255,dtype=np.uint8)])
    values=normalize_rgb(images)
    assert (values[0]==-1).all() and (values[1]==1).all()


@pytest.mark.parametrize('images',[np.zeros((1,137,236),dtype=np.float32),np.zeros((0,137,236),dtype=np.uint8),np.zeros((1,224,224),dtype=np.uint8)])
def test_invalid_raw_boundary_rejected(images):
    with pytest.raises(ValueError):crop_resize(images)
