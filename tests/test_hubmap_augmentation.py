import random
import numpy as np
import pytest
import torch
from sciona.hubmap_augmentation import augment


def test_binary_mask_is_not_divided_by255():
    image=np.full((32,32,3),127,dtype=np.uint8)
    mask=np.ones((32,32),dtype=np.int8)
    _,target,label=augment(image,mask,training=False,python_rng=random.Random(1),numpy_rng=np.random.RandomState(1))
    assert torch.equal(target,torch.ones(1,32,32)) and label.item()==1


def test_training_does_not_mutate_inputs():
    image=np.random.RandomState(5).randint(0,256,(32,32,3)).astype(np.uint8)
    mask=np.ones((32,32),dtype=np.int8)
    before=image.copy()
    augment(image,mask,training=True,python_rng=random.Random(5),numpy_rng=np.random.RandomState(1))
    np.testing.assert_array_equal(image,before)
    np.testing.assert_array_equal(mask,np.ones_like(mask))


@pytest.mark.parametrize('problem',['uint8_mask','fractional_image','nonsquare','mask_values'])
def test_invalid_contract_preserves_rng(problem):
    image=np.zeros((32,32,3),dtype=np.uint8);mask=np.zeros((32,32),dtype=np.int8)
    if problem=='uint8_mask':mask=mask.astype(np.uint8)
    if problem=='fractional_image':image=image.astype(float)
    if problem=='nonsquare':image=image[:16];mask=mask[:16]
    if problem=='mask_values':mask[0,0]=2
    pr=random.Random(1);nr=np.random.RandomState(1)
    p=pr.getstate();n=nr.get_state()
    with pytest.raises(ValueError):augment(image,mask,training=True,python_rng=pr,numpy_rng=nr)
    assert pr.getstate()==p
    assert nr.get_state()[2:]==n[2:]
    np.testing.assert_array_equal(nr.get_state()[1],n[1])
