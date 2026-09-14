import numpy as np
from sciona.hubmap_preparation import prepare_tile,select_training_tiles


def test_nonempty_small_mask_can_be_background_for_balancing():
    image=np.random.RandomState(1).randint(0,256,(32,32,3)).astype(np.uint8)
    mask=np.zeros((32,32),dtype=np.uint8);mask[0,0]=1
    record=prepare_tile(image,mask)
    result=select_training_tiles([record],multiplier_bin=10)
    assert record['rle'] and not result['present'][0]


def test_flat_tile_rejected_by_source_std_filter():
    record=prepare_tile(np.zeros((32,32,3),dtype=np.uint8),np.zeros((32,32),dtype=np.uint8))
    assert select_training_tiles([record],multiplier_bin=10)['images_bgr']==[]


def test_statistics_precede_lossy_jpeg_roundtrip():
    image=np.random.RandomState(2).randint(0,256,(32,32,3)).astype(np.uint8)
    record=prepare_tile(image,np.ones((32,32),dtype=np.uint8))
    assert record['std_img']==image[:,:,::-1].std()
    assert record['std_img']!=record['image_bgr'].std()
