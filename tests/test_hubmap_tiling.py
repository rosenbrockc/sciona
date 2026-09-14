import numpy as np
import pytest
from sciona.hubmap_tiling import RawTiles


def test_divisible_image_retains_source_empty_boundary_tiles():
    tiles=RawTiles(np.ones((16,16,3),dtype=np.uint8),np.ones((16,16),dtype=np.uint8),tile_size=16)
    assert len(tiles)==4
    assert tiles[0][1].sum()==256
    assert all(not tiles[i][0].any() and not tiles[i][1].any() for i in [1,2,3])


def test_shift_can_make_the_source_grid_empty():
    tiles=RawTiles(np.zeros((7,7,3),dtype=np.uint8),np.zeros((7,7),dtype=np.uint8),tile_size=16,shift_h=8,shift_w=8)
    assert len(tiles)==0
    with pytest.raises(IndexError):tiles[0]


def test_tile_mutation_does_not_modify_runtime_inputs():
    image=np.ones((16,16,3),dtype=np.uint8);mask=np.ones((16,16),dtype=np.uint8)
    tiles=RawTiles(image,mask,tile_size=16)
    a,b=tiles[0];a.fill(0);b.fill(0)
    assert image.all() and mask.all()
