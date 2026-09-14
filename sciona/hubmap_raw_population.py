"""Initial source grid preparation and selection in source concatenation order."""
import numpy as np
from sciona.hubmap_tiling import RawTiles
from sciona.hubmap_preparation import prepare_tile,select_training_tiles


def prepare_population(slides,*,tile_size=1024,multiplier_bin=20):
    """Prepare unshifted grids, then half-shifted grids, preserving slide order.

    Within each slide/grid the source reverses NumPy argsort of mask-pixel
    counts before JPEG generation. That ordering, including ties, affects later
    seeded sampling and is preserved. Empty source grids reject explicitly.
    """
    slides=list(slides)
    if not slides:
        raise ValueError('Nonempty ordered slide collection required')
    if isinstance(tile_size,bool) or not isinstance(tile_size,int) or tile_size<2:
        raise ValueError('Integer tile size at least two required')
    records=[]
    for shift in [0,tile_size//2]:
        for image,mask in slides:
            tiles=RawTiles(image,mask,tile_size=tile_size,shift_h=shift,shift_w=shift)
            if not len(tiles):
                raise ValueError('Source preparation cannot stack an empty tile grid')
            # Retain only counts before ordering, avoiding a second collection
            # of full image tiles in memory.
            counts=np.array([tiles[i][1].sum() for i in range(len(tiles))])
            order=np.argsort(counts)[::-1]
            for index in order:
                image_tile,mask_tile=tiles[index]
                records.append(prepare_tile(image_tile,mask_tile))
    return select_training_tiles(records,multiplier_bin=multiplier_bin)
