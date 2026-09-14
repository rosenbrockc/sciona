"""Source JPEG roundtrip, binary encoding and tile-selection statistics."""
import cv2
import numpy as np


def prepare_tile(image_rgb,mask):
    image_rgb=np.asarray(image_rgb);mask=np.asarray(mask)
    if image_rgb.ndim!=3 or image_rgb.shape[2]!=3 or image_rgb.dtype!=np.uint8 or not image_rgb.size:
        raise ValueError('Nonempty RGB uint8 tile required')
    if mask.shape!=image_rgb.shape[:2] or mask.dtype!=np.uint8 or not np.isin(mask,[0,1]).all():
        raise ValueError('Aligned binary uint8 mask required')
    bgr=cv2.cvtColor(image_rgb,cv2.COLOR_RGB2BGR)
    success,encoded=cv2.imencode('.jpg',bgr)
    if not success:raise ValueError('JPEG encoding failed')
    decoded=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    if decoded is None:raise ValueError('JPEG decoding failed')
    # Source mask2rle uses transpose/flatten and a zero small-mask threshold.
    flat=mask.astype(np.int8).T.flatten()
    padded=np.concatenate(([0],flat,[0]))
    runs=np.where(padded[1:]!=padded[:-1])[0]+1
    runs[1::2]-=runs[::2]
    rle='' if runs[1::2].sum()<=0 else ' '.join(str(x) for x in runs)
    return dict(image_bgr=decoded,rle=rle,num_masked_pixels=mask.sum(),
        ratio_masked_area=mask.sum()/(mask.shape[0]*mask.shape[1]),std_img=bgr.std())


def select_training_tiles(records,*,multiplier_bin):
    """Source std>10 filter and rounded-area presence definition.

    Presence is rounded bin>0, so a nonempty mask may enter the background
    population. The epoch sampler later caps bins; do not cap them here.
    """
    if isinstance(multiplier_bin,bool) or not np.isfinite(multiplier_bin) or multiplier_bin<=0:
        raise ValueError('Positive finite bin multiplier required')
    selected=[record for record in records if record['std_img']>10]
    bins=np.round(np.array([r['ratio_masked_area'] for r in selected])*multiplier_bin).astype(int)
    return dict(images_bgr=[r['image_bgr'] for r in selected],rles=[r['rle'] for r in selected],
        present=bins>0,bins=bins)
