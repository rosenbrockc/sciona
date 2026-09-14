"""Runtime-only detector sampling; image identities remain caller-owned integers."""
import math
import numpy as np
from sciona.dsb_components import _integer, oversample_by_size


def detector_sampling_table(boxes_by_image, resolution=1.):
    """Source 1/2/4 additive size repeats, retaining image association and order."""
    if not math.isfinite(resolution) or resolution<=0:
        raise ValueError('resolution must be positive and finite')
    rows=[]
    for index,boxes in enumerate(boxes_by_image):
        boxes=np.asarray(boxes,dtype=float)
        if boxes.ndim!=2 or boxes.shape[1]!=4 or not np.all(np.isfinite(boxes)) or np.any(boxes[:,3]<=0):
            raise ValueError('each image needs finite boxes with positive diameter')
        repeated=oversample_by_size(boxes,3,thresholds=np.array([6.,30.,40.])/resolution)
        if len(repeated):rows.append(np.column_stack([np.full(len(repeated),index),repeated]))
    return np.concatenate(rows) if rows else np.empty((0,5),dtype=float)


def detector_epoch_length(table, random_fraction=.3):
    """Integer loader length, explicitly flooring the source float expression."""
    if not math.isfinite(random_fraction) or not 0<=random_fraction<1:
        raise ValueError('random fraction must be in [0,1)')
    if not len(table):raise ValueError('source detector sampling requires at least one eligible target')
    return int(len(table)/(1-random_fraction))


def select_detector_sample(index, table, eligible_image_indices, *, image_count, rng=None):
    """Select target-centered, same-image random, or independent-image random crop.

    Explicit global image indices repair the source filtered-list association:
    its random-image branch indexes boxes by filtered position instead of the
    selected image's global position. No filename-based eligibility inference.
    """
    index=_integer(index,'sample index',0);image_count=_integer(image_count,'image count')
    table=np.asarray(table,dtype=float)
    eligible=tuple(_integer(i,'eligible image index',0) for i in eligible_image_indices)
    if (table.ndim!=2 or table.shape[1]!=5 or not len(table) or not np.all(np.isfinite(table))
            or np.any(table[:,0]!=np.floor(table[:,0])) or np.any(table[:,0]<0)
            or np.any(table[:,0]>=image_count) or np.any(table[:,4]<=0)
            or any(i>=image_count for i in eligible)):
        raise ValueError('invalid sampling table or image associations')
    rng=np.random.RandomState() if rng is None else rng
    random_crop=index>=len(table)
    independent=bool(rng.randint(2)) if random_crop else False
    row=table[index%len(table)]
    if independent:
        if not eligible:raise ValueError('independent random crop requires eligible images')
        image_index=eligible[int(rng.randint(len(eligible)))]
        target=None
    else:
        image_index=int(row[0]);target=row[1:].copy()
    return dict(image_index=image_index,target=target,random_crop=random_crop,independent_image=independent)
