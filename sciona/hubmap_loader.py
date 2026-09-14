"""In-memory source image/RLE preparation and explicitly ordered minibatches."""
import cv2
import numpy as np
import torch
from sciona.hubmap_augmentation import augment


def decode_rle(rle,shape):
    """Decode valid one-based column-major binary runs; empty text is empty mask."""
    if not isinstance(rle,str) or len(shape)!=2 or any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in shape):
        raise ValueError('RLE text and positive image shape required')
    pieces=rle.split()
    if len(pieces)%2:
        raise ValueError('Paired starts and lengths required')
    try:values=[int(value) for value in pieces]
    except ValueError:raise ValueError('Integer RLE values required') from None
    total=shape[0]*shape[1];previous=0
    for start,length in zip(values[::2],values[1::2]):
        if start<1 or length<1 or start-1<previous or start-1+length>total:
            raise ValueError('Ordered nonoverlapping in-range positive runs required')
        previous=start-1+length
    mask=np.zeros(total,dtype=np.uint8)
    for start,length in zip(values[::2],values[1::2]):mask[start-1:start-1+length]=1
    return mask.reshape(shape,order='F')


def prepare_sample(image_bgr,rle,*,input_side,training,python_rng,numpy_rng):
    """Source channel swap, area resize, int8 mask conversion and augmentation.

    image_bgr is the decoded runtime image, equivalent to cv2.imread output.
    No filename, dataset record or image-decoder metadata is retained.
    """
    image_bgr=np.asarray(image_bgr)
    if image_bgr.dtype!=np.uint8 or image_bgr.ndim!=3 or image_bgr.shape[2]!=3 or not image_bgr.size:
        raise ValueError('Nonempty decoded BGR uint8 image required')
    if isinstance(input_side,bool) or not isinstance(input_side,int) or input_side<32 or input_side%32:
        raise ValueError('Network input side must be a positive multiple of32')
    mask=decode_rle(rle,image_bgr.shape[:2])
    image=cv2.cvtColor(image_bgr,cv2.COLOR_RGB2BGR)
    image=cv2.resize(image,(input_side,input_side),interpolation=cv2.INTER_AREA)
    mask=cv2.resize(mask,(input_side,input_side),interpolation=cv2.INTER_AREA)
    return augment(image.astype(np.uint8),mask.astype(np.int8),training=training,
                   python_rng=python_rng,numpy_rng=numpy_rng)


def ordered_batches(images_bgr,rles,indices,*,batch_size,input_side,training,python_rng,numpy_rng):
    """Replay explicit sample order; train drops incomplete tail, validation keeps it.

    Caller supplies already balanced/shuffled original indices; repeated indices
    are meaningful. Historical multiworker scheduling is not inferred here.
    """
    if not isinstance(training,bool) or len(images_bgr)!=len(rles) or not len(images_bgr):
        raise ValueError('Aligned nonempty runtime image/RLE collections required')
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<(2 if training else 1):
        raise ValueError('Valid positive batch size required')
    order=np.asarray(indices)
    if order.ndim!=1 or not np.issubdtype(order.dtype,np.integer) or np.any(order<0) or np.any(order>=len(images_bgr)):
        raise ValueError('Explicit in-range integer indices required')
    length=len(order)//batch_size*batch_size if training else len(order)
    batches=[]
    for start in range(0,length,batch_size):
        values=[prepare_sample(images_bgr[i],rles[i],input_side=input_side,training=training,
            python_rng=python_rng,numpy_rng=numpy_rng) for i in order[start:min(start+batch_size,length)]]
        batches.append(tuple(torch.stack([value[k] for value in values]) for k in range(3)))
    return batches
