"""Active DFDC B7 sample preparation from explicit arrays and dependencies.

MIT 2020 Selim Seferbekov, 89c6290490bac96b29193a4061b3db9dd3933e36;
see docs/licenses/DFDC-MIT.txt. Active defaults: hardcore enabled, rotation off,
padding_part=3. No file reads, identity output, or unbounded broken-file retries.
"""
import random
import numpy as np
from albumentations.pytorch.functional import img_to_tensor

from sciona.dfdc_landmark_removal import remove_landmark
from sciona.dfdc_convex_hull import blackout_convex_hull
from sciona.dfdc_occlusion_masks import prepare_bit_masks


def prepare_sample(image, label, *, mode, transforms, detector, predictor,
                   mask=None, landmarks=None, label_smoothing=.01):
    if mode not in ('train','val'):
        raise ValueError('mode must be train or val')
    if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3
            or image.shape[-1]!=3 or min(image.shape[:2])<1):
        raise ValueError('image must be a nonempty uint8 RGB array')
    if not np.isscalar(label) or label not in (0,1):
        raise ValueError('label must be binary before smoothing')
    if not np.isscalar(label_smoothing) or not 0<=label_smoothing<.5:
        raise ValueError('label smoothing must be in [0,.5)')
    if mode=='train' and (not callable(detector) or not callable(predictor)):
        raise ValueError('training requires explicit hull detector and predictor')
    if mask is not None and (not isinstance(mask,np.ndarray) or mask.dtype!=np.uint8 or mask.shape!=image.shape[:2]):
        raise ValueError('mask must match the image spatial shape and be uint8')
    if landmarks is not None and (not isinstance(landmarks,np.ndarray) or landmarks.shape!=(5,2) or landmarks.dtype.kind not in 'iu'):
        raise ValueError('landmarks must be a five-point integer array')
    image=image.copy()
    mask=np.zeros(image.shape[:2],dtype=np.uint8) if mask is None else mask.copy()
    if mode=='train':
        label=np.clip(label,label_smoothing,1-label_smoothing)
        if landmarks is not None and random.random()<.7:
            image=remove_landmark(image,landmarks)
        elif random.random()<.2:
            blackout_convex_hull(image,detector,predictor)
        elif random.random()<.1:
            binary_mask=mask>.4*255
            masks=prepare_bit_masks((binary_mask*1).astype(np.uint8))
            current_try=1
            while current_try<6:
                bitmap_msk=random.choice(masks)
                if label<.5 or np.count_nonzero(mask*bitmap_msk)>20:
                    mask*=bitmap_msk
                    image*=np.expand_dims(bitmap_msk,axis=-1)
                    break
                current_try+=1
    valid_label=int(np.count_nonzero(mask[mask>20])>32 or label<.5)
    if transforms:
        data=transforms(image=image,mask=mask)
        image,mask=data['image'],data['mask']
    image=img_to_tensor(image,{'mean':[.485,.456,.406],'std':[.229,.224,.225]})
    return {'image':image,'labels':np.array((label,)),'valid':valid_label,'rotations':0}
