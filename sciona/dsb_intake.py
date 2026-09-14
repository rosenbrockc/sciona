"""Runtime proposal annotation and validation samples from pinned DSB semantics.

Cubic IoU follows the MIT source; see docs/licenses/DSB2017-MIT.txt.
All coordinates must already share the preprocessed volume frame.
"""
import numpy as np

from sciona.dsb_components import (suppress_proposals,prepare_classifier_batch,
                                    assign_anchor_labels)
from sciona.dsb_detector_crop import crop_detection


def _cubic_iou(box0,box1):
    r0=box0[3]/2;s0=box0[:3]-r0;e0=box0[:3]+r0
    r1=box1[3]/2;s1=box1[:3]-r1;e1=box1[:3]+r1
    overlap=[max(0,min(e0[i],e1[i])-max(s0[i],s1[i])) for i in range(3)]
    intersection=overlap[0]*overlap[1]*overlap[2]
    union=box0[3]*box0[3]*box0[3]+box1[3]*box1[3]*box1[3]-intersection
    return intersection/union


def annotate_classifier_proposals(proposals,boxes,*,confidence_threshold=-1.,
                                  nms_threshold=.05,detection_threshold=.05):
    """Strict confidence filter, source NMS, then strict IoU known-label assignment."""
    proposals=np.asarray(proposals);boxes=np.asarray(boxes)
    if (proposals.ndim!=2 or proposals.shape[1]!=5 or boxes.ndim!=2 or boxes.shape[1]!=4
            or not np.all(np.isfinite(proposals)) or not np.all(np.isfinite(boxes))
            or np.any(proposals[:,4]<=0) or np.any(boxes[:,3]<=0)
            or not np.isfinite(confidence_threshold) or not 0<=detection_threshold<=1):
        raise ValueError('finite proposals and boxes with positive diameters required')
    retained=suppress_proposals(proposals[proposals[:,0]>confidence_threshold],nms_threshold)
    known=np.array([any(_cubic_iou(p[1:5],box)>detection_threshold for box in boxes)
                    for p in retained],dtype=bool)
    return retained,known


def classifier_validation_sample(volume,proposals,known_proposals,case_label,*,topk=5,crop_size=96):
    """Top-score crops without augmentation; retain zero slots and label alignment."""
    known=np.asarray(known_proposals)
    if known.shape!=(len(proposals),) or not np.all((known==0)|(known==1)) or case_label not in (0,1):
        raise ValueError('aligned binary proposal and case labels required')
    images,coords,chosen=prepare_classifier_batch(volume,proposals,topk,crop_size)
    indicators=np.zeros(images.shape[0],dtype=np.int32)
    indicators[:len(chosen)]=known[chosen]
    return images,coords,indicators,np.array([case_label],dtype=np.int64)


def detector_validation_sample(volume,target,boxes,*,crop_size=128,bound_size=12,
                                rng=None,label_rng=None):
    """Target-centered unscaled crop; validation anchor threshold and no subsampling."""
    sample,selected,transformed,coords=crop_detection(volume,target,boxes,
        crop_size=crop_size,bound_size=bound_size,scale=False,random_crop=False,rng=rng)
    labels=assign_anchor_labels(sample.shape[1:],selected,transformed,phase='val',rng=label_rng)
    return sample.astype(np.float32),labels,coords
