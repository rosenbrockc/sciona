"""Assemble DFDC paired training records without filesystem identities.

Source MIT2020SelimSeferbekov,89c6290490bac96b29193a4061b3db9dd3933e36.
See docs/licenses/DFDC-MIT.txt. Caller order and shuffle seed replace source
unordered process/set collection followed by an unseeded shuffle.
"""
import random

from sciona.dfdc_folds import assign_folds,eligible_crop_positions
from sciona.dfdc_training_crops import detect_training_boxes,crop_training_clip
from sciona.dfdc_crop_annotations import annotate_pair
from sciona.dfdc_difference_mask import difference_mask


def build_records(clips, *, box_detector, landmark_detector, record_order_seed, n_splits=16):
    """Clips supply frames,part and original positional reference; no names needed.

    Originals without an altered counterpart are excluded, matching source pair
    discovery. Returned arrays are runtime data and must not be published.
    """
    if type(record_order_seed) is not int or record_order_seed<0:
        raise ValueError('record_order_seed must be a nonnegative integer')
    clips=tuple(clips)
    folds=assign_folds([c['part'] for c in clips],[c['original'] for c in clips],n_splits=n_splits)
    altered=[i for i,c in enumerate(clips) if c['original']!=i]
    originals=sorted({clips[i]['original'] for i in altered})
    cache={};records=[]
    def record(clip_position,crop,*,mask,landmarks):
        return {'image':crop['image'],'label':int(clips[clip_position]['original']!=clip_position),
                'clip_position':clip_position,'original_position':clips[clip_position]['original'],
                'frame':crop['frame'],'actor':crop['actor'],'fold':int(folds[clip_position]),
                'mask':mask,'landmarks':landmarks}
    for original in originals:
        boxes=detect_training_boxes(clips[original]['frames'],box_detector)
        crops=crop_training_clip(clips[original]['frames'],boxes)
        landmarks=annotate_pair(crops,[],landmark_detector)['landmarks']
        eligible=[crops[i] for i in eligible_crop_positions(crops)]
        images={(c['frame'],c['actor']):c['image'] for c in eligible}
        cache[original]=(boxes,images,landmarks)
        for crop in eligible:
            key=(crop['frame'],crop['actor'])
            records.append(record(original,crop,mask=None,landmarks=landmarks.get(key)))
    for position in altered:
        original=clips[position]['original']
        boxes,images,landmarks=cache[original]
        crops=crop_training_clip(clips[position]['frames'],boxes)
        for i in eligible_crop_positions(crops):
            crop=crops[i];key=(crop['frame'],crop['actor'])
            mask=difference_mask(images[key],crop['image']) if key in images else None
            records.append(record(position,crop,mask=mask,landmarks=landmarks.get(key)))
    random.Random(record_order_seed).shuffle(records)
    return records
