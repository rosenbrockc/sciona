"""Auxiliary training preparation with explicit runtime orientation and origin.

Follows training/prepare.py; see docs/licenses/DSB2017-MIT.txt. The source handles
orientation with a two-axis flip, not a general affine transform.
"""
import numpy as np
from sciona.dsb_components import _integer
from sciona.dsb_training_preprocessing import prepare_training_masks


def world_training_annotations(annotations_xyz_diameter,origin_zyx,spacing,crop_bounds,
                                volume_shape,*,flip=False,requested_spacing=(1.,1.,1.)):
    """Source absolute world displacement, optional size-coordinate flip, and crop.

    This path has no first-zero-Z sentinel: a world annotation on the origin plane
    remains valid. Empty arrays represent absence of annotations explicitly.
    """
    rows=np.asarray(annotations_xyz_diameter,dtype=float)
    origin=np.asarray(origin_zyx,dtype=float);spacing=np.asarray(spacing,dtype=float)
    resolution=np.asarray(requested_spacing,dtype=float);bounds=np.asarray(crop_bounds,dtype=float)
    shape=np.array([_integer(n,'volume dimension') for n in volume_shape])
    if (rows.ndim!=2 or rows.shape[1]!=4 or not np.all(np.isfinite(rows)) or np.any(rows[:,3]<=0)
            or origin.shape!=(3,) or spacing.shape!=(3,) or resolution.shape!=(3,) or shape.shape!=(3,)
            or bounds.shape!=(3,2) or not all(np.all(np.isfinite(a)) for a in [origin,spacing,resolution,bounds])
            or np.any(spacing<=0) or np.any(resolution<=0) or not isinstance(flip,(bool,np.bool_))):
        raise ValueError('finite world annotations and explicit spatial transform required')
    labels=[]
    for row in rows:
        position=np.absolute(row[:3][::-1]-origin)/spacing
        if flip:position[1:]=shape[1:]-position[1:]
        labels.append(np.concatenate([position,[row[3]/spacing[1]]]))
    if not labels:return np.empty((0,4),dtype=float)
    result=np.array(labels).T
    result[:3]=result[:3]*spacing[:,None]/resolution[:,None]
    result[3]=result[3]*spacing[1]/resolution[1]
    result[:3]=result[:3]-bounds[:,0,None]
    return result.T


def prepare_world_training_volume(volume,left_mask,right_mask,spacing,origin_zyx,
                                   annotations_xyz_diameter,*,flip=False,requested_spacing=(1.,1.,1.)):
    """Flip supplied volume and segmentation together, then clean and map boxes."""
    if not isinstance(flip,(bool,np.bool_)):raise ValueError('explicit Boolean flip required')
    volume=np.asarray(volume);left_mask=np.asarray(left_mask);right_mask=np.asarray(right_mask)
    if flip:
        volume=volume[:,::-1,::-1];left_mask=left_mask[:,::-1,::-1];right_mask=right_mask[:,::-1,::-1]
    processed,effective,bounds=prepare_training_masks(volume,left_mask,right_mask,spacing,requested_spacing)
    boxes=world_training_annotations(annotations_xyz_diameter,origin_zyx,spacing,bounds,volume.shape,
                                     flip=flip,requested_spacing=requested_spacing)
    return processed,boxes,effective,bounds
