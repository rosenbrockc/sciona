"""Training preparation from pinned public code; runtime arrays only.

MIT source attribution: docs/licenses/DSB2017-MIT.txt. Training uses a 1.5 hull
guard, unlike the separate inference preparation's 2.0 guard.
"""
import numpy as np
from scipy.ndimage import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from sciona.dsb_components import resample_volume,segment_volume


def training_lung_mask(mask):
    mask=np.asarray(mask)
    if mask.dtype!=np.bool_ or mask.ndim!=3 or min(mask.shape)<1:
        raise ValueError('nonempty boolean ZYX mask required')
    result=mask.copy()
    for i,plane in enumerate(mask):
        if plane.any():
            hull=convex_hull_image(np.ascontiguousarray(plane))
            if hull.sum()<=1.5*plane.sum():result[i]=hull
    return binary_dilation(result,structure=generate_binary_structure(3,1),iterations=10)


def prepare_training_masks(volume,left_mask,right_mask,spacing,requested_spacing=(1.,1.,1.)):
    """Clean and crop already-oriented masks; preserve inputs and source bounds."""
    volume=np.asarray(volume);left_mask=np.asarray(left_mask);right_mask=np.asarray(right_mask)
    if volume.dtype.kind not in 'fi' or volume.shape!=left_mask.shape or volume.shape!=right_mask.shape:
        raise ValueError('aligned real volume and masks required')
    dilated=training_lung_mask(left_mask)|training_lung_mask(right_mask)
    mask=left_mask|right_mask
    if not mask.any():raise ValueError('cannot crop empty segmentation')
    # Modern NumPy disallows source Boolean subtraction. Dilation contains mask,
    # so its nonnegative difference is exactly the Boolean boundary below.
    boundary=dilated & ~mask
    values=volume.copy();values[np.isnan(values)]=-2000
    window=np.array([-1200.,600.])
    scaled=(values-window[0])/(window[1]-window[0])
    scaled=np.clip(scaled,0,1)
    cleaned=(scaled*255).astype(np.uint8)
    cleaned[~dilated]=170;cleaned[boundary & (cleaned>210)]=170
    resized,effective=resample_volume(cleaned,spacing,requested_spacing,order=1)
    axes=np.where(mask)
    box=np.array([[a.min(),a.max()] for a in axes])
    box=np.floor(box*np.asarray(spacing)[:,None]/np.asarray(requested_spacing)[:,None]).astype(int)
    bounds=np.stack((np.maximum(0,box[:,0]-5),np.minimum(resized.shape,box[:,1]+10)),axis=1)
    cropped=resized[tuple(slice(int(lo),int(hi)) for lo,hi in bounds)]
    if min(cropped.shape)<1:raise ValueError('empty training crop')
    return cropped[None],effective,bounds


def training_voxel_annotations(annotations_xyz_diameter,spacing,crop_bounds,requested_spacing=(1.,1.,1.)):
    """Source XYZ voxel annotations -> cropped ZYX boxes; diameter uses Y spacing.

    Empty and first-zero-Z source sentinels become loader-ready (0,4) arrays.
    No identifier columns enter this numerical boundary.
    """
    rows=np.asarray(annotations_xyz_diameter,dtype=float)
    spacing=np.asarray(spacing,dtype=float);resolution=np.asarray(requested_spacing,dtype=float)
    bounds=np.asarray(crop_bounds)
    if (rows.ndim!=2 or rows.shape[1]!=4 or not np.all(np.isfinite(rows))
            or spacing.shape!=(3,) or resolution.shape!=(3,) or bounds.shape!=(3,2)
            or not np.all(np.isfinite(spacing)) or not np.all(np.isfinite(resolution))
            or np.any(spacing<=0) or np.any(resolution<=0) or not np.all(np.isfinite(bounds))):
        raise ValueError('finite annotation rows and spatial transform required')
    if not len(rows) or rows[0,2]==0:return np.empty((0,4),dtype=float)
    if np.any(rows[:,3]<=0):raise ValueError('positive annotation diameters required')
    out=rows[:,[2,1,0,3]].copy()
    out[:,:3]=out[:,:3]*spacing/resolution-bounds[:,0]
    out[:,3]=out[:,3]*spacing[1]/resolution[1]
    return out


def prepare_training_volume(volume,spacing,annotations_xyz_diameter,requested_spacing=(1.,1.,1.)):
    """Segment a raw volume and connect training cleanup with voxel annotations."""
    left,right=segment_volume(volume,spacing)
    processed,effective,bounds=prepare_training_masks(volume,left,right,spacing,requested_spacing)
    boxes=training_voxel_annotations(annotations_xyz_diameter,spacing,bounds,requested_spacing)
    return processed,boxes,effective,bounds
