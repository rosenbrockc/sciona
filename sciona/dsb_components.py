"""Candidate components for the pinned ten-stage competition realization.

These functions are not catalog-approved. Arrays are supplied at runtime. Volume
reconstruction retains the source's padded spatial extent and float32 output.
Adapted from lfz/DSB2017; MIT notice: docs/licenses/DSB2017-MIT.txt.
"""
from itertools import product
import numbers

import numpy as np


def prepare_classifier_batch(volume, proposals, topk=5, crop_size=96, stride=4):
    """Source test-time descending selection with zero-filled unused slots.

    Crops are raw float32 transformed intensities, without detector normalization.
    The source computes but does not apply a pad mask; unused slots are retained.
    """
    topk = _integer(topk, 'topk')
    crop_size, stride = _integer(crop_size, 'crop_size'), _integer(stride, 'stride')
    volume, proposals = np.asarray(volume), np.asarray(proposals)
    if (volume.ndim != 4 or volume.shape[0] != 1 or min(volume.shape) < 1
            or volume.dtype.kind not in 'fiu' or proposals.ndim != 2 or proposals.shape[1] != 5
            or proposals.dtype.kind != 'f' or not np.all(np.isfinite(proposals))
            or np.any(proposals[:, 4] <= 0) or crop_size % stride):
        raise ValueError('invalid classifier volume, proposals, or crop geometry')
    chosen = proposals[:, 0].argsort()[::-1][:topk]
    crops = np.zeros((topk, 1, crop_size, crop_size, crop_size), dtype=np.float32)
    coords = np.zeros((topk, 3, crop_size//stride, crop_size//stride, crop_size//stride), dtype=np.float32)
    for slot, index in enumerate(chosen):
        crops[slot], coords[slot] = crop_classifier_proposal(volume, proposals[index, 1:],
                                                           crop_size=crop_size, stride=stride, phase='test')
    return crops, coords, chosen


def crop_classifier_proposal(volume, target, crop_size=96, stride=4, phase='test',
                             scale_enabled=False, scale_limits=(.85, 1.15),
                             radius_limits=(6., 100.), jitter_range=.15,
                             filling_value=160, random_state=None, rng=None):
    """Source proposal crop, jitter, scale and coordinates for the classifier.

    Coordinates follow the source's clamped start and padded image dimensions;
    they are not coordinates of the original unpadded volume. Python 2 integer
    crop halves are preserved. Returned image is not intensity-normalized yet.
    """
    from scipy.ndimage import zoom
    volume, target = np.asarray(volume), np.asarray(target, dtype=float)
    crop_size, stride = _integer(crop_size, 'crop_size'), _integer(stride, 'stride')
    scale_limits, radius_limits = np.asarray(scale_limits, dtype=float), np.asarray(radius_limits, dtype=float)
    if (volume.ndim != 4 or min(volume.shape) < 1 or volume.dtype.kind not in 'fiu'
            or target.shape != (4,) or not np.all(np.isfinite(target)) or target[3] <= 0
            or crop_size % stride or phase not in ['train', 'val', 'test']
            or not np.isfinite(jitter_range) or jitter_range < 0 or not np.isfinite(filling_value)):
        raise ValueError('invalid classifier crop inputs')
    for limits in [scale_limits, radius_limits]:
        if limits.shape != (2,) or not np.all(np.isfinite(limits)) or not 0 < limits[0] <= limits[1]:
            raise ValueError('scale and radius limits must be ordered positive pairs')
    if rng is not None and random_state is not None:
        raise ValueError("provide a shared RNG or a seed, not both")
    rng = np.random.RandomState(random_state) if rng is None else rng
    scaling = scale_enabled and phase == 'train'
    scale = 1.
    if scaling:
        lower = min(max(radius_limits[0]/target[3], scale_limits[0]), 1)
        upper = max(min(radius_limits[1]/target[3], scale_limits[1]), 1)
        scale = rng.rand() * (upper-lower) + lower
    size = np.full(3, int(crop_size / scale), dtype=int)
    if np.any(size < 1):
        raise ValueError('scale yields an empty source crop')
    jitter = (rng.rand(3)-.5) * target[3] * jitter_range if phase == 'train' else 0
    start = (target[:3] - size // 2 + jitter).astype(int)
    padding = [(0, 0)]
    for axis in range(3):
        left = max(0, -start[axis])
        start[axis] = max(0, start[axis])
        right = max(0, start[axis]+size[axis]-volume.shape[axis+1])
        padding.append((left, right))
    padded = np.pad(volume, padding, mode='constant', constant_values=filling_value)
    crop = padded[(slice(None),) + tuple(slice(s, s+n) for s, n in zip(start, size))]
    normalized_start = start.astype(np.float32)/np.array(padded.shape[1:])-.5
    normalized_size = size.astype(np.float32)/np.array(padded.shape[1:])
    coords = np.stack(np.meshgrid(*(np.linspace(s, s+n, crop_size//stride)
                                    for s, n in zip(normalized_start, normalized_size)), indexing='ij')).astype(np.float32)
    if scaling:
        crop = zoom(crop, [1, scale, scale, scale], order=1)
        # Nominal source scale geometry cannot overshoot: floor(N/s)*s <= N.
        # Explicit target-size trimming is defensive against interpolation drift.
        crop = crop[:, :crop_size, :crop_size, :crop_size]
        crop = np.pad(crop, [(0, 0)] + [(0, crop_size-n) for n in crop.shape[1:]],
                      mode='constant', constant_values=filling_value)
    return crop, coords


def prepare_detector_tiles(volume, side_len=144, max_stride=16, stride=4, margin=32, pad_value=170):
    """Source inference padding, global coordinates, paired tiling and scaling.

    Input is C,Z,Y,X. Coordinates span the stride-padded full volume and are
    subsequently split with the same tile ordering, never regenerated per tile.
    """
    volume = np.asarray(volume)
    stride = _integer(stride, 'stride')
    side_len = _integer(side_len, 'side_len')
    max_stride = _integer(max_stride, 'max_stride')
    margin = _integer(margin, 'margin', 0)
    if (volume.ndim != 4 or min(volume.shape) < 1 or volume.dtype.kind not in 'fiu'
            or any(value % stride for value in [side_len, max_stride, margin])
            or not np.isfinite(pad_value)):
        raise ValueError('invalid volume, padding, or detector/coordinate stride alignment')
    shape = tuple(((n + stride - 1) // stride) * stride for n in volume.shape[1:])
    padded = np.pad(volume, [(0, 0)] + [(0, p-n) for p, n in zip(shape, volume.shape[1:])],
                    mode='constant', constant_values=pad_value)
    coordinates = np.stack(np.meshgrid(*(np.linspace(-.5, .5, n // stride) for n in shape),
                                      indexing='ij')).astype(np.float32)
    tiles, grid = split_volume(padded, side_len, max_stride, margin)
    coordinate_tiles, coordinate_grid = split_volume(coordinates, side_len // stride,
                                                    max_stride // stride, margin // stride)
    if coordinate_grid != grid:
        raise ValueError('image and coordinate tile grids differ')
    return (tiles.astype(np.float32)-128)/128, coordinate_tiles, grid


def decode_proposals(output, threshold=-3., anchors=(10., 30., 60.), stride=4):
    """Decode anchor offsets/log-diameters; return proposals and grid indices.

    Source selection is a strict logit threshold. No sigmoid or implicit NMS is
    applied. Unselected extreme regression entries do not invalidate selection.
    """
    output = np.asarray(output)
    anchors = np.asarray(anchors, dtype=float)
    stride = _integer(stride, 'stride')
    if (output.ndim != 5 or output.shape[-1] != 5 or output.dtype.kind != 'f'
            or min(output.shape) < 1 or anchors.ndim != 1 or len(anchors) != output.shape[-2]
            or not np.all(np.isfinite(anchors)) or np.any(anchors <= 0)
            or not np.isfinite(threshold) or not np.all(np.isfinite(output[..., 0]))):
        raise ValueError('invalid detector grid, anchors, or threshold')
    result = output.copy()
    offset = (float(stride) - 1) / 2
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        for axis in range(3):
            grid = np.arange(offset, offset + stride * (output.shape[axis] - 1) + 1, stride)
            shape = [1, 1, 1, 1]
            shape[axis] = len(grid)
            result[..., axis + 1] = grid.reshape(shape) + output[..., axis + 1] * anchors.reshape(1, 1, 1, -1)
        result[..., 4] = np.exp(output[..., 4]) * anchors.reshape(1, 1, 1, -1)
    indices = np.where(result[..., 0] > threshold)
    selected = result[indices]
    if not np.all(np.isfinite(selected)) or np.any(selected[:, 4] <= 0):
        raise ValueError('selected proposal geometry is nonfinite or nonpositive')
    return selected, indices


def suppress_proposals(proposals, threshold=.05):
    """Source greedy cube-IoU NMS, descending logits and inclusive suppression."""
    proposals = np.asarray(proposals)
    if (proposals.ndim != 2 or proposals.shape[1] != 5 or proposals.dtype.kind != 'f'
            or not np.all(np.isfinite(proposals)) or np.any(proposals[:, 4] <= 0)
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('invalid proposals or suppression threshold')
    if not len(proposals):
        return proposals.copy()
    ordered = proposals[np.argsort(-proposals[:, 0])]
    kept = []
    for candidate in ordered:
        rejected = False
        for previous in kept:
            first, second = candidate[1:5], previous[1:5]
            start0, end0 = first[:3] - first[3]/2, first[:3] + first[3]/2
            start1, end1 = second[:3] - second[3]/2, second[:3] + second[3]/2
            overlap = [max(0, min(end0[i], end1[i]) - max(start0[i], start1[i])) for i in range(3)]
            intersection = overlap[0] * overlap[1] * overlap[2]
            union = first[3] * first[3] * first[3] + second[3] * second[3] * second[3] - intersection
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                iou = intersection / union
            if not np.isfinite(iou):
                raise ValueError('proposal geometry exceeds finite source IoU arithmetic')
            if iou >= threshold:
                rejected = True
                break
        if not rejected:
            kept.append(candidate)
    result = np.asarray(kept, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError('proposals exceed source float32 output range')
    return result


def assign_anchor_labels(input_size, target, boxes, anchors=(10., 30., 60.), stride=4,
                         phase='train', num_neg=800, negative_iou=.02,
                         positive_iou_train=.5, positive_iou_val=1., random_state=None, rng=None):
    """Source selected-target assignment with independent surrounding boxes.

    target=None denotes a negative crop. Random selection uses a local Python
    RNG; deterministic seeded parity is against the source under this runtime,
    not a claim about historical Python 2 random.sample sequences.
    """
    from sciona.dsb_anchor_mapping import LabelMapping
    size = tuple(_integer(n, 'input size') for n in input_size)
    stride = _integer(stride, 'stride')
    num_neg = _integer(num_neg, 'num_neg', 0)
    anchors = np.asarray(anchors, dtype=float)
    boxes = np.asarray(boxes, dtype=float)
    if (len(size) != 3 or any(n % stride for n in size) or phase not in ['train', 'val']
            or anchors.ndim != 1 or not len(anchors) or not np.all(np.isfinite(anchors))
            or np.any(anchors <= 0) or boxes.ndim != 2 or boxes.shape[1] != 4
            or not np.all(np.isfinite(boxes)) or np.any(boxes[:, 3] <= 0)):
        raise ValueError('invalid grid, phase, anchors, or surrounding boxes')
    for threshold in [negative_iou, positive_iou_train, positive_iou_val]:
        if not np.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError('IoU thresholds must be in (0,1]')
    if target is None:
        target = np.full(4, np.nan)
    else:
        target = np.asarray(target, dtype=float)
        if (target.shape != (4,) or not np.all(np.isfinite(target)) or target[3] <= 0
                or np.any(target[:3] < 0) or np.any(target[:3] >= size)):
            raise ValueError('selected target must have positive size and lie inside the crop')
    if rng is not None and random_state is not None:
        raise ValueError('provide a label RNG or a seed, not both')
    config = dict(stride=stride, num_neg=num_neg, th_neg=negative_iou, anchors=anchors,
                  th_pos_train=positive_iou_train, th_pos_val=positive_iou_val, random_state=random_state, rng=rng)
    try:
        return LabelMapping(config, phase)(size, target, boxes)
    except IndexError as error:
        raise ValueError('source fallback target rounds outside the anchor grid') from error


def segment_volume(volume, spacing):
    """Source segmentation sequence, returning two masks without merging them.

    The source uses square in-plane images. Mask order is component order,
    not an independently established anatomical left/right orientation.
    """
    from sciona.dsb_segmentation import binarize_per_slice, all_slice_analysis, fill_hole, two_lung_only
    volume, spacing = np.asarray(volume), np.asarray(spacing, dtype=float)
    if (volume.ndim != 3 or min(volume.shape) < 1 or volume.shape[1] != volume.shape[2]
            or volume.dtype.kind not in 'fiu' or spacing.shape != (3,)
            or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0)):
        raise ValueError('source segmentation requires a numeric volume with square slices and positive spacing')
    initial = binarize_per_slice(volume, spacing)
    flag, cut = 0, 0
    mask = initial.copy()
    while flag == 0 and cut < mask.shape[0]:
        mask, flag = all_slice_analysis(initial.copy(), spacing, cut_num=cut, vol_limit=[.68, 7.5])
        cut += 2
    first, second, _ = two_lung_only(fill_hole(mask), spacing)
    return first, second


def preprocess_volume(volume, spacing, requested_spacing=(1., 1., 1.)):
    """Connected source preprocessing from runtime array through segmentation/crop."""
    first, second = segment_volume(volume, spacing)
    return preprocess_from_masks(volume, first, second, spacing, requested_spacing)


def resample_volume(volume, spacing, requested_spacing=(1., 1., 1.), order=1):
    """Source 3D zoom with rounded output shape and effective spacing."""
    from scipy.ndimage import zoom
    volume = np.asarray(volume)
    spacing, requested_spacing = np.asarray(spacing, dtype=float), np.asarray(requested_spacing, dtype=float)
    order = _integer(order, 'order', 0)
    if (volume.ndim != 3 or min(volume.shape) < 1 or volume.dtype.kind not in 'fiu'
            or spacing.shape != (3,) or requested_spacing.shape != (3,)
            or not np.all(np.isfinite(spacing)) or not np.all(np.isfinite(requested_spacing))
            or np.any(spacing <= 0) or np.any(requested_spacing <= 0) or order > 5):
        raise ValueError('invalid 3D volume, spacing, or interpolation order')
    with np.errstate(over='ignore', invalid='ignore'):
        shape = np.round(np.array(volume.shape) * spacing / requested_spacing)
    if not np.all(np.isfinite(shape)) or np.any(shape < 1) or np.any(shape >= np.iinfo(np.intp).max):
        raise ValueError('requested spacing produces an invalid output shape')
    result = zoom(volume, shape / volume.shape, mode='nearest', order=order)
    return result, spacing * volume.shape / shape


def preprocess_from_masks(volume, left_mask, right_mask, spacing, requested_spacing=(1., 1., 1.)):
    """Compose source mask cleanup, resampling, and asymmetric bounding-box crop.

    Returns cropped uint8 volume, effective spacing, and exclusive crop bounds
    in the resampled grid. Source crop bounds use requested spacing, even where
    rounding changes effective spacing; both are retained for downstream mapping.
    Segmentation must supply the masks; an empty union is rejected explicitly.
    """
    cleaned = mask_and_remove_bone(volume, left_mask, right_mask)
    union = np.asarray(left_mask) | np.asarray(right_mask)
    if not union.any():
        raise ValueError('cannot crop an empty segmentation')
    resampled, effective_spacing = resample_volume(cleaned, spacing, requested_spacing, order=1)
    coordinates = np.where(union)
    box = np.array([[axis.min(), axis.max()] for axis in coordinates])
    box = np.floor(box * np.asarray(spacing)[:, None] / np.asarray(requested_spacing)[:, None]).astype(int)
    bounds = np.stack([np.maximum(0, box[:, 0] - 5),
                       np.minimum(resampled.shape, box[:, 1] + 10)], axis=1)
    crop = resampled[tuple(slice(int(lo), int(hi)) for lo, hi in bounds)]
    if min(crop.shape) < 1:
        raise ValueError('source crop bounds produce an empty volume')
    return crop, effective_spacing, bounds


def process_lung_mask(mask):
    """Source per-slice hull guard followed by ten 3-connected dilations."""
    from scipy.ndimage import binary_dilation, generate_binary_structure
    from skimage.morphology import convex_hull_image
    mask = np.asarray(mask)
    if mask.ndim != 3 or min(mask.shape) < 1 or mask.dtype != np.bool_:
        raise ValueError('mask must be a nonempty boolean Z,Y,X array')
    convex = mask.copy()
    for index, plane in enumerate(mask):
        if plane.any():
            hull = convex_hull_image(np.ascontiguousarray(plane))
            if hull.sum() <= 2 * plane.sum():
                convex[index] = hull
    return binary_dilation(convex, structure=generate_binary_structure(3, 1), iterations=10)


def mask_and_remove_bone(volume, left_mask, right_mask):
    """Apply source intensity transform and boundary bone replacement.

    This component consumes the two segmentation masks; it does not perform
    segmentation, spatial resampling, or cropping. The 210 threshold is on the
    transformed uint8 intensity, not on the input intensity. Inputs are retained.
    """
    volume = np.asarray(volume)
    left_mask, right_mask = np.asarray(left_mask), np.asarray(right_mask)
    if (volume.ndim != 3 or volume.dtype.kind not in 'fiu'
            or left_mask.shape != volume.shape or right_mask.shape != volume.shape):
        raise ValueError('volume and masks must have matching Z,Y,X shapes')
    left_dilated, right_dilated = process_lung_mask(left_mask), process_lung_mask(right_mask)
    dilated = left_dilated | right_dilated
    original = left_mask | right_mask
    boundary = dilated ^ original
    values = volume.astype(np.float64, copy=True)
    values[np.isnan(values)] = -2000.
    # Clip infinities and extreme values before arithmetic to avoid overflow.
    transformed = ((np.clip(values, -1200., 600.) + 1200.) / 1800. * 255.).astype(np.uint8)
    transformed[~dilated] = 170
    transformed[boundary & (transformed > 210)] = 170
    return transformed


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def split_volume(volume, side_len, max_stride, margin):
    """Split C,Z,Y,X into ordered, edge-padded cubes; return cubes and grid."""
    side_len = _integer(side_len, 'side_len')
    max_stride = _integer(max_stride, 'max_stride')
    margin = _integer(margin, 'margin', 0)
    volume = np.asarray(volume)
    if volume.ndim != 4 or min(volume.shape) < 1 or volume.dtype.kind not in 'fiu':
        raise ValueError('volume must be a nonempty numeric C,Z,Y,X array')
    if side_len <= margin or side_len % max_stride or margin % max_stride:
        raise ValueError('side must exceed margin and both must align to max_stride')
    grid = tuple((n + side_len - 1) // side_len for n in volume.shape[1:])
    padding = [(0, 0)] + [(margin, g * side_len - n + margin)
                         for n, g in zip(volume.shape[1:], grid)]
    padded = np.pad(volume, padding, mode='edge')
    tiles = []
    for index in product(*(range(g) for g in grid)):
        slices = tuple(slice(i * side_len, (i + 1) * side_len + 2 * margin) for i in index)
        tiles.append(padded[(slice(None),) + slices])
    return np.stack(tiles), grid


def combine_volume(predictions, grid, side_len, stride, margin):
    """Strip margins from T,Z,Y,X,A,O predictions and reconstruct Z,Y,X,A,O.

    Explicit grid metadata avoids the original object's inconsistent implicit
    grid state. Every tile is required; missing tiles must not masquerade as
    valid very-negative predictions.
    """
    side_len = _integer(side_len, 'side_len')
    stride = _integer(stride, 'stride')
    margin = _integer(margin, 'margin', 0)
    grid = tuple(_integer(g, 'grid entry') for g in grid)
    if len(grid) != 3 or side_len <= margin or side_len % stride or margin % stride:
        raise ValueError('invalid grid or stride alignment')
    predictions = np.asarray(predictions)
    side, border = side_len // stride, margin // stride
    extent = side + 2 * border
    if (predictions.ndim != 6 or predictions.shape[0] != np.prod(grid)
            or predictions.shape[1:4] != (extent,) * 3
            or min(predictions.shape[4:]) < 1 or predictions.dtype.kind not in 'fiu'):
        raise ValueError('predictions do not match the complete tile grid and margins')
    result = np.full(tuple(g * side for g in grid) + predictions.shape[4:], -1e6, dtype=np.float32)
    for tile, index in zip(predictions, product(*(range(g) for g in grid))):
        slices = tuple(slice(i * side, (i + 1) * side) for i in index)
        result[slices] = tile[border:border + side, border:border + side, border:border + side]
    return result


def oversample_by_size(boxes, diameter_column, thresholds=(6., 30., 40.), repeats=(1, 2, 4)):
    """Apply source strict-threshold multiplicities in input order, starting empty."""
    boxes = np.asarray(boxes)
    diameter_column = _integer(diameter_column, 'diameter_column', 0)
    thresholds = np.asarray(thresholds, dtype=float)
    repeats = tuple(_integer(r, 'repeat', 0) for r in repeats)
    if (boxes.ndim != 2 or diameter_column >= boxes.shape[1] or boxes.dtype.kind not in 'fiu'
            or thresholds.ndim != 1 or len(thresholds) != len(repeats)
            or not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0)
            or np.any(np.diff(thresholds) <= 0)):
        raise ValueError('invalid boxes or ordered thresholds')
    sizes = boxes[:, diameter_column]
    if not np.all(np.isfinite(sizes)) or np.any(sizes < 0):
        raise ValueError('sizes must be finite and nonnegative')
    counts = (sizes[:, None] > thresholds) @ np.array(repeats, dtype=np.int64)
    return np.repeat(boxes, counts, axis=0)


def sample_proposals(scores, k, temperature=1., random_state=None, rng=None):
    """Source sequential softmax sampling with a fresh probability floor per draw.

    RandomState preserves the original NumPy RNG algorithm for seeded parity.
    When k covers all candidates, their original order is retained.
    """
    scores = np.asarray(scores, dtype=float)
    k = _integer(k, 'k', 0)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError('scores must be a finite vector')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if k >= len(scores):
        return np.arange(len(scores), dtype=np.int64)
    if rng is not None and random_state is not None:
        raise ValueError("provide a shared RNG or a seed, not both")
    rng = np.random.RandomState(random_state) if rng is None else rng
    remaining = np.arange(len(scores))
    chosen = []
    for _ in range(k):
        # Ordinary inputs follow the source arithmetic. For overflow, center
        # first: negative infinity has zero softmax weight before the floor,
        # while at least one maximum retains a finite zero exponent.
        with np.errstate(over='ignore', invalid='ignore'):
            scaled = scores[remaining] / temperature
            shifted = scaled - np.max(scaled)
        if not np.all(np.isfinite(shifted)):
            values = scores[remaining]
            with np.errstate(over='ignore'):
                shifted = (values - np.max(values)) / temperature
        weights = np.exp(shifted)
        probabilities = weights / weights.sum()
        probabilities = np.maximum(probabilities, 1e-5)
        probabilities = np.asarray(probabilities / probabilities.sum(), dtype=float)
        index = int(rng.choice(len(remaining), size=1, replace=False, p=probabilities)[0])
        chosen.append(remaining[index])
        remaining = np.delete(remaining, index)
    return np.asarray(chosen, dtype=np.int64)
