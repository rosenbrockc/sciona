"""Candidate connected array inference; caller supplies both model states.

No checkpoints, datasets or default prediction models are embedded. This path
does not establish model accuracy or imply approval of the full competition CDG.
"""
import numpy as np
import torch

from sciona.dsb_components import (prepare_detector_tiles, combine_volume,
    decode_proposals, suppress_proposals, prepare_classifier_batch, preprocess_volume)


def infer_preprocessed(volume, detector, classifier, *, tile_side=144, margin=32,
                       tile_batch_size=1, topk=5, crop_size=96,
                       decode_threshold=-3., confidence_threshold=-1., nms_threshold=.05):
    """Connect tiling, detection, reconstruction, selection, cropping and CaseNet.

    Caller supplies a transformed, preprocessed C,Z,Y,X volume and eval-mode
    detector/classifier with explicit states. Outputs retain padded-grid geometry.
    """
    if detector.training or classifier.training:
        raise ValueError('inference requires eval-mode models')
    if isinstance(tile_batch_size, bool) or not isinstance(tile_batch_size, int) or tile_batch_size < 1:
        raise ValueError('tile_batch_size must be a positive integer')
    if not np.isfinite(confidence_threshold):
        raise ValueError('confidence threshold must be finite')
    detector_parameter, classifier_parameter = next(detector.parameters()), next(classifier.parameters())
    tiles, coordinates, grid = prepare_detector_tiles(volume, tile_side, 16, 4, margin)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(tiles), tile_batch_size):
            x = torch.from_numpy(tiles[start:start+tile_batch_size]).to(detector_parameter)
            c = torch.from_numpy(coordinates[start:start+tile_batch_size]).to(detector_parameter)
            outputs.append(detector(x, c).cpu().numpy())
        combined = combine_volume(np.concatenate(outputs), grid, tile_side, 4, margin)
        decoded, _ = decode_proposals(combined, decode_threshold)
        # Source classifier intake uses a strict confidence filter before NMS.
        retained = suppress_proposals(decoded[decoded[:, 0] > confidence_threshold], nms_threshold)
        crops, crop_coordinates, chosen = prepare_classifier_batch(volume, retained, topk, crop_size)
        x = torch.from_numpy(crops[None]).to(classifier_parameter)
        c = torch.from_numpy(crop_coordinates[None]).to(classifier_parameter)
        _, case, each = classifier(x, c)
    return dict(case_probability=case.cpu().numpy(), proposal_probabilities=each.cpu().numpy(),
                proposals=retained, chosen_indices=chosen, tile_grid=grid)


def infer_volume(volume, spacing, detector, classifier, *, requested_spacing=(1.,1.,1.), **kwargs):
    """Connected source segmentation/preprocessing and model inference from arrays."""
    processed, effective_spacing, crop_bounds = preprocess_volume(volume, spacing, requested_spacing)
    result = infer_preprocessed(processed[None], detector, classifier, **kwargs)
    return dict(result, effective_spacing=effective_spacing, preprocessing_crop_bounds=crop_bounds)
