import numpy as np
import torch

from sciona.dsb_components import prepare_classifier_batch
from sciona.dsb_network import DetectorNet, CaseNet
from sciona.dsb_pipeline import infer_preprocessed, infer_volume


def models(logit):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    detector, classifier = DetectorNet().eval(), CaseNet(2).eval()
    # Synthetic head supplies predictable proposal geometry; feature net executes.
    with torch.no_grad():
        detector.output[-1].weight.zero_()
        detector.output[-1].bias.zero_()
        detector.output[-1].bias[::5] = logit
    return detector, classifier


def test_classifier_batch_preserves_raw_crops_and_zero_slots():
    volume = np.full((1, 20, 20, 20), 200, dtype=np.uint8)
    proposals = np.array([[2., 10., 10., 10., 4.]])
    crops, coords, chosen = prepare_classifier_batch(volume, proposals, topk=3, crop_size=16)
    assert np.all(crops[0] == 200)  # No detector normalization.
    assert not crops[1:].any() and not coords[1:].any()
    np.testing.assert_array_equal(chosen, [0])


def test_connected_no_proposals_still_runs_zero_classifier_slots():
    detector, classifier = models(-4)
    result = infer_preprocessed(np.zeros((1, 16, 16, 16), dtype=np.uint8), detector, classifier,
                                tile_side=16, margin=0, topk=2, crop_size=16)
    assert result['proposals'].shape == (0, 5)
    assert result['chosen_indices'].size == 0
    with torch.inference_mode():
        _, expected_case, expected_each = classifier(torch.zeros(1, 2, 1, 16, 16, 16),
                                                     torch.zeros(1, 2, 3, 4, 4, 4))
    np.testing.assert_array_equal(result['case_probability'], expected_case.numpy())
    np.testing.assert_array_equal(result['proposal_probabilities'], expected_each.numpy())


def test_connected_detector_proposals_reach_classifier():
    detector, classifier = models(2)
    volume = np.full((1, 19, 17, 21), 100, dtype=np.uint8)
    result = infer_preprocessed(volume, detector, classifier, tile_side=16, margin=0,
                                tile_batch_size=2, topk=2, crop_size=16)
    assert result['tile_grid'] == (2, 2, 2)
    assert len(result['proposals']) >= 2 and len(result['chosen_indices']) == 2
    crops, coords, chosen = prepare_classifier_batch(volume, result['proposals'], 2, 16)
    with torch.inference_mode():
        _, expected, _ = classifier(torch.from_numpy(crops[None]), torch.from_numpy(coords[None]))
    np.testing.assert_array_equal(result['case_probability'], expected.numpy())


def test_array_segmentation_to_classifier_connected_execution():
    detector, classifier = models(-4)
    z, y, x = np.indices((24, 64, 64))
    volume = np.full(z.shape, 50.)
    mask = ((z-12)/9)**2 + ((y-32)/17)**2 + ((x-20)/9)**2 < 1
    volume[mask] = -900.
    result = infer_volume(volume, [6.,6.,6.], detector, classifier, requested_spacing=(6.,6.,6.),
                          tile_side=32, margin=0, crop_size=16, topk=2)
    assert result['preprocessing_crop_bounds'].shape == (3, 2)
    assert np.isfinite(result['case_probability']).all()
