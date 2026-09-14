import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.ensemble_alignment import aligned_eleven_model_blend


def test_alignment_ranks_extra_ids_before_selection_and_preserves_baseline():
    scores = np.array([30., 10., 10., 50.])
    ids = np.array([8, 7, 9, 6])
    baseline = np.array([.2, -.1])
    result = aligned_eleven_model_blend([scores]*11, [ids]*11, np.array([9, 8]), baseline)
    np.testing.assert_allclose(result, [.2+1.5/4, -.1+3/4], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(baseline, [.2, -.1])
    np.testing.assert_array_equal(scores, [30., 10., 10., 50.])


@pytest.mark.parametrize('failure', ['models', 'missing', 'duplicate', 'nonfinite', 'baseline', 'output_duplicate', 'fractional_ids'])
def test_alignment_rejects_ambiguous_or_incomplete_inputs(failure):
    predictions = [np.array([.1, .2]) for _ in range(11)]
    identities = [np.array([5, 6]) for _ in range(11)]
    output, baseline = np.array([5, 6]), np.zeros(2)
    if failure == 'models': predictions.pop()
    elif failure == 'missing': identities[0] = np.array([5, 7])
    elif failure == 'duplicate': identities[0] = np.array([5, 5])
    elif failure == 'nonfinite': predictions[0][0] = np.nan
    elif failure == 'baseline': baseline = np.zeros(3)
    elif failure == 'output_duplicate': output = np.array([5, 5])
    else: identities[0] = np.array([5., 6.])
    with pytest.raises(ValueError):
        aligned_eleven_model_blend(predictions, identities, output, baseline)
