"""Independent scalar blend oracle, with source nonnormalized coefficients."""
import numpy as np
import pytest
from sciona.amex_pipeline import blend


def test_literal_weights_and_scalar_oracle():
    values=np.random.default_rng(117).uniform(size=(4,20))
    expected=[.3*values[0,i]+.35*values[1,i]+.15*values[2,i]+.1*values[3,i] for i in range(20)]
    np.testing.assert_array_equal(blend(*values),expected)
    assert blend(*([[1.]]*4))[0]==pytest.approx(.9)


def test_invalid_branch_probabilities():
    with pytest.raises(ValueError):blend([.1],[.2,.3],[.4],[.5])
    with pytest.raises(ValueError):blend([.1],[float('nan')],[.4],[.5])
