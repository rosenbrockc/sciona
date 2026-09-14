"""Synthetic padding direction, channel scale and feature complement contracts."""
import numpy as np
import pytest
from sciona.amex_sequence import encode_series,prediction_slots,pack_neural


def test_joint_encoding_scale_and_missing():
    r=encode_series([[[100.,np.nan],[200.,-100.]],[[50.,0.]]],[[[1.],[np.nan]],[[2.]]])
    np.testing.assert_array_equal(r['vocabulary'][0],[1,2])
    np.testing.assert_array_equal(r['sequences'][0],[[1,0,.01,0],[2,-1,0,0]])
    np.testing.assert_array_equal(r['sequences'][1],[[.5,0,0,.01]])


def test_opposite_padding_directions_and_no_sequence_complement():
    targets=prediction_slots([[.2,.7],[.9]])
    assert np.isnan(targets[0,:11]).all()
    np.testing.assert_array_equal(targets[0,-2:],[.2,.7])
    batch=pack_neural([[[.2],[.7]],[[.9]]],[[0.,.25,1.],[.5,0.,.2]])
    np.testing.assert_allclose(batch['series'][0,:2,0],[.2,.7])
    assert batch['series'].shape==(2,13,1) and np.all(batch['series'][0,2:]==0)
    np.testing.assert_array_equal(batch['mask'].sum(axis=1),[2,1])
    np.testing.assert_allclose(batch['features'][0],[0,.25,1,0,.751,.001],atol=1e-7)


def test_full_length_retained():
    p=np.linspace(0,1,13)
    np.testing.assert_array_equal(prediction_slots([p])[0],p)
    np.testing.assert_allclose(pack_neural([p[:,None]],[[0.]])['series'][0,:,0],p,atol=1e-7)


@pytest.mark.parametrize('p',[[],[.5]*14,[float('nan')],[-.1],[True]])
def test_invalid_predictions(p):
    with pytest.raises(ValueError):prediction_slots([p])


def test_invalid_packing():
    with pytest.raises(ValueError):pack_neural([[[float('nan')]]],[[0.]])
    with pytest.raises(ValueError):pack_neural([np.ones((14,1))],[[0.]])
