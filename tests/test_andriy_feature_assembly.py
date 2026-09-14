import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_feature_assembly import andriy_assemble_clip_features


def inputs():
    return np.zeros((16, 3, 102)), np.zeros((16, 3, 9)), np.zeros((2, 3, 9)), np.zeros((3, 180))


def test_matlab_flatten_order_log_boundaries_and_first_csp_component():
    main, ar, csp, con = inputs()
    main[:] = np.arange(16)[:, None, None]+100*np.arange(102)[None, None, :]
    ar[:] = np.arange(16)[:, None, None]+100*np.arange(9)[None, None, :]
    csp[0] = 7; csp[1] = 999; con[:] = 8
    result, valid = andriy_assemble_clip_features(main, ar, csp, con)
    expected = [np.log(1+main[c, 0, f]) if 20 <= f < 52 or 88 <= f < 95 else main[c, 0, f] for f in range(102) for c in range(16)]
    np.testing.assert_array_equal(result[0, :1632], expected)
    np.testing.assert_array_equal(result[0, 1632:1776], [ar[c, 0, f] for f in range(9) for c in range(16)])
    assert np.all(result[:, 1776:1785] == 7) and np.all(result[:, 1785:] == 8) and valid.all()


def test_imputation_precedes_log_and_does_not_mutate_inputs():
    main, ar, csp, con = inputs()
    main[0, :, 20] = [0, np.nan, 8]
    ar[1, :, 2] = [2, np.nan, 6]
    csp[0, :, 1] = [2, np.nan, 8]
    con[:, 3] = [1, np.nan, 7]
    result, valid = andriy_assemble_clip_features(main, ar, csp, con)
    assert result[1, 20*16] == np.log(5)
    assert result[1, 1632+2*16+1] == 4
    assert result[1, 1776+1] == 5 and result[1, 1785+3] == 4 and valid.all()
    assert np.isnan(main[0, 1, 20]) and np.isnan(con[1, 3])


def test_missing_series_invalidates_rows_but_negative_infinity_does_not():
    args = inputs();args[0][0, :, 20] = -1
    result, valid = andriy_assemble_clip_features(*args)
    assert np.isneginf(result[:, 320]).all() and valid.all()
    args[1][0, :, 0] = np.nan
    _, valid = andriy_assemble_clip_features(*args)
    assert not valid.any()


@pytest.mark.parametrize('bad', ['shape', 'windows', 'complex', 'inf', 'log_domain'])
def test_invalid_inputs(bad):
    args = list(inputs())
    if bad == 'shape': args[0] = args[0][:15]
    if bad == 'windows': args[3] = args[3][:2]
    if bad == 'complex': args[1] = args[1].astype(complex)
    if bad == 'inf': args[3][0, 0] = np.inf
    if bad == 'log_domain': args[0][0, 0, 20] = -2
    with pytest.raises(ValueError):
        andriy_assemble_clip_features(*args)
