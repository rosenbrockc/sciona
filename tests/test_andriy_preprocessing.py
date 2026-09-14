import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.andriy_preprocessing import andriy_documented_preprocess_clip, andriy_csp_training_classes, witness_andriy_csp_training_classes


def test_preprocess_preserves_inclusive_endpoint_and_source_window_count():
    fs=128
    x=np.zeros((16,600*fs))
    high,low=andriy_documented_preprocess_clip(x,fs)
    assert high.shape==(16,151554) and low.shape==(16,75777)
    assert high.shape[1]//7680==low.shape[1]//3840==19
    np.testing.assert_array_equal(high,0)
    np.testing.assert_array_equal(low,0)
    np.testing.assert_array_equal(x,0)


def test_selection_skips_first_two_mid_sequences_and_flat_channels():
    rng=np.random.default_rng(681)
    candidates=[rng.normal(size=(16,3840+i)) for i in range(8)]
    candidates[5][0]=0
    originals=[x.copy() for x in candidates]
    late,early=andriy_csp_training_classes(candidates,np.array([6,1,1,6,3,6,6,1]))
    np.testing.assert_array_equal(late,np.concatenate([candidates[3],candidates[6]],axis=1))
    np.testing.assert_array_equal(early,np.concatenate([candidates[2],candidates[7]],axis=1))
    for x,original in zip(candidates,originals): np.testing.assert_array_equal(x,original)


def test_missing_selected_class_fails_and_symbolic_witness_is_valid():
    x=np.tile(np.arange(3840.),(16,1))
    with pytest.raises(ValueError,match='both'):
        andriy_csp_training_classes([x]*4,np.array([1,6,1,1]))
    late,early=witness_andriy_csp_training_classes(None,None)
    assert late.shape==('16','sequence6_samples') and early.shape==('16','sequence1_samples')


@pytest.mark.parametrize('shape,fs',[((16,1000),128),((15,76800),128),((16,76800),True),((16,76800),120)])
def test_invalid_preprocessing(shape,fs):
    with pytest.raises(ValueError):
        andriy_documented_preprocess_clip(np.ones(shape),fs)


def test_invalid_sequence_values():
    x=np.tile(np.arange(3840.),(16,1))
    with pytest.raises(ValueError):
        andriy_csp_training_classes([x]*4,np.array([1,6,0,6]))
