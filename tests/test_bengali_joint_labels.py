import numpy as np
import pytest
from sciona.bengali_joint_labels import encode_components, decode_joint_classes


def test_entire_joint_space_is_bijective_and_retains_eighth_consonant_state():
    classes=np.arange(14784)
    triples=decode_joint_classes(classes)
    np.testing.assert_array_equal(encode_components(triples),classes)
    np.testing.assert_array_equal(triples[[0,7,8,87,88,-1]],
                                  [[0,0,0],[0,0,7],[0,1,0],[0,10,7],[1,0,0],[167,10,7]])
    assert len(np.unique(triples,axis=0))==14784


@pytest.mark.parametrize('bad', [[[168,0,0]],[[0,11,0]],[[0,0,8]],[[-1,0,0]],[[0.,0.,0.]]])
def test_rejects_invalid_components(bad):
    with pytest.raises(ValueError):encode_components(bad)


def test_small_unsigned_inputs_do_not_overflow():
    values=np.array([[167,10,7]],dtype=np.uint8)
    np.testing.assert_array_equal(encode_components(values),[14783])


@pytest.mark.parametrize('bad', [[14784],[-1],[1.0]])
def test_rejects_invalid_joint_classes(bad):
    with pytest.raises(ValueError):decode_joint_classes(bad)
