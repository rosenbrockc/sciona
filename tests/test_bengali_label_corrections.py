import numpy as np
import pandas as pd
import pytest

from sciona.bengali_label_corrections import corrected_joint_labels
from sciona.bengali_joint_labels import decode_joint_classes


def test_matches_source_update_alignment_with_synthetic_identities():
    identities=['synthetic-c','synthetic-a','synthetic-b']
    base=np.array([[1,2,3],[4,5,6],[7,8,0]],dtype=np.uint8)
    correction_ids=['synthetic-b','synthetic-c','synthetic-outside']
    corrections=np.array([[167,np.nan,7],[np.nan,10,0],[0,0,0]])
    before=base.copy()
    expected=pd.DataFrame(base.astype(np.int64),index=identities,columns=['a','b','c'])
    expected.update(pd.DataFrame(corrections,index=correction_ids,columns=['a','b','c']))
    actual=decode_joint_classes(corrected_joint_labels(identities,base,correction_ids,corrections))
    np.testing.assert_array_equal(actual,expected.to_numpy())
    np.testing.assert_array_equal(actual,[[1,10,0],[4,5,6],[167,8,7]])
    np.testing.assert_array_equal(base,before)


def test_empty_corrections_preserve_original_labels():
    np.testing.assert_array_equal(corrected_joint_labels(['synthetic-a'],[[167,10,7]],[],np.empty((0,3))),[14783])


@pytest.mark.parametrize('updates', [[[1.5,0,0]],[[np.inf,0,0]],[[-1,0,0]],[[168,0,0]],[[0,11,0]],[[0,0,8]]])
def test_invalid_updates_rejected_without_changing_input(updates):
    base=np.zeros((1,3),dtype=np.int64)
    with pytest.raises(ValueError,match='corrections'):
        corrected_joint_labels(['synthetic-a'],base,['synthetic-a'],updates)
    np.testing.assert_array_equal(base,[[0,0,0]])


def test_duplicate_correction_identities_rejected():
    with pytest.raises(ValueError,match='unique'):
        corrected_joint_labels(['synthetic-a'],[[0,0,0]],['synthetic-a','synthetic-a'],[[1,0,0],[2,0,0]])


def test_duplicate_original_identities_rejected():
    with pytest.raises(ValueError,match='unique'):
        corrected_joint_labels(['synthetic-a','synthetic-a'],[[0,0,0],[0,0,0]],[],np.empty((0,3)))
