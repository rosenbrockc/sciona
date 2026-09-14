from pathlib import Path
import numpy as np
import pytest
from sciona.otto_interactions import fit_selection,InteractionSelection

LIBRARY=str(Path(__file__).resolve().parents[1]/'.venv/otto-r-library')


def test_native_importance_selects_signal_over_constant_and_repeats():
    y=np.tile(np.arange(9),30)
    x=np.column_stack((np.eye(9)[y],np.zeros((len(y),7))))
    kwargs=dict(seed=12,ntree=64,mtry=4,nodesize=1,r_library=LIBRARY)
    selection=fit_selection(x,y,**kwargs)
    assert len(selection.selected)==13
    assert set(selection.selected[:9])==set(range(9))
    assert selection.selected[9:]==(9,10,11,12)
    assert selection==fit_selection(x,y,**kwargs)


def test_explicit_subset_matches_hand_products_and_query_isolation():
    selection=InteractionSelection(15,tuple(range(12,-1,-1)))
    q=np.arange(1,31,dtype=float).reshape(2,15)
    triples=[(0,1,2),(3,7,12)]
    actual=selection.transform(q,triples=triples)
    expected=np.column_stack((q[:,12]*q[:,11]*q[:,10],q[:,9]*q[:,5]*q[:,0]))
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual[:1],selection.transform(q[:1],triples=triples))
    assert selection.selected==tuple(range(12,-1,-1))


@pytest.mark.parametrize('triples',[[],[(0,0,1)],[(0,1,13)],[(0,1,2),(2,0,1)],[(0,1)]])
def test_invalid_subsets_rejected(triples):
    with pytest.raises(ValueError):InteractionSelection(13,tuple(range(13))).transform(np.ones((1,13)),triples=triples)


def test_product_overflow_rejected():
    with pytest.raises(ValueError,match='Nonfinite'):
        InteractionSelection(13,tuple(range(13))).transform(np.full((1,13),1e200),triples=[(0,1,2)])
