"""Synthetic text-only feature invariants; no external corpora."""
import numpy as np
import pytest
from sciona.text_pair_features import PairFeatures,normalize_pairs,edit_distance


def pairs():return [['red bird','blue bird'],['blue bird','green tree'],['red bird','green tree'],['small lake','wide river']]


def test_normalization_and_hand_lexical_features():
    assert normalize_pairs([['ＲＥＤ, Bird!','red bird']])==[('red bird','red bird')]
    assert edit_distance('kitten','sitting')==3
    f=PairFeatures().fit(pairs())
    x=f.transform([['red bird','blue bird']])[0]
    assert x[0]==0 and x[1]==pytest.approx(1/3) and x[3]==pytest.approx(8/9)
    assert 0<=x[5]<=1 and x.shape==(12,)


def test_pair_orientation_symmetry():
    f=PairFeatures().fit(pairs())
    np.testing.assert_allclose(f.transform(pairs()),f.transform([p[::-1] for p in pairs()]),atol=1e-12)


def test_training_graph_removes_own_edge_and_keeps_common_neighbor():
    f=PairFeatures().fit(pairs())
    training=f.transform_training()[0];query=f.transform([pairs()[0]])[0]
    np.testing.assert_array_equal(training[8:],[1,1,1,0])
    assert query[8]==query[9]==2 and query[11]==1


def test_query_population_does_not_change_fit_or_other_predictions():
    f=PairFeatures().fit(pairs());before=dict(f.vectorizer.vocabulary_);svd=f.svd.components_.copy()
    one=f.transform([['unknown word','red bird']])
    many=f.transform([['unknown word','red bird'],['brandnew token','never seen']])
    np.testing.assert_array_equal(one[0],many[0])
    assert before==f.vectorizer.vocabulary_
    np.testing.assert_array_equal(svd,f.svd.components_)
    assert np.isfinite(many).all()


@pytest.mark.parametrize('bad',[[],[['','a']],[['!','?']],[['a']],[[None,'a']],[['a'*513,'b']]])
def test_invalid_pairs(bad):
    with pytest.raises(ValueError):normalize_pairs(bad)


def test_insufficient_lsa_population_rejected():
    with pytest.raises(ValueError):PairFeatures().fit([['same','same']])
