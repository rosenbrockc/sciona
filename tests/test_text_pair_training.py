"""Synthetic ensemble fitting, threshold optimization and isolation evidence."""
import numpy as np
import pytest
from sklearn.metrics import f1_score
from sciona.text_pair_training import fit_ensemble,ensemble_scores,calibrate_threshold,predict_pairs,best_f1_threshold


def data():
    train=[['red bird','bird red'],['blue lake','lake blue'],['green tree','tree green'],['small cat','cat small'],
        ['red bird','wide river'],['blue lake','small cat'],['green tree','blue lake'],['wide river','tree green']]
    calibration=[['wide river','river wide'],['small cat','small kitten'],['red bird','green tree'],['lake blue','cat small']]
    query=[['green tree','green trees'],['purple cloud','small cat']]
    return train,[1,1,1,1,0,0,0,0],calibration,[1,1,0,0],query


def test_full_ensemble_calibration_prediction_and_reproducibility():
    train,y,cal,cy,query=data()
    first=fit_ensemble(train,y);second=fit_ensemble(train,y)
    result=predict_pairs(calibrate_threshold(first,cal,cy),query)
    assert result==predict_pairs(calibrate_threshold(second,cal,cy),query)
    assert len(first.forest.estimators_)==64 and len(result['matches'])==2
    assert all(0<=s<=1 for s in result['scores'])
    assert result['matches']==[int(s>=result['threshold']) for s in result['scores']]


def test_calibration_labels_cannot_change_features_or_base_models():
    train,y,cal,cy,query=data();fitted=fit_ensemble(train,y)
    vocab=dict(fitted.features.vectorizer.vocabulary_)
    components=fitted.features.svd.components_.copy()
    coefficients=fitted.linear[-1].coef_.copy()
    before=ensemble_scores(fitted,query)
    a=calibrate_threshold(fitted,cal,cy)
    b=calibrate_threshold(fitted,cal,1-np.asarray(cy))
    np.testing.assert_array_equal(before,ensemble_scores(b.fitted,query))
    np.testing.assert_array_equal(coefficients,fitted.linear[-1].coef_)
    np.testing.assert_array_equal(components,fitted.features.svd.components_)
    assert vocab==fitted.features.vectorizer.vocabulary_ and a.fitted is b.fitted


def test_threshold_matches_independent_f1_search_with_high_tie_rule():
    scores=np.array([.1,.2,.2,.6,.8]);labels=np.array([0,1,0,1,0])
    expected=max((f1_score(labels,scores>=t),t) for t in np.unique(scores))
    threshold,score=best_f1_threshold(scores,labels)
    assert (score,threshold)==expected


def test_normalized_reversed_pair_overlap_rejected():
    train,y,cal,cy,query=data();fitted=fit_ensemble(train,y)
    with pytest.raises(ValueError):calibrate_threshold(fitted,[['BIRD red!','RED bird'],cal[2]],[1,0])
    calibrated=calibrate_threshold(fitted,cal,cy)
    with pytest.raises(ValueError):predict_pairs(calibrated,[cal[0][::-1]])


@pytest.mark.parametrize('labels',[[1]*8,[0,1],[0,1,2,1,0,0,0,0],[.1]*8])
def test_invalid_training_labels(labels):
    with pytest.raises(ValueError):fit_ensemble(data()[0],labels)
