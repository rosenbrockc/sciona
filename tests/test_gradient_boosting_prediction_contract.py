import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sciona.atoms.ml.sklearn.ensemble.gradient_boosting.prediction import gradient_boosting_class_probabilities


@pytest.fixture
def fitted():
    X=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,0.],[0.,2.]])
    return GradientBoostingClassifier(n_estimators=3,random_state=1).fit(X,[-1,0,1,-1,0,1])


def test_prediction_returns_class_order_and_independent_label_copy(fitted):
    X=np.array([[.2,.3],[.8,.1]])
    before=X.copy()
    p,labels=gradient_boosting_class_probabilities(fitted,X)
    np.testing.assert_array_equal(p,fitted.predict_proba(X))
    np.testing.assert_array_equal(labels,fitted.classes_)
    np.testing.assert_array_equal(X,before)
    labels[0]=99
    assert 99 not in fitted.classes_


@pytest.mark.parametrize('X',[np.empty((0,2)),np.ones((2,3)),np.array([1.,2.]),[[np.nan,0.]],[[np.inf,0.]],[[True,False]],[[1j,0j]]])
def test_invalid_features_rejected(fitted,X):
    with pytest.raises(ValueError):gradient_boosting_class_probabilities(fitted,X)


def test_model_must_be_fitted_classifier():
    with pytest.raises(TypeError):gradient_boosting_class_probabilities(object(),np.ones((2,2)))
    with pytest.raises(NotFittedError):gradient_boosting_class_probabilities(GradientBoostingClassifier(),np.ones((2,2)))


def test_causal_model_role_mismatch_rejected(fitted):
    from sciona.atoms.ml.sklearn.ensemble.gradient_boosting.causal_predictions import causal_classifier_predictions
    with pytest.raises(ValueError,match='causal role'):
        causal_classifier_predictions(np.ones((2,2)),fitted,fitted,fitted,fitted,fitted)
