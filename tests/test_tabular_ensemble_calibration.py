"""Synthetic full-stack calibration and group-isolation evidence."""
import pickle
import numpy as np
import pytest
from sciona.tabular_ensemble_training import fit_stacker
from sciona.tabular_ensemble_calibration import calibrate_stack,predict_calibrated,_logits


def training():
    return ([[float(i),float(i%3)] for i in range(12)],[[f'c{i%2}'] for i in range(12)],
        [i%2 for i in range(12)],[i//4 for i in range(12)],[f'train{i//2}' for i in range(12)])


def calibration():return [[1.,0.],[4.,1.],[7.,2.],[10.,0.]],[['c0'],['c1'],['c0'],['new']],[0,1,0,1],['cal0','cal1','cal2','cal3']


def test_complete_group_disjoint_calibration_and_prediction():
    fitted=fit_stacker(*training());calibrated=calibrate_stack(fitted,*calibration())
    result=predict_calibrated(calibrated,[[3.,None],[99.,1.]],[['c0'],['unseen']],['query0','query1'])
    assert len(result['probabilities'])==2 and all(0<=p<=1 for p in result['probabilities'])
    assert result['classes']==[int(p>=.5) for p in result['probabilities']]
    expected=calibrated.sigmoid.predict_proba(_logits(fitted.predict([[3.,None],[99.,1.]],[['c0'],['unseen']])) )[:,1]
    np.testing.assert_array_equal(result['probabilities'],expected)


def test_calibration_labels_cannot_mutate_or_refit_stack(monkeypatch):
    fitted=fit_stacker(*training());before=pickle.dumps(fitted)
    n,c,y,g=calibration()
    a=calibrate_stack(fitted,n,c,y,g)
    b=calibrate_stack(fitted,n,c,[1-v for v in y],g)
    assert before==pickle.dumps(fitted) and a.stacked is b.stacked
    assert not np.array_equal(a.sigmoid.coef_,b.sigmoid.coef_)
    def forbidden(*args,**kwargs):raise AssertionError('Existing models must not refit')
    monkeypatch.setattr(fitted.stacker,'fit',forbidden)
    monkeypatch.setattr(fitted.base.linear,'fit',forbidden)
    monkeypatch.setattr(fitted.base.trees,'fit',forbidden)
    calibrate_stack(fitted,n,c,y,g)


def test_query_batch_does_not_affect_other_predictions():
    calibrated=calibrate_stack(fit_stacker(*training()),*calibration())
    a=predict_calibrated(calibrated,[[3.,1.]],[['c0']],['query0'])
    b=predict_calibrated(calibrated,[[3.,1.],[1000.,2.]],[['c0'],['unknown']],['query0','query1'])
    assert a['probabilities'][0]==b['probabilities'][0]


@pytest.mark.parametrize('kind',['training_calibration','training_query','calibration_query'])
def test_cross_population_groups_rejected(kind):
    fitted=fit_stacker(*training());n,c,y,g=calibration()
    if kind=='training_calibration':
        g[0]='train0'
        with pytest.raises(ValueError):calibrate_stack(fitted,n,c,y,g)
    else:
        calibrated=calibrate_stack(fitted,n,c,y,g)
        group='train0' if kind=='training_query' else 'cal0'
        with pytest.raises(ValueError):predict_calibrated(calibrated,[[3.,1.]],[['c0']],[group])


def test_extreme_probability_logits_finite_and_invalid_scores_rejected():
    values=_logits([0.,.5,1.])
    assert np.isfinite(values).all() and values[1,0]==0
    with pytest.raises(ValueError):_logits([1.1])
