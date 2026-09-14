"""Synthetic OOF isolation and real ensemble fitting evidence."""
import numpy as np
import pytest
from sciona.tabular_ensemble_training import fit_stacker


def sample():
    return ([[float(i),None if i%3==0 else float(i%5)] for i in range(12)],
            [[f'category{i%3}'] for i in range(12)], [i%2 for i in range(12)],
            [i//4 for i in range(12)], [f'group{i//2}' for i in range(12)])


def test_complete_oof_rows_and_full_refit():
    n,c,y,f,g=sample();model=fit_stacker(n,c,y,f,g)
    assert model.oof_predictions.shape==(12,2) and np.isfinite(model.oof_predictions).all()
    assert len(model.fold_models)==3 and model.base.features.rows==12
    for fold,base in model.fold_models.items():
        valid=[i for i,v in enumerate(f) if v==fold]
        assert base.features.rows==8 and len(base.trees.estimators_)==64
        np.testing.assert_array_equal(model.oof_predictions[valid],base.predict([n[i] for i in valid],[c[i] for i in valid]))
    expected=model.stacker.predict_proba(model.base.predict([[20.,None]],[['unseen']]))[:,1]
    np.testing.assert_array_equal(model.predict([[20.,None]],[['unseen']]),expected)


def test_heldout_labels_and_features_do_not_influence_their_fold_models():
    n,c,y,f,g=sample();original=fit_stacker(n,c,y,f,g)
    mutated_n=[row[:] for row in n];mutated_y=y[:]
    for i in range(4):mutated_n[i]=[999.,888.];mutated_y[i]=1-y[i]
    changed=fit_stacker(mutated_n,c,mutated_y,f,g)
    a,b=original.fold_models[0],changed.fold_models[0]
    np.testing.assert_array_equal(a.features.medians,b.features.medians)
    np.testing.assert_array_equal(a.linear[-1].coef_,b.linear[-1].coef_)
    np.testing.assert_array_equal(a.predict(n[:4],c[:4]),b.predict(n[:4],c[:4]))


def test_exact_repeat():
    a=fit_stacker(*sample());b=fit_stacker(*sample())
    np.testing.assert_array_equal(a.oof_predictions,b.oof_predictions)
    np.testing.assert_array_equal(a.predict([[3.,1.]],[['category1']]),b.predict([[3.,1.]],[['category1']]))


@pytest.mark.parametrize('case',['groups','folds','classes','seed'])
def test_invalid_split_contract(case):
    n,c,y,f,g=sample();options={}
    if case=='groups':g[5]=g[0]
    if case=='folds':f=[0]*12
    if case=='classes':y=[0]*4+[1]*8
    if case=='seed':options['seed']=-1
    with pytest.raises(ValueError):fit_stacker(n,c,y,f,g,**options)
