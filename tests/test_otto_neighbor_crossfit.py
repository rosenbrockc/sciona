import numpy as np
import pytest
from sciona import otto_neighbor_crossfit as module


def fixture():
    x=np.array([[label*10+fold+1.,fold+1.] for label in range(9) for fold in range(5)])
    y=np.repeat(np.arange(9),5);f=np.tile(np.arange(5),9)
    ids=[f'synthetic-t{i}' for i in range(45)]
    return x,y,f,ids,np.array([[1.,1.],[45.,3.]]),['synthetic-q0','synthetic-q1']


def test_independent_excluding_fold_distance_oracle():
    x,y,f,ids,q,qids=fixture()
    result=module.crossfit_distances(x,y,f,ids,q,qids,metric='cityblock')
    assert result['folds']==5 and result['oof_rows']==45 and result['query_rows']==2
    for i,point in enumerate(x):
        for label in range(9):
            distances=sorted(sum(abs(point-ref)) for ref in x[(f!=f[i])&(y==label)])
            for col,k in enumerate((1,2,4)):
                assert result['oof'][i,col,label]==pytest.approx(sum(distances[:k]))
        assert result['oof'][i,0,y[i]]>0
    assert result['query'][0,0,0]==0


def test_heldout_labels_cannot_change_own_fold_features():
    values=list(fixture());first=module.crossfit_distances(*values,metric='euclidean')
    changed=values[1].copy();held=values[2]==2
    changed[held]=(changed[held]+1)%9;values[1]=changed
    second=module.crossfit_distances(*values,metric='euclidean')
    np.testing.assert_array_equal(first['oof'][held],second['oof'][held])


def test_query_population_does_not_affect_oof():
    values=list(fixture());first=module.crossfit_distances(*values,metric='euclidean')
    values[4]=np.array([[999.,999.]]);values[5]=['synthetic-other']
    second=module.crossfit_distances(*values,metric='euclidean')
    np.testing.assert_array_equal(first['oof'],second['oof'])


def test_five_excluding_references_then_full_training_query(monkeypatch):
    real=module.class_distances;calls=[]
    def capture(x,y,q,**kwargs):
        calls.append((len(x),len(q)))
        return real(x,y,q,**kwargs)
    monkeypatch.setattr(module,'class_distances',capture)
    module.crossfit_distances(*fixture(),metric='euclidean')
    assert calls==[(36,9)]*5+[(45,2)]


@pytest.mark.parametrize('problem',['overlap','duplicate','missing_fold','deficient_class'])
def test_invalid_split_rejected_before_distances(problem,monkeypatch):
    values=list(fixture())
    if problem=='overlap':values[5][0]=values[3][0]
    if problem=='duplicate':values[3][0]=values[3][1]
    if problem=='missing_fold':values[2][values[2]==4]=3
    if problem=='deficient_class':values[1][0]=1
    def forbidden(*args,**kwargs):raise AssertionError('Distance calculation started')
    monkeypatch.setattr(module,'class_distances',forbidden)
    with pytest.raises(ValueError):module.crossfit_distances(*values,metric='euclidean')
