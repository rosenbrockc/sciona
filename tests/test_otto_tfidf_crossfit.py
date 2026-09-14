import numpy as np
import pytest
from sciona import otto_tfidf_crossfit as module


def fixture():
    y=np.repeat(np.arange(9),5);f=np.tile(np.arange(5),9)
    x=np.zeros((45,4))
    for i in range(45):x[i]=[1+i%7,i%3,1+(i%2),int(f[i]==0)]
    return x,y,f,[f'synthetic-t{i}' for i in range(45)],np.array([[2.,1,0,1]]),['synthetic-q']


def run(values):
    return module.crossfit_tfidf_distances(*values,metric='euclidean',smooth_idf=True,sublinear_tf=False,norm='l2')


def test_excluding_fold_idf_and_distances_independently():
    x,y,f,ids,q,qids=fixture();result=run((x,y,f,ids,q,qids))
    for fold in range(5):
        ref=x[f!=fold];target=y[f!=fold]
        idf=np.log((len(ref)+1)/((ref>0).sum(axis=0)+1))+1
        def transform(v):
            w=v*idf
            return w/np.sqrt((w*w).sum(axis=1,keepdims=True))
        reference=transform(ref)
        for i in np.flatnonzero(f==fold):
            point=transform(x[i:i+1])[0]
            for label in range(9):
                expected=min(float(np.sqrt(sum((point-r)**2))) for r in reference[target==label])
                assert result['oof'][i,label]==pytest.approx(expected,abs=1e-15)
    assert result['oof'].shape==(45,9) and result['query'].shape==(1,9)


def test_reference_membership_and_query_independence(monkeypatch):
    real=module.fit_weighting;seen=[]
    def capture(reference,**kwargs):
        seen.append(reference.copy());return real(reference,**kwargs)
    monkeypatch.setattr(module,'fit_weighting',capture)
    values=list(fixture());a=run(values)
    x,_,f,*_=values
    for fold in range(5):np.testing.assert_array_equal(seen[fold],x[f!=fold])
    np.testing.assert_array_equal(seen[5],x)
    values[4]=np.array([[99.,0,0,0]])
    b=run(values)
    np.testing.assert_array_equal(a['oof'],b['oof'])


def test_heldout_labels_isolated():
    values=list(fixture());a=run(values);held=values[2]==0
    values[1]=values[1].copy();values[1][held]=(values[1][held]+1)%9
    b=run(values)
    np.testing.assert_array_equal(a['oof'][held],b['oof'][held])


def test_overlap_rejected_before_weighting(monkeypatch):
    values=list(fixture());values[5][0]=values[3][0]
    def forbidden(*args,**kwargs):raise AssertionError('Weighting started')
    monkeypatch.setattr(module,'fit_weighting',forbidden)
    with pytest.raises(ValueError):run(values)
