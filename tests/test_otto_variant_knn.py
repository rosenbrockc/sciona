import numpy as np
import pytest
from sciona import otto_variant_knn as module


def inputs():
    rng=np.random.default_rng(17)
    x=rng.integers(0,12,size=(180,4)).astype(float)
    y=np.arange(180)%9;f=np.arange(180)%5
    return [x,y,f,[f'r{i}' for i in range(180)],np.array([[0.,1.,2.,3.],[10.,0.,1.,0.]]),['q0','q1']]


def controls(variant):
    return dict(variant=variant,neighbors=4,metric='cityblock',ddof=0 if variant=='scaled_log' else None)


@pytest.mark.parametrize('variant,metric',[(v,m) for v in ('scaled_log','raw_zero','raw_zero_log') for m in ('cityblock','euclidean','braycurtis') if not (v=='scaled_log' and m=='braycurtis')])
def test_hand_transformation_and_sorted_frequency_oracle(variant,metric):
    x,y,_,_,q,_=inputs()
    kwargs=controls(variant);kwargs['metric']=metric
    result=module.predict(x,y,q,**kwargs)
    if variant=='scaled_log':
        a=np.log1p(x);mean=a.mean(axis=0);scale=a.std(axis=0)
        a=(a-mean)/scale;b=(np.log1p(q)-mean)/scale
    else:
        a=np.column_stack((x,x==0));b=np.column_stack((q,q==0))
        if variant=='raw_zero_log':
            a=np.column_stack((a,np.log1p(x)));b=np.column_stack((b,np.log1p(q)))
    for i,point in enumerate(b):
        def distance(j):
            delta=a[j]-point
            if metric=='euclidean':return float(np.sqrt(np.square(delta).sum()))
            numerator=float(np.abs(delta).sum())
            if metric=='cityblock':return numerator
            denominator=float(np.abs(a[j]+point).sum())
            return numerator/denominator if denominator else 0.
        ordered=sorted(range(len(a)),key=distance)
        expected=np.bincount(y[ordered[:4]],minlength=9)/4
        np.testing.assert_array_equal(result[i],expected)
    assert (result==0).any()


@pytest.mark.parametrize('variant',['scaled_log','raw_zero','raw_zero_log'])
def test_five_fold_reference_and_isolation(variant,monkeypatch):
    args=inputs();original=module.predict;references=[]
    def capture(x,y,q,**kwargs):
        references.append(x.copy())
        return original(x,y,q,**kwargs)
    monkeypatch.setattr(module,'predict',capture)
    baseline=module.crossfit(*args,**controls(variant))
    assert len(references)==6
    for fold in range(5):
        np.testing.assert_array_equal(references[fold],args[0][args[2]!=fold])
    np.testing.assert_array_equal(references[5],args[0])
    np.testing.assert_array_equal(baseline['query'],original(args[0],args[1],args[4],**controls(variant)))
    altered=inputs();held=altered[2]==0;altered[1][held]=(altered[1][held]+1)%9
    changed=module.crossfit(*altered,**controls(variant))
    np.testing.assert_array_equal(baseline['oof'][held],changed['oof'][held])
    altered=inputs();altered[4]*=100
    changed=module.crossfit(*altered,**controls(variant))
    np.testing.assert_array_equal(baseline['oof'],changed['oof'])


def test_scaler_sees_only_excluding_reference(monkeypatch):
    args=inputs();original=module.fit_scaling;seen=[]
    def capture(x,**kwargs):
        seen.append(x.copy());return original(x,**kwargs)
    monkeypatch.setattr(module,'fit_scaling',capture)
    module.crossfit(*args,**controls('scaled_log'))
    assert len(seen)==6
    for fold in range(5):np.testing.assert_array_equal(seen[fold],args[0][args[2]!=fold])
    np.testing.assert_array_equal(seen[-1],args[0])


def test_ties_preserve_reference_order_and_chunks():
    x=np.zeros((18,2));y=np.arange(18)%9;q=np.zeros((3,2))
    kwargs=dict(variant='raw_zero',neighbors=2,metric='braycurtis',ddof=None)
    a=module.predict(x,y,q,chunk_size=1,**kwargs)
    np.testing.assert_array_equal(a,module.predict(x,y,q,chunk_size=128,**kwargs))
    np.testing.assert_array_equal(a[0],np.array([.5,.5,0,0,0,0,0,0,0]))


@pytest.mark.parametrize('problem',['overlap','large_k','scaled_bray'])
def test_invalid_controls_and_reference(problem):
    args=inputs();kwargs=controls('scaled_log')
    if problem=='overlap':args[5][0]=args[3][0]
    elif problem=='large_k':kwargs['neighbors']=145
    else:kwargs['metric']='braycurtis'
    with pytest.raises(ValueError):module.crossfit(*args,**kwargs)
