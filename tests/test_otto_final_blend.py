import copy
import numpy as np
import pytest
from sciona.otto_final_blend import blend


def bags():
    a=np.arange(1,10,dtype=float);a/=a.sum()
    b=a[::-1].copy();c=np.full(9,1/9)
    return [np.tile(p,(runs,2,1)) for p,runs in [(a,250),(b,600),(c,250)]]


def test_exact_formula_and_final_normalization():
    a,b,c=bags();result=blend(a,b,c)
    raw=np.array([.85*(a[0,0,i]**.65)*(b[0,0,i]**.35)+.15*c[0,0,i] for i in range(9)])
    np.testing.assert_allclose(result['raw_scores'][0],raw)
    np.testing.assert_allclose(result['probabilities'][0],raw/raw.sum())
    assert result['classes'][0]==int(np.argmax(raw))
    assert not np.isclose(raw.sum(),1.)


def test_average_runs_before_geometric_blend():
    a,b,c=bags()
    a[:125]=np.roll(a[:125],1,axis=2)
    result=blend(a,b,c)
    expected=.85*a.mean(axis=0)**.65*b.mean(axis=0)**.35+.15*c.mean(axis=0)
    np.testing.assert_allclose(result['raw_scores'],expected)


def test_zero_probabilities_and_disjoint_support():
    a,b,c=bags();a[:]=0;b[:]=0;c[:]=0
    a[:,:,0]=1;b[:,:,1]=1;c[:,:,2]=1
    result=blend(a,b,c)
    assert result['raw_scores'][0][2]==pytest.approx(.15)
    assert result['probabilities'][0][2]==1 and result['classes']==[2,2]


def test_rows_independent_and_inputs_unchanged():
    a,b,c=bags();before=[x.copy() for x in (a,b,c)]
    complete=blend(a,b,c)
    single=blend(a[:,:1],b[:,:1],c[:,:1])
    assert single['probabilities'][0]==complete['probabilities'][0]
    for x,y in zip((a,b,c),before):np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('problem',['missing_run','bad_class_count','misaligned_rows','nan','negative','unnormalized'])
def test_invalid_bags(problem):
    a,b,c=bags()
    if problem=='missing_run':a=a[:-1]
    if problem=='bad_class_count':b=b[:,:,:8]
    if problem=='misaligned_rows':c=c[:,:1]
    if problem=='nan':a[0,0,0]=np.nan
    if problem=='negative':a[0,0,0]=-.1
    if problem=='unnormalized':a[0,0,:]=1
    with pytest.raises(ValueError):blend(a,b,c)
