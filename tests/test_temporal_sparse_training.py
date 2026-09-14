import copy
import pickle
import numpy as np
import pytest
from sciona.temporal_sparse_training import fit,clipped


def sample():
    data=[dict(entity=f'e{i%2}',time=float(i),value=float(i%2),category=f'c{i%2}',target=1.+(i%2)) for i in range(40)]
    encoder=dict(hash_size=16,lookback=8.,period=12.,max_entities=4,max_history=10,max_pending=4)
    controls=dict(chunk_size=7,seed=12,learning_rate=.005,alpha=.001,feature_clip=3.,max_prediction=20.,gap=0.)
    return data,encoder,controls


def test_one_pass_training_and_independent_metric():
    data,e,c=sample();f=fit(iter(data[:30]),encoder_controls=e,controls=c)
    assert f.training_rows==30 and f.updates_per_model==5 and len(f.models)==2
    batches=list(f.predict_batches(iter(data[30:]),labeled=True))
    assert max(len(p) for p,_ in batches)<=7
    predicted=np.concatenate([p for p,_ in batches]);target=np.concatenate([y for _,y in batches])
    assert ((predicted>=0)&(predicted<=20.)).all()
    assert f.evaluate(iter(data[30:]))==pytest.approx(dict(rows=10,rmsle=float(np.sqrt(np.mean((np.log1p(predicted)-np.log1p(target))**2)))))


def test_validation_labels_never_update_state_or_predictions():
    data,e,c=sample();f=fit(iter(data[:30]),encoder_controls=e,controls=c);before=pickle.dumps(f)
    first=np.concatenate([p for p,_ in f.predict_batches(data[30:],labeled=True)])
    altered=copy.deepcopy(data[30:])
    for row in altered:row['target']=999.
    second=np.concatenate([p for p,_ in f.predict_batches(altered,labeled=True)])
    np.testing.assert_array_equal(first,second)
    assert pickle.dumps(f)==before
    assert f.evaluate(data[30:])['rmsle']!=f.evaluate(altered)['rmsle']


def test_equal_log_ensemble_and_prediction_chunk_invariance():
    data,e,c=sample();f=fit(data[:30],encoder_controls=e,controls=c)
    queries=[{k:v for k,v in row.items() if k!='target'} for row in data[30:]]
    encoder=copy.deepcopy(f.encoder)
    expected=[]
    for x,_ in encoder.encode(queries,chunk_size=7,labeled=False):
        logs=np.mean([m.predict(clipped(x,c['feature_clip'])) for m in f.models],axis=0)
        expected.extend(np.expm1(np.clip(logs,0,np.log1p(20.))))
    first=np.concatenate([p for p,_ in f.predict_batches(queries)])
    np.testing.assert_allclose(first,expected)
    f.controls['chunk_size']=1
    np.testing.assert_allclose(first,np.concatenate([p for p,_ in f.predict_batches(queries)]))


def test_repeat_and_future_boundary():
    data,e,c=sample();a=fit(data[:30],encoder_controls=e,controls=c);b=fit(data[:30],encoder_controls=e,controls=c)
    for first,second in zip(a.models,b.models):np.testing.assert_array_equal(first.coef_,second.coef_)
    with pytest.raises(ValueError):list(a.predict_batches(data[29:],labeled=True))


def test_no_dense_conversion(monkeypatch):
    from scipy.sparse import csr_matrix
    def forbidden(*args,**kwargs):raise AssertionError('Dense conversion forbidden')
    monkeypatch.setattr(csr_matrix,'toarray',forbidden)
    data,e,c=sample();f=fit(data[:30],encoder_controls=e,controls=c)
    assert f.evaluate(data[30:])['rows']==10
