import copy
import numpy as np
import pytest
import torch
from sciona.biosignal_sequence_training import fit,features,Scaling,window_weights


def sample():
    labels=[0,1,0,1]
    signals=[[(np.sin(2*np.pi*(4 if y==0 else 12)*np.arange(64+16*i)/64+.2*i)).tolist()] for i,y in enumerate(labels)]
    controls=dict(sample_rate=64.,window_size=32,stride=16,n_fft=16,hop=8,seed=12,width=4,epochs=40,batch_size=3,learning_rate=.02,max_frequency=1,max_time=1)
    return signals,labels,controls


def test_hand_recording_class_weights():
    weights=window_weights([0,0,1],[1,3,2])
    np.testing.assert_allclose(weights,[.25,1/12,1/12,1/12,.25,.25])
    assert weights.sum()==pytest.approx(1.)


def test_scaling_equal_recording_contribution():
    wave=np.array([[[0.,0.]],[[10.,10.]],[[10.,10.]],[[10.,10.]]])
    spectral=wave[:,:,:,None]
    scaling=Scaling.fit(wave,spectral,[1,3])
    assert scaling.wave_mean.item()==pytest.approx(5,abs=1e-12) and scaling.wave_scale.item()==pytest.approx(5,abs=1e-12)
    assert scaling.spectral_mean.item()==pytest.approx(5,abs=1e-12) and scaling.spectral_scale.item()==pytest.approx(5,abs=1e-12)


def test_augmented_training_and_recording_prediction():
    signals,labels,c=sample();state=torch.random.get_rng_state().clone();f=fit(signals,labels,c)
    assert torch.equal(state,torch.random.get_rng_state())
    assert len(f.history)==41 and f.history[-1]<f.history[0]*.25
    scores,counts,tails=f.predict(signals)
    assert counts==[3,4,5,6] and tails==[0]*4
    assert [int(v>=.5) for v in scores]==labels
    wave,spectral,_,_=features(signals,c);x,s=f.scaling.transform(wave,spectral)
    with torch.no_grad():values=torch.sigmoid(f.network(x,s)).numpy()
    offset=0
    for score,n in zip(scores,counts):
        assert score==pytest.approx(values[offset:offset+n].mean());offset+=n


def test_query_isolation_and_frozen_scaling():
    signals,labels,c=sample();c['epochs']=2;f=fit(signals,labels,c)
    before=copy.deepcopy(f.scaling)
    one=f.predict(signals[:1])[0]
    batch=f.predict([signals[0],[[999.]*64]])[0]
    np.testing.assert_allclose(one,batch[:1])
    for name in vars(before):np.testing.assert_array_equal(getattr(before,name),getattr(f.scaling,name))


def test_deterministic_repeat():
    signals,labels,c=sample();c['epochs']=2
    first=fit(signals,labels,c);second=fit(signals,labels,c)
    assert first.history==second.history
    np.testing.assert_array_equal(first.predict(signals)[0],second.predict(signals)[0])
