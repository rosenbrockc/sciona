import numpy as np
import pytest
import torch
from sciona.porto_neural import fit_predict,Classifier


def controls():return dict(hidden=[12,8],epochs=100,batch_size=32,learning_rate=.15,decay=.995,l2=.001,momentum=.5,dropout=.1,input_dropout=.05,dropout_scaling='inverted')


def test_actual_binary_learning_query_alignment_and_rng_isolation():
    rng=np.random.default_rng(8);y=np.tile([0,1],64)
    x=y[:,None]*2+rng.normal(0,.15,(128,4));q=np.array([x[y==label].mean(axis=0) for label in (0,1)])
    torch.manual_seed(31);state=torch.random.get_rng_state().clone()
    r=fit_predict(x,y,q,seed=7,controls=controls())
    assert torch.equal(state,torch.random.get_rng_state())
    assert r['training_loss'][-1]<r['training_loss'][0]*.2
    assert r['probabilities'][0]<.1 and r['probabilities'][1]>.9
    other=fit_predict(x,y,q[::-1].copy(),seed=7,controls=controls())
    np.testing.assert_array_equal(other['probabilities'],r['probabilities'][::-1])


def test_explicit_dropout_scaling():
    c=controls();model=Classifier(4,c);model.train();x=torch.ones((100,4))
    torch.manual_seed(5);inverted=model.drop(x,model.hidden_dropout)
    model.scaling='unscaled';torch.manual_seed(5);unscaled=model.drop(x,model.hidden_dropout)
    torch.testing.assert_close(unscaled,inverted*(1-c['dropout']))
    model.eval();torch.testing.assert_close(model.drop(x,model.hidden_dropout),x)


@pytest.mark.parametrize('key,value',[('dropout',1.),('dropout_scaling','unspecified'),('hidden',[]),('epochs',True)])
def test_invalid_controls(key,value):
    c=controls();c[key]=value
    with pytest.raises(ValueError):fit_predict([[0.],[1.]],[0,1],[[.5]],seed=7,controls=c)
