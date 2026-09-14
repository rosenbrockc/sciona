import numpy as np
import pytest
import torch
from sciona import porto_autoencoder as module


def controls():return dict(hidden=[12,6,12],feature_layers=[0,1,2],epochs=100,batch_size=32,learning_rate=.15,decay=.995,swap_probability=.15,momentum=.5)


def population():
    rng=np.random.default_rng(5);z=rng.uniform(-1,1,(128,1))
    return np.column_stack((z,z*.7,z*-.4,z*.2)).astype(np.float32)


def test_actual_denoising_learning_and_layer_extraction_oracles():
    x=population();fitted=module.fit_population(x,seed=7,controls=controls())
    assert fitted.clean_mse[-1]<fitted.clean_mse[0]*.25
    assert len(fitted.clean_mse)==101 and len(fitted.noisy_training_mse)==100
    query=x[:7][::-1].copy()
    with torch.no_grad():
        hidden=fitted.model.activations(torch.from_numpy(query))
        expected=torch.cat(hidden,dim=1).numpy()
    np.testing.assert_array_equal(fitted.transform(query),expected)
    np.testing.assert_allclose(np.mean((fitted.reconstruct(x)-x)**2),fitted.clean_mse[-1],rtol=1e-6)
    assert fitted.transform(query).shape==(7,30)
    bottleneck=module.FittedDAE(fitted.model,4,(1,),fitted.clean_mse,fitted.noisy_training_mse)
    np.testing.assert_array_equal(bottleneck.transform(query),hidden[1].numpy())


def test_clean_targets_full_donor_population_and_rng_isolation(monkeypatch):
    x=population();seen=[];clean_batches=[];original=module.swap_noise
    original_loss=torch.nn.functional.mse_loss
    def capture(values,reference,**kwargs):
        np.testing.assert_array_equal(reference,x);seen.append(len(values));clean_batches.append(values.copy())
        return original(values,reference,**kwargs)
    def loss(predicted,target,*args,**kwargs):
        if torch.is_grad_enabled():
            np.testing.assert_array_equal(target.numpy(),clean_batches.pop(0))
        return original_loss(predicted,target,*args,**kwargs)
    monkeypatch.setattr(module,'swap_noise',capture)
    monkeypatch.setattr(torch.nn.functional,'mse_loss',loss)
    torch.manual_seed(31);state=torch.random.get_rng_state().clone()
    c=controls();c['epochs']=2
    first=module.fit_population(x,seed=7,controls=c)
    assert torch.equal(state,torch.random.get_rng_state())
    second=module.fit_population(x,seed=7,controls=c)
    np.testing.assert_array_equal(first.reconstruct(x),second.reconstruct(x))
    assert sum(seen)==len(x)*4 and not clean_batches


@pytest.mark.parametrize('key,value',[('hidden',[]),('feature_layers',[1,1]),('epochs',True),('swap_probability',float('nan')),('momentum',-1)])
def test_invalid_controls(key,value):
    c=controls();c[key]=value
    with pytest.raises(ValueError):module.fit_population(population(),seed=7,controls=c)
