import numpy as np
import pytest
from sciona.flavours_pipeline import execute
from sciona.flavours_neural import checkpoint_predictions, predict


def test_checkpoint_includes_scaler_with_constant_feature():
    rng = np.random.default_rng(67)
    raw = rng.normal(size=(9,3))
    raw[:,1] = 5
    mean, scale = np.array([2.,5.,-3.]), np.array([.5,1.,2.])
    params = [rng.normal(size=(3,8)),rng.normal(size=8),rng.normal(size=(8,2)),rng.normal(size=2)]
    np.testing.assert_array_equal(checkpoint_predictions(raw,params,mean,scale),
                                  predict((raw-mean)/scale,params))


@pytest.mark.parametrize('case', ['labels', 'mass', 'rare_class', 'query'])
def test_invalid_inputs_rejected_before_first_mass_fit(monkeypatch, case):
    def forbidden(*args, **kwargs):
        pytest.fail('Training began before all population validation completed')
    monkeypatch.setattr('sciona.flavours_pipeline.correct_mass',forbidden)
    def inputs(n):
        return (np.ones((n,46)),np.full((n,3),5.),np.full((n,3),3.),
                np.ones(n),np.ones(n),np.ones(n))
    train, query = inputs(600), inputs(20)
    labels, mass = np.arange(600)%2,np.ones(600)
    if case == 'labels': labels=labels[:-1]
    elif case == 'mass': mass[0]=np.nan
    elif case == 'rare_class': labels[:590]=0
    else: query=(np.ones((20,45)),)+query[1:]
    with pytest.raises(ValueError): execute(train,labels,mass,query,7)
