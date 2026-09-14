import numpy as np
import pytest
from sciona.physical_operator_projection import project_and_smooth

def test_hand_projection():
    # Threshold 1: max([-1,2,3]-1,0) = [0,1,2].
    np.testing.assert_allclose(project_and_smooth([[-1,2,3]],[[1,1,1]],smoothing=0),[[0,1,2]])

def test_projection_kkt_and_mass():
    rng=np.random.default_rng(4)
    y=rng.normal(size=(5,17));initial=rng.uniform(size=(5,17))
    x=project_and_smooth(y,initial,smoothing=0)
    assert (x>=0).all()
    np.testing.assert_allclose(x.sum(1),initial.sum(1),rtol=1e-12)
    for raw,projected in zip(y,x):
        active=projected>0;lagrange=(raw-projected)[active][0]
        np.testing.assert_allclose((raw-projected)[active],lagrange,atol=1e-12)
        assert (raw[~active]<=lagrange+1e-12).all()

def test_periodic_smoothing():
    np.testing.assert_allclose(project_and_smooth([[4,0,0,0]],[[1,1,1,1]],smoothing=.25),[[2,1,0,1]])

def test_zero_mass_and_idempotent_projection():
    np.testing.assert_array_equal(project_and_smooth([[9,1]],[[0,0]]),[[0,0]])
    x=np.array([[.3,.5,.2]])
    np.testing.assert_allclose(project_and_smooth(x,x,smoothing=0),x)

def test_smoothing_reduces_periodic_roughness():
    raw=np.array([[0.,4.,0.,2.,0.]])
    smooth=project_and_smooth(raw,raw,smoothing=.2)
    assert np.square(smooth-np.roll(smooth,1)).sum()<np.square(raw-np.roll(raw,1)).sum()
    np.testing.assert_allclose(smooth.sum(),raw.sum())

@pytest.mark.parametrize('smoothing',[-.1,.6,True,float('nan')])
def test_bad_smoothing(smoothing):
    with pytest.raises(ValueError):project_and_smooth([[1,2]],[[1,2]],smoothing=smoothing)

def test_negative_initial_rejected():
    with pytest.raises(ValueError):project_and_smooth([[1,2]],[[-1,2]])
