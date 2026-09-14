import math
import pytest
from sciona.physics_ingest.first_wave_dynamics_execution import execute_dynamics


def tree(op,*args): return {'op':op,'args':list(args)}


@pytest.mark.parametrize('mass',[.25,1,7])
@pytest.mark.parametrize('position,acceleration',[
    (0,lambda t:0),('t',lambda t:0),
    (tree('add',tree('mul',3,tree('pow','t',2)),tree('mul',-2,'t'),4),lambda t:6),
    (tree('pow','t',4),lambda t:12*t*t),
    (tree('sin','t'),lambda t:-math.sin(t)),
    (tree('exp','t'),lambda t:math.exp(t)),
    (tree('log','t'),lambda t:-1/t**2),
    (tree('pow','t',.5),lambda t:-.25*t**(-1.5))])
def test_analytic_acceleration_and_force(mass,position,acceleration):
    times=[.2,.8,1.,2.]
    result=execute_dynamics(dict(version=1,mass=mass,position=position,times=times))
    for t,row in zip(times,result['samples']):
        assert row[1]==pytest.approx(acceleration(t),rel=1e-12,abs=1e-12)
        assert row[2]==pytest.approx(mass*acceleration(t),rel=1e-12,abs=1e-12)


def test_arbitrary_position_preserves_formal_derivative():
    result=execute_dynamics(dict(version=1,mass=2,position=tree('position'),times=[]))
    assert "Derivative(Function('x')(Symbol('t', real=True))" in result['acceleration']
    assert result['acceleration']==result['acceleration_from_force']
    assert result['samples']==[]


@pytest.mark.parametrize('change',[
    {'mass':0},{'mass':-1},{'mass':True},{'mass':float('inf')},
    {'position':"__import__('os')"},{'position':tree('unknown','t')},
    {'position':tree('log','t'),'times':[-1]},
    {'position':tree('pow','t',-1),'times':[0]},
    {'position':tree('position'),'times':[1]}, {'times':[True]}, {'version':True}])
def test_rejects_invalid_or_undefined_inputs(change):
    payload=dict(version=1,mass=1,position='t',times=[1]); payload.update(change)
    with pytest.raises(ValueError): execute_dynamics(payload)
