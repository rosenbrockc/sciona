import numpy as np
import pytest
from sciona import otto_h2o_crossfit as module

CONTROLS=dict(hidden=[16],epochs=30,activation='Rectifier',standardize=True)


def inputs():
    y=np.tile(np.arange(9),60)
    return [np.eye(9)[y]*20,y,np.arange(540)%5,[f'r{i}' for i in range(540)],np.eye(9)*20,[f'q{i}' for i in range(9)]]


def run(args):return module.crossfit_h2o(*args,seed=12,controls=CONTROLS)


def test_actual_sixty_models_and_reference_populations(monkeypatch):
    args=inputs();original=module.fit_predict;seen=[]
    def capture(x,y,q,**kwargs):
        seen.append((x.copy(),y.copy(),q.copy()))
        return original(x,y,q,**kwargs)
    monkeypatch.setattr(module,'fit_predict',capture)
    result=run(args)
    assert len(seen)==6 and result['total_model_fits']==60
    for fold in range(5):
        np.testing.assert_array_equal(seen[fold][0],args[0][args[2]!=fold])
        np.testing.assert_array_equal(seen[fold][1],args[1][args[2]!=fold])
        np.testing.assert_array_equal(seen[fold][2],args[0][args[2]==fold])
    np.testing.assert_array_equal(seen[-1][0],args[0])
    np.testing.assert_array_equal(result['oof'].argmax(axis=1),args[1])
    np.testing.assert_array_equal(result['query'].argmax(axis=1),np.arange(9))


@pytest.mark.parametrize('problem',['overlap','missing_class','negative_query'])
def test_invalid_population_before_worker(problem,monkeypatch):
    args=inputs()
    if problem=='overlap':args[5][0]=args[3][0]
    elif problem=='missing_class':args[2][args[1]==8]=0
    else:args[4][0,0]=-1
    def forbidden(*args,**kwargs):raise AssertionError('Worker started')
    monkeypatch.setattr(module,'fit_predict',forbidden)
    with pytest.raises(ValueError):run(args)
