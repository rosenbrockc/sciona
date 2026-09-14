import numpy as np
import pytest
from sciona import otto_first_level as module
from sciona.otto_tsne import PopulationEmbedding


@pytest.mark.parametrize('problem',['missing_controls','insufficient_neighbors','overlap','missing_class','bad_seed'])
def test_invalid_complete_bank_request_rejected_before_producer(problem,monkeypatch):
    x=np.ones((1350,13));y=np.arange(1350)%9;folds=np.arange(1350)%5
    ids=[f'r{i}' for i in range(1350)];q=np.ones((2,13));qids=['q0','q1'];seed=12
    embedding=PopulationEmbedding(tuple(ids+qids),np.zeros((1352,3)),0.)
    controls={i:{} for i in range(1,34)}
    if problem=='missing_controls':del controls[33]
    elif problem=='insufficient_neighbors':x=x[:1200];y=y[:1200];folds=folds[:1200];ids=ids[:1200]
    elif problem=='overlap':qids[0]=ids[0]
    elif problem=='missing_class':folds[y==8]=0
    else:seed=-1
    def forbidden(*args,**kwargs):raise AssertionError('Producer imported')
    monkeypatch.setattr(module.importlib,'import_module',forbidden)
    with pytest.raises(ValueError):module.build(x,y,folds,ids,q,qids,embedding=embedding,seed=seed,controls=controls)
