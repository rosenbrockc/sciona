import numpy as np
import pytest
from sciona.flavours_evaluated_execution import execute_evaluated


def population(n, marker):
    return (np.full((n,46),marker),np.full((n,3),5.),np.full((n,3),3.),
            np.ones(n),np.ones(n),np.ones(n))


def controls():
    return dict(agreement_a=population(2,2),agreement_b=population(2,3),
                correlation=population(400,4),weights_a=np.ones(2),weights_b=np.ones(2),
                correlation_mass=np.arange(400.),quality=np.ones(4))


def test_population_routing_and_evaluation_only_metadata(monkeypatch):
    seen = []
    def trainer(training, labels, mass, query, excluded):
        seen.append(query[0][:,0])
        # Evaluation metadata cannot be passed through this trainer signature.
        return dict(query_scores=np.r_[.3,.4,0.,1.,0.,1.,np.linspace(0,1,400)],
                    oof_scores=np.array([0.,1.,0.,1.]))
    monkeypatch.setattr('sciona.flavours_evaluated_execution.execute',trainer)
    result=execute_evaluated(population(4,0),[0,1,0,1],np.ones(4),population(2,1),controls(),7)
    np.testing.assert_array_equal(seen[0],np.r_[np.ones(2),np.full(2,2),np.full(2,3),np.full(400,4)])
    np.testing.assert_array_equal(result['query_scores'],[.3,.4])
    assert result['evaluation']['ks_agreement']==0
    assert result['evaluation']['truncated_weighted_auc']==1
    assert not result['evaluation']['correlation_passed']


def test_bad_controls_fail_before_training(monkeypatch):
    monkeypatch.setattr('sciona.flavours_evaluated_execution.execute',
                        lambda *a: pytest.fail('Training must not begin'))
    c=controls();c['weights_a']=np.zeros(2)
    with pytest.raises(ValueError):
        execute_evaluated(population(4,0),[0,1,0,1],np.ones(4),population(2,1),c,7)
