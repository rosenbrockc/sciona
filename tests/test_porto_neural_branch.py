import numpy as np
import pytest
from sciona import porto_neural_branch as module


def test_real_feature_only_boundary_and_joint_unlabeled_population(monkeypatch):
    rng=np.random.default_rng(5);z=rng.uniform(-1,1,(128,1))
    x=np.column_stack((z,z*.7,-z*.4,z*.2));y=(z[:,0]>0).astype(int);q=x[:8].copy()
    dae=dict(hidden=[12,6,12],feature_layers=[0,1,2],epochs=100,batch_size=32,learning_rate=.15,decay=.995,swap_probability=.15,momentum=.5)
    neural=dict(hidden=[12,8],epochs=100,batch_size=32,learning_rate=.15,decay=.995,l2=.001,momentum=.5,dropout=.1,input_dropout=.05,dropout_scaling='inverted')
    real_encoder=module.fit_population;real_classifier=module.fit_predict;captured={}
    def encode(values,**kwargs):
        np.testing.assert_array_equal(values,np.vstack((x,q)))
        captured['encoder']=real_encoder(values,**kwargs);return captured['encoder']
    def classify(values,targets,queries,**kwargs):
        np.testing.assert_array_equal(values,captured['encoder'].transform(x))
        np.testing.assert_array_equal(queries,captured['encoder'].transform(q))
        np.testing.assert_array_equal(targets,y)
        assert values.shape[1]==30 and queries.shape[1]==30
        return real_classifier(values,targets,queries,**kwargs)
    monkeypatch.setattr(module,'fit_population',encode);monkeypatch.setattr(module,'fit_predict',classify)
    result=module.fit_branch(x,y,q,seed=7,dae_controls=dae,neural_controls=neural)
    assert result['learned_width']==30 and result['dae_population_rows']==136
    assert result['dae_epochs']==100 and len(result['training_loss'])==100
    assert result['dae_final_clean_mse']<result['dae_initial_clean_mse']*.25
    assert np.mean((result['probabilities']>.5)==y[:8])>=.875


def test_invalid_labels_rejected_before_unsupervised_fit(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('DAE started')
    monkeypatch.setattr(module,'fit_population',forbidden)
    with pytest.raises(ValueError):module.fit_branch([[0.],[1.]],[0,0],[[.5]],seed=7,dae_controls={},neural_controls={})
