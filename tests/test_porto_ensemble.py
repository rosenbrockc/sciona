import copy
import numpy as np
import pytest
from sciona import porto_ensemble as module
from sciona.porto_preparation import prepare_populations


def fixture():
    rng=np.random.default_rng(5);z=rng.uniform(-1,1,(96,1));a=np.linspace(-.9,.9,8)[:,None]
    x=np.column_stack((z,z*.7,-z*.4,(z[:,0]>0).astype(float)))
    q=np.column_stack((a,a*.7,-a*.4,(a[:,0]>0).astype(float)))
    prep=dict(dropped_columns=[],categorical_columns=[3],binary_columns=[])
    models=[]
    for i,width in enumerate((12,16,10,14,18)):
        dae=dict(hidden=[width,width//2,width],feature_layers=[0,1,2] if i%2==0 else [1],
            epochs=60,batch_size=32,learning_rate=.15,decay=.995,swap_probability=.07,momentum=.5)
        neural=dict(hidden=[12,8],epochs=60,batch_size=32,learning_rate=.15,decay=.995,l2=.001,momentum=.5,
            dropout=.1,input_dropout=.05,dropout_scaling='inverted')
        models.append(dict(seed=7+i,dae_controls=dae,neural_controls=neural))
    tree=dict(seed=7,controls=dict(rounds=12,num_leaves=3,learning_rate=.2,min_data_in_leaf=2,
        feature_fraction=1.,bagging_fraction=1.,bagging_freq=0,lambda_l2=0.))
    return x,(z[:,0]>0).astype(int),q,dict(preparation=prep,neural_models=models,tree=tree)


def test_complete_native_ensemble_and_feature_routing(monkeypatch):
    x,y,q,options=fixture();p=prepare_populations(x,q,**options['preparation']);seen=[]
    real_branch=module.fit_branch;real_tree=module.fit_tree
    def branch(a,b,c,**kwargs):
        np.testing.assert_array_equal(a,p['dae_training']);np.testing.assert_array_equal(c,p['dae_query']);np.testing.assert_array_equal(b,y)
        seen.append('neural');return real_branch(a,b,c,**kwargs)
    def tree(a,b,c,**kwargs):
        np.testing.assert_array_equal(a,p['tree_training']);np.testing.assert_array_equal(c,p['tree_query']);np.testing.assert_array_equal(b,y)
        seen.append('tree');return real_tree(a,b,c,**kwargs)
    monkeypatch.setattr(module,'fit_branch',branch);monkeypatch.setattr(module,'fit_tree',tree)
    result=module.fit(x,y,q,**options)
    assert seen==['neural']*5+['tree'] and result['models']==6 and result['dae_models']==5
    expected=np.array([(sum(float(row[i]) for row in result['neural_probabilities'])+float(result['tree_probabilities'][i]))/6 for i in range(len(q))])
    np.testing.assert_allclose(result['probabilities'],expected,rtol=0,atol=1e-15)
    assert result['neural_probabilities'].shape==(5,8)
    assert [r['learned_width'] for r in result['neural_records']]==[30,8,25,7,45]
    assert np.mean((result['probabilities']>.5)==(q[:,0]>0))>=.875


@pytest.mark.parametrize('problem',['missing','duplicate','labels','seed'])
def test_invalid_ensemble_precedes_preparation(problem,monkeypatch):
    x,y,q,options=fixture()
    if problem=='missing':options['neural_models'].pop()
    elif problem=='duplicate':options['neural_models'][1]=copy.deepcopy(options['neural_models'][0])
    elif problem=='labels':y[:]=0
    else:options['tree']['seed']=True
    def forbidden(*args,**kwargs):raise AssertionError('Preparation started')
    monkeypatch.setattr(module,'prepare_populations',forbidden)
    with pytest.raises(ValueError):module.fit(x,y,q,**options)
