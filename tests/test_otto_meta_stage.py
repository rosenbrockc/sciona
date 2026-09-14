"""Selection/refit boundary checks; native learner evidence is separate."""
import numpy as np
import pytest
from sciona import otto_meta_stage as stage


def inputs():
    tree=np.arange(144,dtype=float).reshape(36,4)
    neural=np.column_stack((tree,np.ones((36,2))))
    layout=dict(tree_training=tree,tree_query=tree[:9]+1000,
                neural_training=neural,neural_query=neural[:9]+2000)
    return layout,np.tile(np.arange(9),4),np.arange(36)%4,[f'r{i}' for i in range(36)]


def test_selected_controls_refit_full_reference_and_correct_feature_view(monkeypatch):
    layout,y,folds,ids=inputs()
    choices={family:[dict(choice=0),dict(choice=1)] for family in stage.FAMILIES}
    selected={'xgboost':1,'lasagne_neural':0,'adaboost_extratrees':1}
    calls=[]
    def tune(x,labels,partition,identities,*,family,seed,candidates):
        prefix='neural' if family=='lasagne_neural' else 'tree'
        np.testing.assert_array_equal(x,layout[prefix+'_training'])
        np.testing.assert_array_equal(labels,y)
        np.testing.assert_array_equal(partition,folds)
        assert identities==ids and candidates==choices[family] and seed==12
        return dict(selected_index=selected[family],log_losses=[.5,.4],
                    models_per_candidate=4*(600 if family=='lasagne_neural' else 250))
    def learner(family):
        runs=600 if family=='lasagne_neural' else 250
        def fit(x,labels,q,*,seed,controls):
            prefix='neural' if family=='lasagne_neural' else 'tree'
            np.testing.assert_array_equal(x,layout[prefix+'_training'])
            np.testing.assert_array_equal(q,layout[prefix+'_query'])
            np.testing.assert_array_equal(labels,y)
            assert controls==choices[family][selected[family]] and seed==12
            calls.append(family)
            return np.tile(np.eye(9)[None,:,:],(runs,1,1))
        return fit,runs
    monkeypatch.setattr(stage,'tune',tune)
    monkeypatch.setattr(stage,'_learner',learner)
    result=stage.fit(layout,y,folds,ids,seed=12,candidates=choices)
    assert calls==list(stage.FAMILIES)
    np.testing.assert_array_equal(result['blend']['classes'],np.arange(9))
    assert sum(item['refit_models'] for item in result['selection'].values())==1100
    assert sum(item['tuning_model_fits'] for item in result['selection'].values())==8800


@pytest.mark.parametrize('problem',['query_rows','query_width','nan','missing_family'])
def test_invalid_layout_rejected_before_training(problem,monkeypatch):
    layout,y,folds,ids=inputs()
    choices={family:[{}] for family in stage.FAMILIES}
    if problem=='query_rows':layout['neural_query']=layout['neural_query'][:-1]
    elif problem=='query_width':layout['neural_query']=layout['neural_query'][:,:-1]
    elif problem=='nan':layout['neural_query'][0,0]=np.nan
    else:del choices['lasagne_neural']
    def forbidden(*args,**kwargs):raise AssertionError('Training started')
    monkeypatch.setattr(stage,'tune',forbidden)
    with pytest.raises(ValueError):stage.fit(layout,y,folds,ids,seed=12,candidates=choices)
