from dataclasses import replace
import numpy as np
import pytest
from sciona.otto_meta_layout import AlignedBlock,assemble


def inputs():
    ids=[f'r{i}' for i in range(10)];qids=['q0','q1'];folds=np.arange(10)%5
    def block(width,value,supervised=False):
        return AlignedBlock(np.full((10,width),value),np.full((2,width),value),tuple(ids),tuple(qids),folds.copy() if supervised else None)
    first={i:block(5 if i==10 else 9,-10 if i==10 else i/100,supervised=i!=10) for i in range(1,34)}
    extra={i:block(9 if i<=5 else 1,i,supervised=i<=5) for i in range(1,8)}
    return [first,extra,block(2,99),ids,qids,folds]


def run(args):return assemble(*args,tsne_interpretation='five_features')


def test_complete_layout_preserves_all_branches_and_neural_only_raw():
    result=run(inputs())
    assert result['tree_training'].shape==(10,340)
    assert result['neural_training'].shape==(10,342)
    expected=[]
    for entry in range(1,34):expected.extend([-10]*5 if entry==10 else [entry/100]*9)
    for entry in range(1,8):expected.extend([entry]*(9 if entry<=5 else 1))
    np.testing.assert_array_equal(result['tree_query'][0],expected)
    np.testing.assert_array_equal(result['neural_query'][0],expected+[99,99])
    assert result['first_level_width']==293


@pytest.mark.parametrize('problem',['row_order','folds','missing_branch','invented_tsne_scores','nonfinite','overlap','distance_folds'])
def test_misaligned_or_incomplete_features_are_rejected(problem):
    args=inputs();first=args[0]
    if problem=='row_order':first[1]=replace(first[1],training_ids=first[1].training_ids[::-1])
    elif problem=='folds':first[1]=replace(first[1],folds=(args[5]+1)%5)
    elif problem=='missing_branch':del first[1]
    elif problem=='invented_tsne_scores':first[10]=replace(first[10],training=np.zeros((10,9)),query=np.zeros((2,9)))
    elif problem=='nonfinite':first[1].query[0,0]=np.nan
    elif problem=='distance_folds':args[1][1]=replace(args[1][1],folds=None)
    else:args[4][0]=args[3][0]
    with pytest.raises(ValueError):run(args)
