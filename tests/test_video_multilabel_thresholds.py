import numpy as np
import pytest
from sklearn.metrics import f1_score
from sciona.video_multilabel_thresholds import calibrate,decisions


def test_independent_f1_search_and_ties():
    scores=np.array([[.1,.7],[.3,.2],[.4,.7],[.9,.4]])
    labels=[[1,1],[0,0],[0,0],[1,1]]
    actual=calibrate(scores,labels)
    for j in range(2):
        candidates=list(np.unique(scores[:,j]))+[np.nextafter(scores[:,j].max(),np.inf)]
        expected=max(candidates,key=lambda t:(f1_score(np.asarray(labels)[:,j],scores[:,j]>=t),t))
        assert actual[j]==expected
    assert actual[0]==.9 # .1 and .9 both have F1=2/3.
    assert decisions(scores,actual)==(scores>=np.array(actual)).astype(int).tolist()


def test_thresholds_label_separable():
    scores=[[.1,.2],[.9,.8],[.3,.4],[.7,.6]]
    first=calibrate(scores,[[0,0],[1,1],[0,0],[1,1]])
    second=calibrate(scores,[[0,1],[1,0],[0,1],[1,0]])
    assert first[0]==second[0] and first[1]!=second[1]

@pytest.mark.parametrize('scores,labels',[
    ([[1.1],[.2]],[[1],[0]]),
    ([[.1],[.2]],[[1],[1]]),
    ([[.1],[.2]],[[True],[0]]),
    ([[float('nan')],[.2]],[[1],[0]]),
])
def test_bad_inputs(scores,labels):
    with pytest.raises(ValueError):calibrate(scores,labels)
