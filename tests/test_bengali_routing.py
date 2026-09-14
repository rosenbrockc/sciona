import numpy as np
import pytest
from sciona.bengali_routing import route_predictions


def test_routes_unseen_then_falls_back_for_seen_class():
    membership=np.zeros(14784,dtype=bool);membership[[1,7]]=True
    actual=route_predictions([.2,.2,.8,.5],[1,1,7,7],[14783,7,14783,14783],seen_class_mask=membership,threshold=.5)
    np.testing.assert_array_equal(actual,[14783,1,7,7])


@pytest.mark.parametrize('confidence',[[float('nan')],[-.1],[1.1]])
def test_invalid_confidence_rejected(confidence):
    with pytest.raises(ValueError):
        route_predictions(confidence,[1],[2],seen_class_mask=np.zeros(14784,dtype=bool),threshold=.5)


def test_alignment_and_class_bounds_rejected():
    membership=np.zeros(14784,dtype=bool)
    for predictions in [[14784],[-1],[1,2],[1.5]]:
        with pytest.raises(ValueError):
            route_predictions([.1],[1],predictions,seen_class_mask=membership,threshold=.5)
