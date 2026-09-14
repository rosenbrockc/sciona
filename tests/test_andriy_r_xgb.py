import numpy as np
import pytest
from sciona.atoms.ml.xgboost.andriy_r_xgb import andriy_r_xgb_segment_probabilities


def inputs():
    return ([np.zeros((2,1965)) for _ in range(3)],
            [np.array([1,0]) for _ in range(3)],
            [np.zeros((19,1965)) for _ in range(3)],
            [np.ones(19,dtype=bool) for _ in range(3)])


@pytest.mark.parametrize('kind',['populations','width','training_nan','infinity','labels','incomplete','mask'])
def test_invalid_contract_before_runtime(kind):
    train,labels,pred,valid=inputs()
    if kind=='populations': train.pop()
    elif kind=='width': train[0]=np.zeros((2,1964))
    elif kind=='training_nan': train[0][0,0]=np.nan
    elif kind=='infinity': pred[0][0,0]=np.inf
    elif kind=='labels': labels[0][:]=1
    elif kind=='incomplete': pred[0]=pred[0][:-1]
    elif kind=='mask': valid[0]=valid[0].astype(int)
    with pytest.raises(ValueError):
        andriy_r_xgb_segment_probabilities(train,labels,pred,valid)
