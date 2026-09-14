import numpy as np
import pytest
from unittest.mock import patch
from sciona.m5_preprocessing import prepare
from sciona.m5_pooled_training import train
from tests.test_m5_preprocessing import fixture


def test_all_six_families_train_native_at_full_rounds():
    prepared=prepare(**fixture())
    models=train(prepared,249,nonrecursive_first_day=0)
    assert len(models)==10
    groups={(m.recursive,m.pooling) for m in models}
    assert len(groups)==6
    for recursive,pooling in groups:
        matching=[m for m in models if (m.recursive,m.pooling)==(recursive,pooling)]
        rows=np.concatenate([m.rows for m in matching])
        np.testing.assert_equal(np.sort(rows),np.arange(len(prepared['grid']['day'])))
        assert all(m.report['evaluated_rounds']==3000 for m in matching)
        assert all(tuple(m.model.feature_name())==m.features for m in matching)
        assert all(m.report['overlapping_rows']>0 if recursive else m.report['overlapping_rows']==0 for m in matching)


def test_pool_missing_training_target_rejects_before_fitting():
    prepared=prepare(**fixture());prepared['grid']['target'][0]=np.nan
    with patch('sciona.m5_pooled_training.fit') as fit:
        with pytest.raises(ValueError,match='Every pool'):
            train(prepared,249,nonrecursive_first_day=0)
        fit.assert_not_called()
