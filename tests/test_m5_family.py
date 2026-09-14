import numpy as np
import pandas as pd
import pytest
from sciona.m5_family import frame,pools
from sciona.m5_preprocessing import prepare
from tests.test_m5_preprocessing import fixture


@pytest.mark.parametrize('recursive',[False,True])
@pytest.mark.parametrize('pooling',['outlet','outlet_category','outlet_department'])
def test_all_family_selection_order_and_alignment(recursive,pooling):
    prepared=prepare(**fixture())
    for pool in pools(prepared,pooling):
        result,rows=frame(prepared,recursive=recursive,pooling=pooling,pool=pool,first_day=7)
        assert len(result)==len(rows) and (prepared['grid']['day'][rows]>=7).all()
        enc=[n for n in result if n.startswith('encoding_')]
        indices=((2,3,8) if recursive else (7,9)) if pooling=='outlet' else ((7,10) if pooling=='outlet_category' else (10,))
        assert enc==[f'encoding_{i}_{s}' for i in indices for s in ('mean','std')]
        assert sum(n.startswith('temporary_') for n in result)==(12 if recursive else 0)
        if recursive:assert result.columns.get_loc(enc[0])<result.columns.get_loc('lag_28')
        else:assert result.columns.get_loc(enc[0])>result.columns.get_loc('std_180')
        assert isinstance(result['role_4'].dtype,pd.CategoricalDtype)
        np.testing.assert_equal(result['price_value'],prepared['prices']['value'][rows])


def test_mismatched_encoding_order_rejects():
    prepared=prepare(**fixture());prepared['encoding_groupings']=()
    with pytest.raises(ValueError,match='grouping order'):
        frame(prepared,recursive=True,pooling='outlet',pool=(0,),first_day=0)
