import numpy as np
import pandas as pd
import pytest
from sciona.m5u_preparation import prepare,mask_leading_zeros


def test_zero_mask_matches_source_recurrence_including_internal_missing_values():
    values=np.array([[0,2],[0,0],[3,np.nan],[0,0],[np.nan,4],[0,0]],dtype=float)
    expected=pd.DataFrame(values.T)
    expected[0]=expected[0].replace(0,np.nan)
    for col in range(1,len(values)):
        expected[col]=expected[col].where(~((expected[col]==0)&expected[col-1].isna()))
    np.testing.assert_allclose(mask_leading_zeros(values),expected.to_numpy().T,equal_nan=True)


def fixture():
    units=np.array([[0.,2.,4.,6.],[0.,4.,6.,8.],[3.,0.,8.,10.]])
    roles=np.array([[0,0,0,0,0],[0,0,0,0,1],[0,0,1,1,2],[0,0,1,1,3]])
    return units,np.full_like(units,2.),roles


def test_base_partition_uses_filtered_outlet_mean_and_preserves_revenue_zeros():
    units,prices,roles=fixture()
    out=prepare(units,prices,roles,{0:13,1:14},controller_level=13)
    np.testing.assert_equal(out['levels'],[13,13])
    np.testing.assert_equal(out['factors'],[1.,1.])
    np.testing.assert_allclose(out['scaled_history'],[[np.nan,1.],[np.nan,1.],[2.,0.]],equal_nan=True)
    np.testing.assert_allclose(out['revenue'],units[:,:2]*2)


def test_source_combined_batch_has_no_base_outlet_denominator():
    units,prices,roles=fixture()
    with pytest.raises(ValueError,match='no base-series denominator'):
        prepare(units,prices,roles,{0:13,1:14},controller_level=-1,max_level=11)


def test_all_level_batch_also_lacks_aggregate_outlet_denominator():
    units,prices,roles=fixture()
    with pytest.raises(ValueError,match='no base-series denominator'):
        prepare(units,prices,roles,{0:13,1:14},controller_level=0,max_level=15)


def test_correction_preserves_defined_source_partition_denominators():
    units,prices,roles=fixture()
    original=prepare(units,prices,roles,{0:13,1:14},controller_level=13)
    corrected=prepare(units,prices,roles,{0:13,1:14},controller_level=13,
                      normalization='retained_base_reference')
    for key in ('history','scaled_history','revenue','factors'):
        np.testing.assert_allclose(corrected[key],original[key],equal_nan=True)


def test_aggregate_reference_matches_independent_daily_base_mean():
    units,prices,roles=fixture()
    corrected=prepare(units,prices,roles,{0:13,1:14},controller_level=-1,max_level=11,
                      normalization='retained_base_reference')
    reference=pd.DataFrame(mask_leading_zeros(units)).mean(axis=1).fillna(1).to_numpy()
    np.testing.assert_allclose(corrected['scaled_history'],corrected['history']/reference[:,None])
    assert corrected['normalization']=='retained_base_reference'


def test_aggregate_results_are_invariant_to_including_base_rows_in_batch():
    units,prices,roles=fixture()
    aggregate=prepare(units,prices,roles,{0:13,1:14},controller_level=-1,max_level=11,
                      normalization='retained_base_reference')
    all_levels=prepare(units,prices,roles,{0:13,1:14},controller_level=0,max_level=15,
                       normalization='retained_base_reference')
    selected=all_levels['levels']<=11
    np.testing.assert_allclose(aggregate['scaled_history'],all_levels['scaled_history'][:,selected],equal_nan=True)


def test_correction_uses_only_contemporaneous_history_and_preserves_inputs():
    units,prices,roles=fixture();before=units.copy()
    full=prepare(units,prices,roles,{0:13,1:14},controller_level=-1,max_level=11,
                 normalization='retained_base_reference')
    prefix=prepare(units[:2],prices[:2],roles,{0:13,1:14},controller_level=-1,max_level=11,
                   normalization='retained_base_reference')
    np.testing.assert_allclose(full['scaled_history'][:2],prefix['scaled_history'],equal_nan=True)
    np.testing.assert_array_equal(units,before)


def test_concrete_outlet_reference_excludes_other_outlets():
    units=np.array([[2.,4.,20.,40.],[3.,5.,30.,50.]])
    roles=np.array([[0,0,0,0,0],[0,0,0,0,1],[1,1,0,0,0],[1,1,0,0,1]])
    out=prepare(units,np.ones_like(units),roles,{0:13},controller_level=-1,max_level=11,
                normalization='retained_base_reference')
    for outlet in (0,1):
        selected=out['roles'][:,1]==outlet
        denominator=units[:,roles[:,1]==outlet].mean(axis=1)
        np.testing.assert_allclose(out['scaled_history'][:,selected],out['history'][:,selected]/denominator[:,None])
