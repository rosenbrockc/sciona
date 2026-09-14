import numpy as np
import pandas as pd
import pytest
from sciona.competition_dataframe_compat import compile_as_matrix_compat, dataframe_array


@pytest.mark.parametrize('call,expected', [
    ('frame.as_matrix()', [[1., 3.], [2., 4.]]),
    ("frame.as_matrix(columns=['b','a'])", [[3., 1.], [4., 2.]]),
    ("frame.as_matrix(['b'])", [[3.], [4.]]),
])
def test_reviewed_conversion_preserves_values_and_order(call, expected):
    frame = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})
    original = frame.copy(deep=True)
    code, env = compile_as_matrix_compat('result = '+call, expected_calls=1)
    env['frame'] = frame
    exec(code, env)
    np.testing.assert_array_equal(env['result'], expected)
    pd.testing.assert_frame_equal(frame, original)
    assert not hasattr(pd.DataFrame, 'as_matrix')


def test_inventory_and_unknown_signature_fail():
    with pytest.raises(ValueError, match='inventory'):
        compile_as_matrix_compat('frame.as_matrix()', expected_calls=2)
    with pytest.raises(ValueError, match='signature'):
        compile_as_matrix_compat('frame.as_matrix(copy=True)', expected_calls=1)
    with pytest.raises(ValueError, match='collides'):
        compile_as_matrix_compat('_sciona_reviewed_dataframe_array = 1', expected_calls=0)


def test_unreviewed_object_or_missing_columns_fail():
    with pytest.raises(ValueError):
        dataframe_array(np.ones((2, 2)))
    with pytest.raises(ValueError):
        dataframe_array(pd.DataFrame({'a': [1]}), ['missing'])


def test_class_method_conversion_avoids_name_mangling():
    content = "class SourceClass:\n    def run(self, frame):\n        return frame.as_matrix()\n"
    code, env = compile_as_matrix_compat(content, expected_calls=1)
    exec(code, env)
    np.testing.assert_array_equal(env['SourceClass']().run(pd.DataFrame({'a': [2.]})), [[2.]])
