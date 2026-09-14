import cv2
import numpy as np
import pytest

from sciona.cassava_fold_contract import build_plan
from sciona.cassava_population_views import align_training_views


def fixture():
    keys = [f'synthetic-{i}' for i in range(10)]
    plan = build_plan(keys, [i % 5 for i in range(10)], [i // 2 for i in range(10)])
    def rows(shape):
        result = []
        for i, key in enumerate(keys):
            ok, blob = cv2.imencode('.jpg', np.full(shape, 40 + i, np.uint8))
            assert ok
            result.append((key, blob.tobytes()))
        return result
    return plan, rows((600, 800, 3)), rows((512, 512, 3))


def test_family_views_align_to_plan_and_repr_hides_inputs():
    plan, source, prepared = fixture()
    actual = align_training_views(plan, source_rows=source[::-1], efficientnet_rows=prepared[3:] + prepared[:3])
    assert actual.source_images == tuple(row[1] for row in source)
    assert actual.efficientnet_images == tuple(row[1] for row in prepared)
    assert 'synthetic' not in repr(actual)


@pytest.mark.parametrize('kind', ['missing', 'duplicate_key', 'decoded_duplicate', 'wrong_shape', 'bad_bytes'])
def test_rejects_broken_prepared_population(kind):
    plan, source, prepared = fixture()
    if kind == 'missing': prepared.pop()
    elif kind == 'duplicate_key': prepared[-1] = prepared[0]
    elif kind == 'decoded_duplicate': prepared[-1] = (prepared[-1][0], prepared[0][1])
    elif kind == 'wrong_shape': prepared[-1] = (prepared[-1][0], source[-1][1])
    else: prepared[-1] = (prepared[-1][0], b'synthetic invalid image')
    with pytest.raises(ValueError):
        align_training_views(plan, source_rows=source, efficientnet_rows=prepared)
