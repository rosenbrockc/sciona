"""Synthetic batch-level semantic checks, with no competition records."""
import copy
import numpy as np
import pytest
from sciona.santander_augmentation import shuffle_feature_groups


def example():
    raw = np.arange(48).reshape(12, 4).astype(float)
    return (raw.astype(int) % 6, raw, raw * 7 + 3, np.array([0, 1] * 6))


def run(values, seed=42, training=True):
    return shuffle_feature_groups(*values, rng=np.random.default_rng(seed), training=training)


def test_paired_triplets_and_class_conditional_marginals():
    inputs = example()
    before = tuple(x.copy() for x in inputs)
    cat, raw, substitute, labels = run(inputs)
    np.testing.assert_array_equal(cat, raw.astype(int) % 6)
    np.testing.assert_array_equal(substitute, raw * 7 + 3)
    np.testing.assert_array_equal(labels, [1]*6+[0]*6)
    for label in (0, 1):
        for col in range(4):
            np.testing.assert_array_equal(np.sort(raw[labels == label, col]),
                np.sort(inputs[1][inputs[3] == label, col]))
    # Columns must not share a single row permutation.
    assert any(not np.array_equal(raw[:, 0] // 4, raw[:, col] // 4) for col in range(1, 4))
    for old, current in zip(before, inputs):
        np.testing.assert_array_equal(old, current)


def test_reproducibility_and_rng_advancement():
    for a, b in zip(run(example()), run(example())):
        np.testing.assert_array_equal(a, b)
    rng = np.random.default_rng(42)
    first = shuffle_feature_groups(*example(), rng=rng, training=True)
    second = shuffle_feature_groups(*example(), rng=rng, training=True)
    assert not np.array_equal(first[1], second[1])


def test_evaluation_preserves_order_and_rng_state_and_copies():
    inputs = example()
    rng = np.random.default_rng(42)
    state = copy.deepcopy(rng.bit_generator.state)
    output = shuffle_feature_groups(*inputs, rng=rng, training=False)
    assert rng.bit_generator.state == state
    for source, result in zip(inputs, output):
        np.testing.assert_array_equal(source, result)
        assert not np.shares_memory(source, result)


@pytest.mark.parametrize('size,label', [(1,0), (1,1), (6,0), (6,1)])
def test_single_class_and_singleton_batches(size, label):
    values = list(example())
    values = [v[:size] for v in values]
    values[3] = np.full(size, label)
    output = run(values)
    np.testing.assert_array_equal(output[3], values[3])
    np.testing.assert_array_equal(output[2], output[1]*7+3)


@pytest.mark.parametrize('case', ['shape','category','label','nonfinite','dtype','training','rng'])
def test_invalid_inputs_fail_before_randomness(case):
    values = list(example())
    rng = np.random.default_rng(42)
    state = copy.deepcopy(rng.bit_generator.state)
    kwargs = dict(rng=rng, training=True)
    if case == 'shape': values[1] = values[1][:-1]
    if case == 'category': values[0][0,0] = 6
    if case == 'label': values[3][0] = 2
    if case == 'nonfinite': values[2][0,0] = np.nan
    if case == 'dtype': values[0] = values[0].astype(float)
    if case == 'training': kwargs['training'] = 1
    if case == 'rng': kwargs['rng'] = None
    with pytest.raises(ValueError):
        shuffle_feature_groups(*values, **kwargs)
    assert rng.bit_generator.state == state
