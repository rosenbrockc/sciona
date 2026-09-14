import numpy as np
import pytest

from sciona.cassava_fold_contract import FoldPlan, build_plan, members, average_folds


def population():
    return [f'synthetic-{i}' for i in range(25)], [i % 5 for i in range(25)], [i // 5 for i in range(25)]


def test_all_folds_are_disjoint_complements_and_cover_validation_once():
    plan = build_plan(*population())
    held_out = []
    for fold in range(5):
        train, valid = members(plan, fold)
        assert not set(train) & set(valid)
        assert set(train) | set(valid) == set(range(25))
        held_out.extend(valid)
    assert sorted(held_out) == list(range(25))
    assert 'synthetic-' not in repr(plan)


@pytest.mark.parametrize('kind', ['duplicate', 'missing_fold', 'bad_label', 'boolean', 'missing_class', 'unaligned'])
def test_rejects_invalid_population(kind):
    keys, labels, folds = population()
    if kind == 'duplicate': keys[1] = keys[0]
    if kind == 'missing_fold': folds = [min(x, 3) for x in folds]
    if kind == 'bad_label': labels[0] = 5
    if kind == 'boolean': folds[0] = False
    if kind == 'missing_class': labels = [0] * len(labels)
    if kind == 'unaligned': labels.pop()
    with pytest.raises(ValueError): build_plan(keys, labels, folds)


def test_forged_plan_revalidated_at_selection():
    keys, labels, folds = population()
    with pytest.raises(ValueError): members(FoldPlan(tuple(keys), tuple(labels), tuple([0] * 25)), 0)


def test_dispatch_passes_disjoint_populations_to_vit(monkeypatch):
    from sciona.cassava_fold_dispatch import train_planned_fold
    import sciona.cassava_vit_fold as trainer
    plan = build_plan(*population())
    images = [f'synthetic-encoded-{i}'.encode() for i in range(25)]
    captured = []
    monkeypatch.setattr(trainer, 'train_fold', lambda *args, **kwargs: captured.append(args))
    train_planned_fold('vit', plan, images, 2)
    train_images, train_labels, val_images, val_labels = captured[0]
    assert set(train_images).isdisjoint(val_images)
    assert val_images == images[10:15]
    assert len(train_images) == 20 and len(train_labels) == 20 and val_labels == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError): train_planned_fold('vit', plan, images[:-1], 2)
    assert len(captured) == 1


def test_fold_means_align_permuted_keys_and_preserve_dtype():
    keys = ['synthetic-a', 'synthetic-b', 'synthetic-c']
    rng = np.random.default_rng(62)
    arrays = [rng.dirichlet(np.ones(5), size=3).astype('float32') for _ in range(5)]
    output = {}
    for fold, array in enumerate(arrays):
        order = np.roll(np.arange(3), fold)
        output[fold] = ([keys[i] for i in order], array[order])
    result = average_folds(keys, output)
    np.testing.assert_array_equal(result, np.stack(arrays).mean(0))
    assert result.dtype == np.float32


@pytest.mark.parametrize('kind', ['missing_fold', 'extra_row', 'duplicate', 'nan', 'sum', 'precision'])
def test_rejects_unaligned_predictions(kind):
    keys = ['synthetic-a']
    predictions = {i: (keys, np.full((1, 5), .2, dtype='float32')) for i in range(5)}
    if kind == 'missing_fold': predictions.pop(4)
    if kind == 'extra_row': predictions[0] = (['synthetic-other'], predictions[0][1])
    if kind == 'duplicate': predictions[0] = (keys * 2, np.full((2, 5), .2, dtype='float32'))
    if kind == 'nan': predictions[0][1][0, 0] = np.nan
    if kind == 'sum': predictions[0][1][:] = .1
    if kind == 'precision': predictions[0] = (keys, predictions[0][1].astype('float64'))
    with pytest.raises(ValueError): average_folds(keys, predictions)
