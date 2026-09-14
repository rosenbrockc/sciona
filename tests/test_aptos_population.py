"""Synthetic source-role separation and complete second-stage target oracle."""
import numpy as np
import pytest

from sciona.aptos_population import prepare_population, first_stage_targets, second_stage_targets


def population(**overrides):
    entries = dict(base=(['synthetic-base'], [4]), average=(['synthetic-average'], [0]),
                   grouped=(['synthetic-g1', 'synthetic-g2'], [2, 2]), pseudo_keys=['synthetic-pseudo'])
    entries.update(overrides)
    return prepare_population(**entries)


def test_supplemental_and_pseudo_roles_never_enter_first_stage():
    p = population()
    keys, values = first_stage_targets(p)
    assert keys == ('synthetic-base',)
    np.testing.assert_array_equal(values, [4.])
    values[0] = 0
    assert p.base_labels[0] == 4


def test_second_stage_role_oracle_and_teacher_identity_alignment():
    p = population()
    teacher = (['synthetic-pseudo', 'synthetic-g2', 'synthetic-average', 'synthetic-g1'],
               np.array([1.25, 3.3, 1.5, 1.1]))
    keys, values = second_stage_targets(p, teacher, lower_deviation=.5, upper_deviation=.5)
    assert keys == p.all_keys
    np.testing.assert_allclose(values, [4., .75, 1.7, 2.7, 1.25], rtol=0, atol=1e-15)


def test_cross_role_identity_overlap_rejected():
    with pytest.raises(ValueError, match='disjoint'):
        population(pseudo_keys=['synthetic-base'])


def test_grouped_role_has_four_levels():
    with pytest.raises(ValueError, match='four supplied levels'):
        population(grouped=(['synthetic-g1'], [4]))


@pytest.mark.parametrize('fault', ['missing', 'extra_base', 'integer'])
def test_teacher_coverage_or_type_rejected(fault):
    p = population()
    keys = list(p.average_keys + p.group_keys + p.pseudo_keys)
    if fault == 'missing': keys.pop()
    if fault == 'extra_base': keys.append('synthetic-base')
    values = np.ones(len(keys), dtype=int if fault == 'integer' else float)
    with pytest.raises(ValueError):
        second_stage_targets(p, (keys, values), lower_deviation=.5, upper_deviation=.5)
