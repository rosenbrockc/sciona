"""Source-separated first- and second-stage APTOS target populations.

The base population contains the caller-prepared combined first-stage data.
Five-level averaging, four-level group bounding and unlabeled pseudo-targets
enter only stage two. Runtime identities across roles must be disjoint.
Teacher aggregation is an explicit upstream choice, not inferred here.
"""
from dataclasses import dataclass

import numpy as np

from sciona.aptos_ensemble import _keys
from sciona.aptos_refinement import _align, average_ordinal_targets
from sciona.aptos_group_refinement import bound_group_targets


@dataclass(frozen=True)
class Population:
    base_keys: tuple
    base_labels: np.ndarray
    average_keys: tuple
    average_labels: np.ndarray
    group_keys: tuple
    group_labels: np.ndarray
    pseudo_keys: tuple

    @property
    def all_keys(self):
        return self.base_keys + self.average_keys + self.group_keys + self.pseudo_keys


def prepare_population(*, base, average, grouped, pseudo_keys):
    """Prepare role boundaries from (keys, supplied labels) pairs.

    Five-level source labels use 0..4; four-level grouping labels use 0..3.
    Each role must be present. No image contents or identity metadata are
    written; the caller retains responsibility for role provenance.
    """
    roles = []
    for entry in (base, average, grouped):
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError('Each labeled role requires identities and labels')
        keys = _keys(entry[0])
        labels = _align(keys, entry, ordinal=True)
        labels.setflags(write=False)
        roles.append((keys, labels))
    if (roles[2][1] > 3).any():
        raise ValueError('Grouped role requires four supplied levels encoded zero through three')
    pseudo = _keys(pseudo_keys)
    all_keys = tuple(key for keys, _ in roles for key in keys) + pseudo
    if len(set(all_keys)) != len(all_keys):
        raise ValueError('Training population roles must have disjoint identities')
    return Population(*roles[0], *roles[1], *roles[2], pseudo)


def first_stage_targets(population):
    """Only the base population participates in first-stage training."""
    return population.base_keys, population.base_labels.copy()


def second_stage_targets(population, teacher_predictions, *, lower_deviation, upper_deviation):
    """Retain base labels and add the three source-specified target roles.

    Teacher predictions must cover exactly the added populations. Base
    labels are never overwritten by teacher values. Returns identities in
    fixed role order and float64 soft targets for the complete second stage.
    """
    added = population.average_keys + population.group_keys + population.pseudo_keys
    teacher = _align(added, teacher_predictions, ordinal=False)
    a, g = len(population.average_keys), len(population.group_keys)
    average = average_ordinal_targets(population.average_keys,
        (population.average_keys, population.average_labels),
        (population.average_keys, teacher[:a]))
    grouped = bound_group_targets(population.group_keys,
        (population.group_keys, population.group_labels),
        (population.group_keys, teacher[a:a + g]),
        lower_deviation=lower_deviation, upper_deviation=upper_deviation)
    values = np.concatenate([population.base_labels, average, grouped, teacher[a + g:]])
    return population.all_keys, values
