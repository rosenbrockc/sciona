from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_get_auth_rates(TP: AbstractArray, FP: AbstractArray, TN: AbstractArray, FN: AbstractArray, thresholds: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=TP.shape, dtype="float64")


def witness_get_id_rates(H: AbstractArray, M: AbstractArray, R: AbstractArray, N: AbstractArray, thresholds: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=H.shape, dtype="float64")


def witness_get_subject_results(results: AbstractArray, subject: AbstractArray, thresholds: AbstractArray, subjects: AbstractArray, subject_dict: AbstractArray, subject_idx: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=results.shape, dtype="float64")


def witness_assess_classification(results: AbstractArray, thresholds: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=results.shape, dtype="float64")


def witness_assess_runs(results: AbstractArray, subjects: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=results.shape, dtype="float64")


def witness_combination(results: AbstractArray, weights: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=results.shape, dtype="float64")


def witness_majority_rule(labels: AbstractArray, random: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=labels.shape, dtype="float64")


def witness_cross_validation(labels: AbstractArray, n_iter: AbstractArray, test_size: AbstractArray, train_size: AbstractArray, random_state: AbstractArray) -> AbstractArray:
    return AbstractArray(shape=labels.shape, dtype="float64")
