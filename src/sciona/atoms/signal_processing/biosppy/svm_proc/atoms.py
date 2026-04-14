from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Hashable, Union

import biosppy.biometrics as biometrics
import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_assess_classification,
    witness_assess_runs,
    witness_combination,
    witness_cross_validation,
    witness_get_auth_rates,
    witness_get_id_rates,
    witness_get_subject_results,
    witness_majority_rule,
)


@register_atom(witness_get_auth_rates)
@icontract.require(lambda TP: TP is not None, "TP cannot be None")
@icontract.require(lambda FP: FP is not None, "FP cannot be None")
@icontract.require(lambda TN: TN is not None, "TN cannot be None")
@icontract.require(lambda FN: FN is not None, "FN cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.ensure(lambda result: result is not None, "Get Auth Rates output must not be None")
def get_auth_rates(TP: np.ndarray, FP: np.ndarray, TN: np.ndarray, FN: np.ndarray, thresholds: np.ndarray) -> dict:
    total = TP + FP + TN + FN
    FAR = np.where(total > 0, FP / (FP + TN + 1e-15), 0.0)
    FRR = np.where(total > 0, FN / (FN + TP + 1e-15), 0.0)
    accuracy = np.where(total > 0, (TP + TN) / total, 0.0)
    return {"FAR": FAR, "FRR": FRR, "accuracy": accuracy, "thresholds": thresholds}


@register_atom(witness_get_id_rates)
@icontract.require(lambda H: H is not None, "H cannot be None")
@icontract.require(lambda M: M is not None, "M cannot be None")
@icontract.require(lambda R: R is not None, "R cannot be None")
@icontract.require(lambda N: N is not None, "N cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.ensure(lambda result: result is not None, "Get Id Rates output must not be None")
def get_id_rates(H: np.ndarray, M: np.ndarray, R: np.ndarray, N: int, thresholds: np.ndarray) -> dict:
    total = H + M + R
    accuracy = np.where(total > 0, H / total, 0.0)
    miss_rate = np.where(total > 0, M / total, 0.0)
    reject_rate = np.where(total > 0, R / total, 0.0)
    return {"accuracy": accuracy, "miss_rate": miss_rate, "reject_rate": reject_rate, "N": N, "thresholds": thresholds}


@register_atom(witness_get_subject_results)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda subject: subject is not None, "subject cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.require(lambda subjects: subjects is not None, "subjects cannot be None")
@icontract.require(lambda subject_dict: subject_dict is not None, "subject_dict cannot be None")
@icontract.require(lambda subject_idx: subject_idx is not None, "subject_idx cannot be None")
@icontract.ensure(lambda result: result is not None, "Get Subject Results output must not be None")
def get_subject_results(
    results: dict[str, Any],
    subject: Hashable,
    thresholds: np.ndarray,
    subjects: Sequence[Hashable],
    subject_dict: Mapping[Hashable, int],
    subject_idx: Sequence[int],
) -> dict[str, Any]:
    auth = results.get("authentication", {})
    ident = results.get("identification", {})
    return {"authentication": auth, "identification": ident, "subject": subject, "thresholds": thresholds}


@register_atom(witness_assess_classification)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda thresholds: thresholds is not None, "thresholds cannot be None")
@icontract.ensure(lambda result: result is not None, "Assess Classification output must not be None")
def assess_classification(results: dict, thresholds: np.ndarray) -> dict:
    return {"results": results, "thresholds": thresholds}


@register_atom(witness_assess_runs)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda subjects: subjects is not None, "subjects cannot be None")
@icontract.ensure(lambda result: result is not None, "Assess Runs output must not be None")
def assess_runs(results: list, subjects: list) -> dict:
    return {"results": results, "subjects": subjects}


@register_atom(witness_combination)
@icontract.require(lambda results: results is not None, "results cannot be None")
@icontract.require(lambda weights: weights is not None, "weights cannot be None")
@icontract.ensure(lambda result: result is not None, "Combination output must not be None")
def combination(results: dict, weights: dict) -> tuple:
    all_decisions = []
    all_weights = []
    for clf, res in results.items():
        w = weights.get(clf, 1.0) if weights else 1.0
        all_decisions.append(res)
        all_weights.append(w)
    classes = sorted(set(all_decisions))
    counts = np.array([sum(w for d, w in zip(all_decisions, all_weights) if d == c) for c in classes])
    best = classes[np.argmax(counts)]
    return (best, float(np.max(counts) / np.sum(counts)), counts, np.array(classes))


@register_atom(witness_majority_rule)
@icontract.require(lambda labels: labels is not None, "labels cannot be None")
@icontract.require(lambda random: random is not None, "random cannot be None")
@icontract.ensure(lambda result: result is not None, "Majority Rule output must not be None")
def majority_rule(labels: Union[np.ndarray, list], random: bool) -> tuple:
    labels_arr = np.asarray(labels)
    unique, counts = np.unique(labels_arr, return_counts=True)
    best = unique[np.argmax(counts)]
    count = int(np.max(counts))
    if random and np.sum(counts == count) > 1:
        tied = unique[counts == count]
        best = np.random.choice(tied)
    return (best, count)


@register_atom(witness_cross_validation)
@icontract.require(lambda labels: labels is not None, "labels cannot be None")
@icontract.require(lambda n_iter: n_iter is not None, "n_iter cannot be None")
@icontract.require(lambda test_size: test_size is not None, "test_size cannot be None")
@icontract.ensure(lambda result: result is not None, "Cross Validation output must not be None")
def cross_validation(
    labels: Union[list, np.ndarray],
    n_iter: int = 10,
    test_size: Union[float, int] = 0.1,
    train_size: Union[float, int, None] = None,
    random_state: Union[int, None] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    return biometrics.cross_validation(
        labels=labels,
        n_iter=n_iter,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
    )[0]
