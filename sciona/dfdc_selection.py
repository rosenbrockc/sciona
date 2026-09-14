"""Explicit DFDC run/checkpoint selection without historical filename inference.

Source request ordering: Selim Seferbekov, MIT, commit
89c6290490bac96b29193a4061b3db9dd3933e36; docs/licenses/DFDC-MIT.txt.
A run's seed labels its population resets; it does not authenticate old weights.
"""
import math
from collections.abc import Mapping

SOURCE_REQUESTS = ((111, 0, 36), (555, 0, 19), (777, 0, 29),
                   (777, 0, 31), (888, 0, 37), (888, 0, 40), (999, 0, 23))
SOURCE_RUNS = ((111, 0, 40), (555, 0, 40), (777, 0, 40),
               (888, 0, 40), (999, 0, 40))


def _triples(values, name):
    try:
        result = tuple(tuple(row) for row in values)
    except TypeError as error:
        raise ValueError(f'{name} must contain integer triples') from error
    if not result or any(len(row) != 3 or any(type(v) is not int or v < 0 for v in row)
                         for row in result):
        raise ValueError(f'{name} must contain nonnegative integer triples')
    return result


def selection_plan(runs=SOURCE_RUNS, requests=SOURCE_REQUESTS):
    """Validate an explicit executable plan; never extend runs or relabel epochs.

    Runs are (seed, fold, epoch_count); requests are (seed, fold, zero_based_epoch).
    Source defaults deliberately fail: suffix40 requires an unavailable epoch.
    Explicit caller changes return a visible adaptation classification.
    """
    runs = _triples(runs, 'runs'); requests = _triples(requests, 'requests')
    if len({row[:2] for row in runs}) != len(runs) or len(set(requests)) != len(requests):
        raise ValueError('duplicate runs or checkpoint requests are not allowed')
    counts = {row[:2]: row[2] for row in runs}
    if any(count <= 0 or max(count, 2)*seed >= 2**32 for seed, fold, count in runs):
        raise ValueError('run epoch counts must support source population seeds')
    if {row[:2] for row in requests} != set(counts):
        raise ValueError('requested run keys must exactly match configured run keys')
    if any(epoch >= counts[(seed, fold)] for seed, fold, epoch in requests):
        raise ValueError('requested checkpoint is outside its explicit training run; no automatic epoch extension')
    return {'runs': [list(row) for row in runs], 'requests': [list(row) for row in requests],
            'source_requests_preserved': requests == SOURCE_REQUESTS,
            'source_runs_preserved': runs == SOURCE_RUNS,
            'provenance': 'explicit_execution_plan; historical checkpoint identity unverified'}


def select_snapshots(plan, snapshots):
    """Select ordered in-memory snapshots, requiring explicit run and epoch keys.

    Snapshot tensors remain owned by the caller. Classifier construction performs
    the strict tensor key/shape/dtype/finite checks before inference. Metadata
    checks here cannot authenticate externally supplied historical weights.
    """
    if not isinstance(plan, Mapping):
        raise ValueError('selection requires an explicit plan')
    verified = selection_plan(plan.get('runs', ()), plan.get('requests', ()))
    if dict(plan) != verified:
        raise ValueError('selection plan metadata is inconsistent')
    if not isinstance(snapshots, Mapping):
        raise ValueError('snapshots must be keyed by run seed, fold and epoch')
    selected = []
    for key in map(tuple, verified['requests']):
        if key not in snapshots:
            raise ValueError('a requested snapshot is missing')
        snapshot = snapshots[key]
        if (not isinstance(snapshot, Mapping) or snapshot.get('kind') != str(key[2])
                or type(snapshot.get('epoch')) is not int or snapshot['epoch'] != key[2]+1
                or not isinstance(snapshot.get('state_dict'), Mapping) or not snapshot['state_dict']):
            raise ValueError('snapshot numbered epoch metadata or state mapping is invalid')
        best = snapshot.get('bce_best')
        if type(best) not in (float, int) or not math.isfinite(best) or best < 0:
            raise ValueError('snapshot best-loss metadata must be finite and nonnegative')
        selected.append(snapshot)
    return selected
