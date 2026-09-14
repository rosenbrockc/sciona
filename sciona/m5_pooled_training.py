"""Train every selected pool in all six independent M5 families."""
from dataclasses import dataclass
import numpy as np
from sciona.m5_family import frame,pools
from sciona.m5_training import fit


@dataclass(repr=False)
class PoolModel:
    recursive: bool
    pooling: str
    pool: tuple
    rows: np.ndarray
    features: tuple
    model: object
    report: dict


def train(prepared, cutoff, *, recursive_first_day=0, nonrecursive_first_day=710):
    """Preflight exact row coverage, then train all pools at full source controls.

    Runtime day origins may be generalized via explicit first-day controls.
    Returned models retain no copied raw inputs; pool/row maps remain private
    in-memory execution state and must not be included in public evidence.
    """
    for value in (cutoff,recursive_first_day,nonrecursive_first_day):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
            raise ValueError('Training day controls must be integers')
    grid=prepared['grid'];days=np.asarray(grid['day']);targets=np.asarray(grid['target'])
    if days.ndim!=1 or targets.shape!=days.shape or not days.size:
        raise ValueError('Aligned nonempty grid required')
    plans=[]
    for recursive in (True,False):
        first=recursive_first_day if recursive else nonrecursive_first_day
        if first>cutoff:raise ValueError('First day exceeds training cutoff')
        for pooling in ('outlet','outlet_category','outlet_department'):
            coverage=np.zeros(len(days),dtype=np.int16)
            for pool in pools(prepared,pooling):
                selected,rows=frame(prepared,recursive=recursive,pooling=pooling,pool=pool,first_day=first)
                train_rows=(days[rows]<=cutoff)
                diagnostic=(days[rows]>cutoff-28)&train_rows if recursive else ((days[rows]>cutoff)&(days[rows]<=cutoff+28))
                if (not train_rows.any() or not diagnostic.any()
                        or not np.isfinite(targets[rows][train_rows]).all()
                        or not (targets[rows][train_rows]>0).any()):
                    raise ValueError('Every pool needs complete training targets and diagnostics')
                coverage[rows]+=1
                plans.append((recursive,pooling,pool,first,selected,rows))
            if not np.array_equal(coverage,(days>=first).astype(np.int16)):
                raise ValueError('Family must cover each eligible grid row exactly once')
    results=[]
    for recursive,pooling,pool,first,selected,rows in plans:
        model,report=fit(selected,targets[rows],days[rows],cutoff,recursive=recursive,pooling=pooling,first_day=first)
        results.append(PoolModel(recursive,pooling,pool,rows.copy(),tuple(selected.columns),model,report))
    return results
