"""Explicit all-branch meta layout with row/fold alignment checks.

The five-feature t-SNE interpretation yields 293 first-level columns, not the
297 claimed elsewhere in the source. Choosing this layout is an explicit
independent realization; it does not resolve that historical contradiction.
Alignment checks alone do not prove that producers excluded held-out labels.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True,repr=False)
class AlignedBlock:
    training: np.ndarray
    query: np.ndarray
    training_ids: tuple
    query_ids: tuple
    folds: object = None


def assemble(first_level,supplemental,raw_neural,training_ids,query_ids,folds,*,tsne_interpretation):
    if tsne_interpretation!='five_features':raise ValueError('Explicit five-feature t-SNE interpretation required')
    if type(first_level) is not dict or any(type(k) is not int for k in first_level) or set(first_level)!=set(range(1,34)):
        raise ValueError('All 33 first-level entries required')
    if type(supplemental) is not dict or any(type(k) is not int for k in supplemental) or set(supplemental)!=set(range(1,8)):
        raise ValueError('All seven shared supplemental blocks required')
    ids_seen=set()
    for ids in (training_ids,query_ids):
        if type(ids) is not list or not ids or any(type(i) is not str or not i for i in ids):raise ValueError('Nonempty opaque identity lists required')
        for identity in ids:
            if identity in ids_seen:raise ValueError('Duplicate or overlapping identities')
            ids_seen.add(identity)
    f=np.asarray(folds)
    if f.shape!=(len(training_ids),) or f.dtype.kind not in 'iu' or set(f.tolist())!=set(range(5)):
        raise ValueError('Five aligned first-level folds required')
    def read(block,*,width=None,supervised=False):
        if not isinstance(block,AlignedBlock):raise ValueError('Aligned feature block required')
        if block.training_ids!=tuple(training_ids) or block.query_ids!=tuple(query_ids):raise ValueError('Block identity order mismatch')
        a=np.asarray(block.training,dtype=float);b=np.asarray(block.query,dtype=float)
        if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=len(training_ids) or b.shape[0]!=len(query_ids) or a.shape[1]<1 or b.shape[1]!=a.shape[1] or (width is not None and a.shape[1]!=width):raise ValueError('Invalid block dimensions')
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite block values')
        if supervised or block.folds is not None:
            supplied=np.asarray(block.folds)
            if supplied.dtype.kind not in 'iu' or not np.array_equal(supplied,f):raise ValueError('Block fold assignment mismatch')
        return a,b
    training=[];query=[]
    for entry in range(1,34):
        a,b=read(first_level[entry],width=5 if entry==10 else 9,supervised=entry!=10)
        if entry!=10 and ((a<0).any() or (a>1).any() or (b<0).any() or (b>1).any()):raise ValueError('Classifier scores outside probability range')
        training.append(a);query.append(b)
    for entry in range(1,8):
        a,b=read(supplemental[entry],width=1 if entry==7 else None,supervised=entry<=5)
        if entry<=5 and (a.shape[1]%9 or (a<0).any() or (b<0).any()):raise ValueError('Nonnegative nine-class distance groups required')
        training.append(a);query.append(b)
    tree_training=np.column_stack(training);tree_query=np.column_stack(query)
    a,b=read(raw_neural)
    if (a<0).any() or (b<0).any():raise ValueError('Nonnegative raw neural features required')
    return dict(tree_training=tree_training,tree_query=tree_query,
                neural_training=np.column_stack((tree_training,a)),neural_query=np.column_stack((tree_query,b)),
                first_level_width=293,tsne_interpretation=tsne_interpretation)
