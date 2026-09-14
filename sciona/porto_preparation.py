"""Explicit column-role preparation for the pending independent Porto stack.

Column roles come from private caller configuration. Categories are one-hot
encoded over combined training/query features; numeric sentinel values remain
literal. The normalized population feeds DAEs; the unnormalized prepared view
feeds the tree. No labels enter this operation.
"""
import numpy as np
from sciona.porto_transforms import _matrix,rank_gauss_population


def prepare_populations(training,query,*,dropped_columns,categorical_columns,binary_columns):
    x=_matrix(training);q=_matrix(query)
    if x.shape[1]!=q.shape[1]:raise ValueError('Population feature widths differ')
    seen=set()
    for columns in (dropped_columns,categorical_columns,binary_columns):
        if type(columns) is not list or any(type(i) is not int or not 0<=i<x.shape[1] for i in columns) or len(set(columns))!=len(columns) or seen.intersection(columns):raise ValueError('Disjoint explicit column roles required')
        seen.update(columns)
    all_values=np.vstack((x,q));excluded=set(dropped_columns)|set(categorical_columns)
    numeric=[i for i in range(x.shape[1]) if i not in excluded]
    if not numeric and not categorical_columns:raise ValueError('No model features retained')
    pieces=[all_values[:,numeric]] if numeric else []
    binary=[numeric.index(i) for i in binary_columns]
    width=len(numeric)
    for column in categorical_columns:
        categories=np.unique(all_values[:,column])
        one_hot=(all_values[:,column,None]==categories[None,:]).astype(float)
        pieces.append(one_hot);binary.extend(range(width,width+len(categories)));width+=len(categories)
    raw=np.column_stack(pieces)
    normalized=rank_gauss_population(raw,binary_columns=binary)
    return dict(tree_training=raw[:len(x)].copy(),tree_query=raw[len(x):].copy(),
        dae_training=normalized[:len(x)].copy(),dae_query=normalized[len(x):].copy(),
        prepared_width=width,binary_output_columns=tuple(binary))
