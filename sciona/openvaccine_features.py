"""Pinned source features with explicit private folded-input boundaries.

Each record is processed separately, correcting the adapted source's unsupported
multi-record get_inputs call. No folding engine or input/output file I/O here.
"""
import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from sciona.openvaccine_models import SOURCE_SHA256, LICENSE_SHA256, validate_inputs
from sciona.openvaccine_structure import loop_labels

_FUNCTIONS={'return_ohe','get_input','get_pair_idx','calc_dist_to_pair','calc_dist_to_single',
            'get_structure_adj','pandas_list_to_array','preprocess_inputs1','preprocess_inputs',
            'get_inputs','padding_2D','get_distance_matrix','get_distance_matrix_2d','calc_neighbor'}


def folded_features(source_dir,sequences,structures,probabilities):
    if not isinstance(sequences,(list,tuple)) or not sequences or not isinstance(structures,(list,tuple)):
        raise ValueError('Nonempty explicit sequence/structure populations required')
    if len(sequences)!=len(structures) or len(sequences)!=len(probabilities):
        raise ValueError('Folded input counts must match')
    checked=[]
    length=None
    for sequence,structure,probability in zip(sequences,structures,probabilities):
        if not isinstance(sequence,str) or len(sequence)<2 or set(sequence)-set('ACGU'):
            raise ValueError('Supported RNA alphabet and minimum length two required')
        if not isinstance(structure,str) or len(sequence)!=len(structure):
            raise ValueError('Sequence and structure lengths must match')
        loops=loop_labels(structure)
        if length is None:length=len(sequence)
        if len(sequence)!=length:raise ValueError('Feature batches require equal sequence lengths')
        stack=[]
        for pos,token in enumerate(structure):
            if token=='(':stack.append(pos)
            elif token==')':
                start=stack.pop()
                if sequence[start]+sequence[pos] not in ('AU','UA','CG','GC','GU','UG'):
                    raise ValueError('Unsupported paired-base combination')
        p=np.asarray(probability)
        if p.dtype not in (np.dtype('float32'),np.dtype('float64')) or p.shape!=(length,length):
            raise ValueError('Square float pairing probabilities required')
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise ValueError('Pairing probabilities must be finite and within zero to one')
        if not np.allclose(p,p.T,rtol=0,atol=1e-7) or (np.diag(p)!=0).any() or (p.sum(axis=1)>1+1e-6).any():
            raise ValueError('Pairing probabilities must be symmetric, zero-diagonal and normalized')
        checked.append((sequence,structure,loops,p.copy()))
    source_dir=Path(source_dir)
    raw=(source_dir/'scripts/nullrecurrent_inference.py').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256 or hashlib.sha256((source_dir/'LICENSE').read_bytes()).hexdigest()!=LICENSE_SHA256:
        raise ValueError('Feature source/license identity mismatch')
    definitions=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in _FUNCTIONS]
    if {n.name for n in definitions}!=_FUNCTIONS:raise ValueError('Incomplete feature source')
    ns=dict(np=np,pd=pd,DIST_NEW=True,DIST_NEW2=True,BBP=True,BBP1=True,BBP2=True,BBP3=True,BBP4=True,BBP_TOTAL=8,
            token2int={x:i for i,x in enumerate('().ACGUBEHIMSXse')})
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'<pinned-openvaccine-features>','exec'),ns)
    nodes=[];adjacency=[]
    for sequence,structure,loops,p in checked:
        frame=pd.DataFrame([dict(id=0,sequence=sequence,structure=structure,bpRNA_string=loops,seq_length=length)])
        n,a=ns['get_inputs'](frame,p);nodes.append(n);adjacency.append(a)
    return validate_inputs(np.concatenate(nodes,axis=0),np.concatenate(adjacency,axis=0))
