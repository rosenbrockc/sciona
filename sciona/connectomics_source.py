"""Verified Connectomics source with explicit threshold/filter corrections.

Execution source declares BSD 3 clause; retained notices are in docs/licenses.
Original files remain unchanged. This version implements the intended sweep
and filter-specific weighting, without claiming original-output equivalence.
"""
import ast
import hashlib
import itertools
import json
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils import as_float_array

PINS_SHA256='5f6cbd3868c61956af0b63e86dfb6a0c14fcdf56bb6c37f11330478641403057'


def correct_source(raw,expected_sha256):
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Connectomics source hash mismatch')
    text=raw.decode()
    replacements=[('X = h(X)','X = h(X, threshold=threshold)',2),
                  ('threshold=t, weights=True','threshold=threshold, weights=True',2),
                  ('X = w_star(X)','X = w_star(X, filtering=LP)',1)]
    for before,after,count in replacements:
        if text.count(before)!=count:
            raise ValueError('Connectomics source no longer matches reviewed correction')
        text=text.replace(before,after)
    return text


def source_namespace(cache):
    raw=Path(__file__).resolve().parents[1].joinpath('docs/reviews/competition_connectomics_source_pins.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PINS_SHA256:
        raise ValueError('Connectomics source manifest hash mismatch')
    manifest=json.loads(raw)
    files={}
    for prefix,pins in [('',manifest['pins']),('legacy_sklearn/',manifest['legacy_pca']['pins'])]:
        for pin in pins:
            name=prefix+pin['software_path']
            path=Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Invalid Connectomics source path')
            contents=(Path(cache)/path).read_bytes()
            if hashlib.sha256(contents).hexdigest()!=pin['sha256']:
                raise ValueError('Connectomics software hash mismatch')
            files[name]=contents
    ns={'np':np,'linalg':linalg,'sqrt':sqrt,'BaseEstimator':BaseEstimator,
        'TransformerMixin':TransformerMixin,'array2d':lambda x:np.atleast_2d(x),
        'as_float_array':as_float_array,'chain':itertools.chain,'Parallel':joblib.Parallel,
        'delayed':joblib.delayed,'cpu_count':joblib.cpu_count}
    classes=[n for n in ast.parse(files['legacy_sklearn/sklearn/decomposition/pca.py']).body if isinstance(n,ast.ClassDef) and n.name=='PCA']
    if len(classes)!=1:raise ValueError('Missing historical PCA implementation')
    exec(compile(ast.Module(body=classes,type_ignores=[]),'<verified-historical-pca>','exec'),ns)
    for name in ('code/utils.py','code/PCA.py','code/directivity.py'):
        source=files[name].decode()
        if name=='code/PCA.py':
            digest=next(p['sha256'] for p in manifest['pins'] if p['software_path']==name)
            source=correct_source(files[name],digest)
        if name=='code/directivity.py':
            if source.count('dtype=np.int')!=1:raise ValueError('Directivity integer compatibility mismatch')
            source=source.replace('dtype=np.int','dtype=int')
        definitions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions,type_ignores=[]),name,'exec'),ns)
    return ns
