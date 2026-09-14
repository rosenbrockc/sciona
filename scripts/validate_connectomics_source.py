"""Run original Connectomics sweeps with its pinned historical PCA semantics."""
import argparse
import ast
import contextlib
import hashlib
import io
import itertools
import json
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils import as_float_array


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.source/'manifest.json').read_text())
    for p in manifest['pins']:
        assert hashlib.sha256((args.source/p['software_path']).read_bytes()).hexdigest()==p['sha256']
    legacy=args.source/'legacy_sklearn'
    old=json.loads((legacy/'manifest.json').read_text())
    path='sklearn/decomposition/pca.py'
    raw=(legacy/path).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==next(p['sha256'] for p in old['pins'] if p['software_path']==path)
    ns={'np':np,'linalg':linalg,'sqrt':sqrt,'BaseEstimator':BaseEstimator,
        'TransformerMixin':TransformerMixin,'array2d':lambda x:np.atleast_2d(x),
        'as_float_array':as_float_array}
    classes=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='PCA']
    assert len(classes)==1
    exec(compile(ast.Module(body=classes,type_ignores=[]),'<legacy-source-pca>','exec'),ns)
    legacy_pca=ns['PCA']
    fits=[]
    class CountPCA(legacy_pca):
        def fit(self,x,*args,**kwargs):
            fits.append(hashlib.sha256(x.tobytes()).hexdigest())
            return super().fit(x,*args,**kwargs)
    ns.update(PCA=CountPCA,chain=itertools.chain,Parallel=joblib.Parallel,delayed=joblib.delayed,cpu_count=joblib.cpu_count)
    for relative in ['code/utils.py','code/PCA.py','code/directivity.py']:
        source=(args.source/relative).read_text().replace('dtype=np.int','dtype=int')
        nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),relative,'exec'),ns)
    rng=np.random.default_rng(1729)
    x=np.asfortranarray(rng.uniform(0.,.8,size=(240,10)),dtype=np.float32)
    initial=x.copy()
    for name in ['simple_filter','tuned_filter']:
        a=ns[name](x.copy(),threshold=.01)
        b=ns[name](x.copy(),threshold=.9)
        np.testing.assert_array_equal(a,b)
    # Verify historical precision against inverse of its generative covariance.
    pca=legacy_pca(n_components=8,whiten=True).fit(x)
    np.testing.assert_allclose(pca.get_precision(),np.linalg.inv(pca.get_covariance()),rtol=2e-5,atol=2e-5)
    from sklearn.decomposition import PCA as ModernPCA
    modern=ModernPCA(n_components=8,whiten=True,svd_solver='full').fit(x)
    precision_delta=float(np.max(np.abs(modern.get_precision()-pca.get_precision())))
    assert not np.allclose(modern.get_precision(),pca.get_precision(),rtol=1e-4,atol=1e-4)
    sweeps=[]
    with contextlib.redirect_stdout(io.StringIO()):
        for name,expected in [('make_simple_inference',240),('make_tuned_inference',480)]:
            fits.clear()
            result=ns[name](x)
            assert len(fits)==expected
            assert len(set(fits))==(2 if expected==240 else 4)
            assert np.isfinite(result).all() and result.min()==0 and result.max()==1
            sweeps.append({'method':name,'pca_fits':len(fits),'distinct_preprocessed_inputs':len(set(fits)),
                           'finite_scaled_scores':True})
        direct=ns['make_prediction_directivity'](x,n_jobs=1)
    assert np.isfinite(direct).all()
    final=.997*result+.003*direct
    assert np.isfinite(final).all()
    np.testing.assert_array_equal(x,initial)
    # Source uses strict bounds, not the closed interval described by intake.
    boundary=np.array([[0.,0.,0.],[0.,.2,.5]])
    assert np.count_nonzero(ns['_parallel_count'](boundary,0,1))==0
    args.output.write_text(json.dumps({'source_commit':manifest['commit'],'legacy_sklearn_commit':old['commit'],
        'sweeps':sweeps,'threshold_argument_ignored_reproduced':True,
        'legacy_precision_inverse_covariance_check':True,'modern_precision_max_abs_difference':precision_delta,'directivity_strict_bounds':True,
        'finite_tuned_directivity_blend':True,'inputs_unchanged':True,
        'scope':'Original source simple/tuned sweeps and directivity on synthetic float32 input, using pinned historical PCA class. Only np.int and array validation interface adaptations. Source threshold/filter-forwarding bugs remain uncorrected.',
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('720 source PCA fits passed; threshold sweep degeneracy and strict directivity bounds verified')


if __name__=='__main__':main()
