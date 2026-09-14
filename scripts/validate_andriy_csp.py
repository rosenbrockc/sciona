#!/usr/bin/env python3
"""Validate CSP fitting and first projection against pinned Octave helpers."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import scipy
from scipy.io import savemat, loadmat
from sciona.atoms.riemannian_bci.signal_processing.andriy_csp import andriy_csp_filters, andriy_csp_project


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    hashes=json.loads((root/'docs/reviews/andriy_csp_source_hashes.json').read_text())
    cases=[]
    with TemporaryDirectory(prefix='sciona-csp-parity-') as temporary:
        directory=Path(temporary)
        for name,sha in hashes.items():
            source=(args.reference_dir/('Andriy_code_'+name+'.m')).read_bytes()
            if hashlib.sha256(source).hexdigest()!=sha:
                raise ValueError('source drift')
            (directory/(name+'.m')).write_bytes(source)
        (directory/'check.m').write_text("load('input.mat'); filters=csp(a,b); projected=spatFilt(x,filters,1); supplied=spatFilt(x,python_filters,1); reference_version=version; save('-mat7-binary','output.mat','filters','projected','supplied','reference_version');\n")
        for seed in [581,582,583]:
            for samples in [300,3840]:
                rng=np.random.default_rng(seed)
                a=rng.normal(size=(16,samples))*np.linspace(1,3,16)[:,None]
                b=rng.normal(size=(16,samples+100))*np.linspace(3,1,16)[:,None]
                x=rng.normal(size=(16,3840))
                actual_filters=andriy_csp_filters(a,b)
                actual_projection=andriy_csp_project(x,actual_filters)
                savemat(directory/'input.mat',dict(a=a,b=b,x=x,python_filters=actual_filters))
                run=subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','check.m'],cwd=directory,capture_output=True,text=True,timeout=60)
                if run.returncode:
                    raise RuntimeError(run.stderr)
                reference=loadmat(directory/'output.mat')
                signs=np.where(np.sum(actual_filters*reference['filters'],axis=1)<0,-1,1)
                aligned_filters=actual_filters*signs[:,None]
                aligned_projection=actual_projection*signs[0]
                np.testing.assert_allclose(aligned_filters,reference['filters'],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(aligned_projection,reference['projected'],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(actual_projection,reference['supplied'],rtol=1e-12,atol=1e-12)
                cases.append(dict(seed=seed,training_samples=samples,
                    aligned_filter_error=float(np.max(abs(aligned_filters-reference['filters']))),
                    aligned_projection_error=float(np.max(abs(aligned_projection-reference['projected']))),
                    supplied_filter_projection_error=float(np.max(abs(actual_projection-reference['supplied'])))))
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_hashes=hashes,
        provider_sha256=hashlib.sha256(Path(inspect.getsourcefile(andriy_csp_filters)).read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_csp.py').read_bytes()).hexdigest(),
        octave_version=str(reference['reference_version'].reshape(-1)[0]),
        numpy_version=np.__version__,scipy_version=scipy.__version__,cases=cases,
        limitations=['Source eigenvector signs aligned for fitted-filter comparison; supplied-filter projection compared without alignment.',
                     'Provider rejects singular/ill-conditioned class covariances and numerically repeated eigenvalues.',
                     'Current Octave generalized eigenvector scaling; historical MATLAB scaling and downstream AR sign invariance unproven.',
                     'Caller selection, preprocessing, AR and model execution remain separate.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases_passed=len(cases),maximum_projection_error=max(c['aligned_projection_error'] for c in cases))))


if __name__=='__main__':
    main()
