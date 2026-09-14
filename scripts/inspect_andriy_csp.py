#!/usr/bin/env python3
"""Audit CSP generalized-eigenvector scaling against pinned source in Octave."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import scipy
from scipy.io import savemat, loadmat
from scipy.linalg import eig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = (args.reference_dir/'Andriy_code_csp.m').read_bytes()
    cases = []
    with TemporaryDirectory(prefix='sciona-csp-audit-') as temporary:
        directory = Path(temporary)
        (directory/'csp.m').write_bytes(source)
        (directory/'check.m').write_text("load('input.mat'); result=csp(a,b); reference_version=version; save('-mat7-binary','output.mat','result','reference_version');\n")
        for seed in [581,582,583]:
            rng = np.random.default_rng(seed)
            a = rng.normal(size=(16,2000))*np.linspace(1,3,16)[:,None]
            b = rng.normal(size=(16,2500))*np.linspace(3,1,16)[:,None]
            savemat(directory/'input.mat',dict(a=a,b=b))
            run = subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','check.m'],cwd=directory,capture_output=True,text=True,timeout=60)
            if run.returncode:
                raise RuntimeError(run.stderr)
            reference = loadmat(directory/'output.mat')
            target = reference['result']
            ra, rb = a@a.T, b@b.T
            ra, rb = ra/np.trace(ra), rb/np.trace(rb)
            values, vectors = np.linalg.eigh(ra+rb)
            whitening = (1/np.sqrt(values[::-1]))[:,None]*vectors[:,::-1].T
            values, vectors = eig(whitening@ra@whitening.T,whitening@rb@whitening.T)
            assert np.max(np.abs(values.imag)) < 1e-12
            assert np.max(np.abs(vectors.imag)) < 1e-12
            vectors = vectors[:,np.argsort(values.real)].real
            unit_result = vectors.T@whitening
            max_component_result = (vectors/np.max(abs(vectors),axis=0)).T@whitening
            def aligned(candidate):
                signs = np.where(np.sum(candidate*target,axis=1)<0,-1,1)
                return candidate*signs[:,None]
            np.testing.assert_allclose(aligned(max_component_result),target,rtol=1e-11,atol=1e-11)
            cases.append(dict(seed=seed,
                unit_norm_maximum_error=float(np.max(abs(aligned(unit_result)-target))),
                max_component_maximum_error=float(np.max(abs(aligned(max_component_result)-target)))))
    report = dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',
        source_sha256=hashlib.sha256(source).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        octave_version=str(reference['reference_version'].reshape(-1)[0]),
        numpy_version=np.__version__,scipy_version=scipy.__version__,cases=cases,
        limitations=['Audit only; no provider or catalog promotion.',
                     'Eigenvector signs aligned for comparison. Repeated eigenvalues and singular class covariances are not covered.',
                     'Current Octave generalized eigenvector scaling; historical MATLAB scaling is unproven.',
                     'Caller training selection, spatial projection, AR features, and complete model pipeline are not validated here.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases_passed=len(cases),maximum_aligned_error=max(c['max_component_maximum_error'] for c in cases))))


if __name__ == '__main__':
    main()
