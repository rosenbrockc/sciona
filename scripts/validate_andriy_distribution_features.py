#!/usr/bin/env python3
"""Compare distribution features with unchanged pinned helpers executed in Octave."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from scipy.io import savemat, loadmat
from sciona.atoms.riemannian_bci.signal_processing.andriy_distribution_features import andriy_distribution_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    hashes = json.loads((root/'docs/reviews/andriy_distribution_source_hashes.json').read_text())
    cases = []
    with TemporaryDirectory(prefix='sciona-andriy-distribution-') as temporary:
        directory = Path(temporary)
        for name, sha in hashes.items():
            data = (args.reference_dir/('Andriy_code_'+name+'.m')).read_bytes()
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError('source drift')
            (directory/(name+'.m')).write_bytes(data)
        # The repository helpers call toolbox nanmean, which is absent in base
        # Octave. All audit inputs are finite, so ordinary mean is equivalent.
        # Keep this compatibility shim explicit; do not claim original toolbox parity.
        (directory/'nanmean.m').write_text("function y=nanmean(x,dim); assert(all(isfinite(x(:)))); y=mean(x,dim); end;\n")
        (directory/'check.m').write_text("load('input.mat'); result=zeros(rows(windows),4); for k=1:rows(windows); x=windows(k,:)'; [e,f]=inf_theory(x); result(k,:)=[kurtosis(x),skewness(x),e,f]; end; reference_version=version; save('-mat7-binary','output.mat','result','reference_version');\n")
        for seed in [10211,10212,10213]:
            for samples in [21,100,7680]:
                x = np.random.default_rng(seed).normal(size=(3,samples))
                savemat(directory/'input.mat',dict(windows=x))
                run = subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','check.m'],cwd=directory,capture_output=True,text=True,timeout=60)
                if run.returncode:
                    raise RuntimeError(run.stderr)
                reference=loadmat(directory/'output.mat')
                actual=andriy_distribution_features(x)
                np.testing.assert_allclose(actual,reference['result'],rtol=1e-12,atol=1e-13)
                cases.append(dict(seed=seed,samples=samples,maximum_absolute_error=float(np.max(abs(actual-reference['result'])))))
        version=str(reference['reference_version'].reshape(-1)[0])
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_hashes=hashes,
        provider_sha256=hashlib.sha256(Path(inspect.getsourcefile(andriy_distribution_features)).read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_distribution_features.py').read_bytes()).hexdigest(),
        octave_version=version,numpy_version=np.__version__,cases=cases,
        limitations=['Pinned helpers in current Octave; finite-only nanmean compatibility shim uses mean. No original MATLAB toolbox runtime claim.',
                     'Four helper outputs only; low-range masking and full model execution remain separate.',
                     'Random finite windows only in cross-language comparisons; degenerate zero and final-sample behavior tested separately.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases_passed':len(cases),'maximum_error':max(c['maximum_absolute_error'] for c in cases)}))


if __name__ == '__main__':
    main()
