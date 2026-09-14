#!/usr/bin/env python3
"""Measure default versus explicit-filter resampling across Octave and SciPy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import scipy
from scipy.io import savemat, loadmat
from scipy.signal import firwin, resample_poly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--octave', default='/opt/homebrew/bin/octave-cli')
    parser.add_argument('--package-list', type=Path, default=Path('/private/tmp/sciona-octave/octave_packages'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cases = []
    with TemporaryDirectory(prefix='sciona-resample-audit-') as temporary:
        directory = Path(temporary)
        package_path = str(args.package_list).replace("'", "''")
        script = "pkg local_list '"+package_path+"'; pkg load signal; load('input.mat');\n"
        script += "[default_result,default_filter]=resample(x,p,q); explicit_result=resample(x,p,q,b); reference_version=version;\nsave('-mat7-binary','output.mat','default_result','default_filter','explicit_result','reference_version');\n"
        (directory/'check.m').write_text(script)
        for target in [128, 256]:
            gcd = math.gcd(target, 400)
            p, q = target//gcd, 400//gcd
            # Current MathWorks documentation: order=2*10*max(p,q), beta=5,
            # cutoff=1/max(p,q), normalized DC gain=p. This is a documented
            # design reconstruction, NOT coefficients captured from MATLAB.
            b = p*firwin(2*10*max(p,q)+1, 1/max(p,q), window=('kaiser', 5.))
            for count in [4001, 236801]:
                x = np.random.default_rng(10101).normal(size=(count, 2))
                expected = resample_poly(x, p, q, axis=0, window=b/p)
                savemat(directory/'input.mat', dict(x=x, p=float(p), q=float(q), b=b))
                run = subprocess.run([args.octave,'--quiet','--no-gui','--no-history','check.m'],cwd=directory,capture_output=True,text=True,timeout=120)
                if run.returncode:
                    raise RuntimeError(run.stderr)
                result = loadmat(directory/'output.mat')
                np.testing.assert_allclose(result['explicit_result'],expected,rtol=1e-12,atol=1e-13)
                cases.append(dict(target_frequency=target, synthetic_input_samples=count,
                    documented_design_taps=len(b), octave_default_taps=result['default_filter'].size,
                    explicit_filter_max_error=float(np.max(abs(result['explicit_result']-expected))),
                    default_filter_max_difference=float(np.max(abs(result['default_result']-expected)))))
        version = str(result['reference_version'].reshape(-1)[0])
    report = dict(documentation='https://www.mathworks.com/help/signal/ref/resample.html',
        documentation_design={'neighbor_terms':10,'kaiser_beta':5,'filter_order':'2*n*max(reduced_p,reduced_q)','dc_gain':'reduced_p'},
        octave_version=version,numpy_version=np.__version__,scipy_version=scipy.__version__,cases=cases,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        conclusions=['Octave default resample is not an interchangeable MATLAB-default reference.',
                     'With identical explicit coefficients, Octave and SciPy agree on polyphase filtering, output length and delay compensation.',
                     'Coefficients reconstruct current documented MATLAB design; historical MATLAB runtime/default coefficient parity remains unproven.',
                     'No Andriy resampling atom or full graph approval is implied.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'maximum_explicit_filter_error':max(c['explicit_filter_max_error'] for c in cases),'maximum_default_difference':max(c['default_filter_max_difference'] for c in cases)}))


if __name__ == '__main__':
    main()
