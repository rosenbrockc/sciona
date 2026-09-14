#!/usr/bin/env python3
"""Compare complete 102-feature vector with unchanged pinned helpers executed in Octave."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from scipy.io import savemat, loadmat
from sciona.atoms.riemannian_bci.signal_processing.andriy_mainstream_features import andriy_documented_mainstream_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    hashes = json.loads((root/'docs/reviews/andriy_mainstream_source_hashes.json').read_text())
    cases = []
    with TemporaryDirectory(prefix='sciona-andriy-mainstream-') as temporary:
        directory = Path(temporary)
        for name, sha in hashes.items():
            data = (args.reference_dir/('Andriy_code_'+name+'.m')).read_bytes()
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError('source drift')
            (directory/(name+'.m')).write_bytes(data)
        source = (directory/'iwmf_and_bw_low.m').read_text()
        old = '[fft_result,freq] = pwelch(data,[],[],N,Fs);'
        assert source.count(old) == 1
        adapted = source.replace(old,
            '[fft_result,freq] = pwelch(data,hamming(floor(length(data)/4.5)),floor(floor(length(data)/4.5)/2),N,Fs);')
        (directory/'iwmf_and_bw_low.m').write_text(adapted)
        (directory/'nanmean.m').write_text("function y=nanmean(x,dim); assert(all(isfinite(x(:)))); y=mean(x,dim); end;\n")
        (directory/'check.m').write_text("pkg local_list '/private/tmp/sciona-octave/octave_packages'; pkg load signal; pwelch('R12+'); load('input.mat'); result=zeros(rows(windows),102); for k=1:rows(windows); result(k,:)=SPC_extract_features(windows(k,:)')'; end; reference_version=version; save('-mat7-binary','output.mat','result','reference_version');\n")
        for seed in [10211,10212,10213]:
            for samples in [256,768,7680]:
                x = np.random.default_rng(seed).normal(size=(3,samples))
                x = np.vstack([x, np.zeros(samples), np.linspace(0, .0099, samples)])
                savemat(directory/'input.mat',dict(windows=x))
                run = subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','check.m'],cwd=directory,capture_output=True,text=True,timeout=60)
                if run.returncode:
                    raise RuntimeError(run.stderr)
                reference=loadmat(directory/'output.mat')
                actual=andriy_documented_mainstream_features(x)
                np.testing.assert_allclose(actual,reference['result'],rtol=1e-12,atol=1e-13)
                cases.append(dict(seed=seed,samples=samples,maximum_absolute_error=float(np.nanmax(abs(actual-reference['result']))), masked_rows=int(np.isnan(actual).all(axis=1).sum())))
        version=str(reference['reference_version'].reshape(-1)[0])
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_hashes=hashes,
        dependency_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            Path(inspect.getsourcefile(andriy_documented_mainstream_features)).parent.glob('andriy_*features.py')},
        provider_sha256=hashlib.sha256(Path(inspect.getsourcefile(andriy_documented_mainstream_features)).read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_mainstream_features.py').read_bytes()).hexdigest(),
        octave_version=version,numpy_version=np.__version__,cases=cases,
        documented_defaults_url='https://www.mathworks.com/help/signal/ref/pwelch.html',
        signal_package_version='1.4.8',
        octave_pwelch_sha256=hashlib.sha256(Path('/private/tmp/sciona-octave/packages/signal-1.4.8/pwelch.m').read_bytes()).hexdigest(),
        limitations=['Unchanged SPC feature extraction; Welch helper uses explicit documented Hamming window and overlap, with Octave R12+ no-detrending mode. Finite-only nanmean compatibility shim uses mean.',
                     'Current documented Welch settings only; historical MATLAB parity is unproven. No full model execution claim.',
                     'Source low-range random-replacement RNG side effects are not reproduced; final masked feature values are compared.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases_passed':len(cases),'maximum_error':max(c['maximum_absolute_error'] for c in cases)}))


if __name__ == '__main__':
    main()
