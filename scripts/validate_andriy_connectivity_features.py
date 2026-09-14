#!/usr/bin/env python3
"""Compare connectivity features with unchanged pinned helpers executed in Octave."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from scipy.io import savemat, loadmat
from sciona.atoms.riemannian_bci.signal_processing.andriy_connectivity_features import andriy_connectivity_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    hashes = json.loads((root/'docs/reviews/andriy_connectivity_source_hashes.json').read_text())
    cases = []
    with TemporaryDirectory(prefix='sciona-andriy-connectivity-') as temporary:
        directory = Path(temporary)
        for name, sha in hashes.items():
            data = (args.reference_dir/('Andriy_code_'+name+'.m')).read_bytes()
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError('source drift')
            (directory/(name+'.m')).write_bytes(data)
        (directory/'combnk.m').write_text("function y=combnk(x,k); y=nchoosek(x,k); end;\n")
        (directory/'nanmean.m').write_text("function y=nanmean(x,dim); if nargin<2; dim=find(size(x)~=1,1); if isempty(dim); dim=1; end; end; good=~isnan(x); x(~good)=0; y=sum(x,dim)./sum(good,dim); end;\n")
        (directory/'check.m').write_text("pkg local_list '/private/tmp/sciona-octave/octave_packages'; pkg load signal; pwelch('R12+'); load('input.mat'); result=zeros(size(windows,1),180); for k=1:size(windows,1); x=squeeze(windows(k,:,:)); if all(x(:)==0); result(k,:)=NaN; else; result(k,:)=connectivity_features_ALL(x,256,[.5 4;3 8;7 15;14 31;30 100],1,0)'; end; end; reference_version=version; save('-mat7-binary','output.mat','result','reference_version');\n")
        for seed in [10211,10212,10213]:
            for samples in [768,7680]:
                x = np.random.default_rng(seed).normal(size=(2,16,samples))
                x[1,0] = 0
                x = np.concatenate([x, np.zeros((1,16,samples))])
                savemat(directory/'input.mat',dict(windows=x))
                run = subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','check.m'],cwd=directory,capture_output=True,text=True,timeout=60)
                if run.returncode:
                    raise RuntimeError(run.stderr)
                reference=loadmat(directory/'output.mat')
                actual=andriy_connectivity_features(x)
                np.testing.assert_allclose(actual,reference['result'],rtol=1e-5,atol=1e-5)
                cases.append(dict(seed=seed,samples=samples,maximum_absolute_error=float(np.nanmax(abs(actual-reference['result'])))))
        version=str(reference['reference_version'].reshape(-1)[0])
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_hashes=hashes,
        provider_sha256=hashlib.sha256(Path(inspect.getsourcefile(andriy_connectivity_features)).read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_connectivity_features.py').read_bytes()).hexdigest(),
        octave_version=version,numpy_version=np.__version__,scipy_version=__import__('scipy').__version__,cases=cases,
        octave_signal_hashes={name:hashlib.sha256((Path('/private/tmp/sciona-octave/packages/signal-1.4.8')/(name+'.m')).read_bytes()).hexdigest() for name in ['butter','filtfilt','pwelch','cpsd','hilbert','xcorr']},
        limitations=['Pinned source in Octave signal R12+ mode with combnk=nchoosek and nanmean compatibility shims. Historical MATLAB numerical identity unproven.',
                     'Current SciPy Butterworth/filtfilt arithmetic differs from Octave; compare all features at rtol=1e-5, atol=1e-5. Full model execution remains separate.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases_passed':len(cases),'maximum_error':max(c['maximum_absolute_error'] for c in cases)}))


if __name__ == '__main__':
    main()
