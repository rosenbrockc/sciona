#!/usr/bin/env python3
"""Compare time features with unchanged pinned helpers executed in Octave."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from scipy.io import savemat, loadmat
from sciona.atoms.riemannian_bci.signal_processing.andriy_spectral_features import andriy_spectral_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    hashes = json.loads((root/'docs/reviews/andriy_spectral_source_hashes.json').read_text())
    cases = []
    with TemporaryDirectory(prefix='sciona-andriy-time-') as temporary:
        directory = Path(temporary)
        for name, sha in hashes.items():
            data = (args.reference_dir/('Andriy_code_'+name+'.m')).read_bytes()
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError('source drift')
            (directory/(name+'.m')).write_bytes(data)
        (directory/'check.m').write_text("load('input.mat'); result=zeros(rows(windows),82);\nfor k=1:rows(windows)\n data=windows(k,:)'; n=length(data); Y=fft(data,n); spectrum=(Y.*conj(Y))/n; spectrum=spectrum(1:n/2+1); freq=256*(0:n/2)/n;\n total=EEG_subbands(spectrum,freq,[0,128]); bands=zeros(1,31);\n for j=0:30; bands(j+1)=EEG_subbands(spectrum,freq,[j,j+2]); end\n edges=zeros(1,3); percentages=[.9,.95,.8]; for j=1:3; edges(j)=spectral_edge(spectrum,freq,1,32,percentages(j)); end\n ranges=[3,15;15,30;59,61;51,69;20,30;59,61;25,128]; extra=zeros(1,7);\n for j=1:7; extra(j)=EEG_subbands(spectrum,freq,ranges(j,:)); end\n [peak,peak_freq]=EEG_PSD_features(spectrum,freq,1,32); entropy=spectral_entropy_g(spectrum,length(spectrum));\n result(k,:)=[total,bands,bands([12:31,1:11])/total,edges,extra,extra/total,peak_freq,entropy];\nend\nreference_version=version;save('-mat7-binary','output.mat','result','reference_version');\n")
        for seed in [10211,10212,10213]:
            for samples in [256,768,7680]:
                x = np.random.default_rng(seed).normal(size=(3,samples))
                savemat(directory/'input.mat',dict(windows=x))
                run = subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','check.m'],cwd=directory,capture_output=True,text=True,timeout=60)
                if run.returncode:
                    raise RuntimeError(run.stderr)
                reference=loadmat(directory/'output.mat')
                actual=andriy_spectral_features(x)
                np.testing.assert_allclose(actual,reference['result'],rtol=1e-12,atol=1e-13)
                cases.append(dict(seed=seed,samples=samples,maximum_absolute_error=float(np.max(abs(actual-reference['result'])))))
        version=str(reference['reference_version'].reshape(-1)[0])
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_hashes=hashes,
        provider_sha256=hashlib.sha256(Path(inspect.getsourcefile(andriy_spectral_features)).read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_spectral_features.py').read_bytes()).hexdigest(),
        octave_version=version,numpy_version=np.__version__,cases=cases,
        limitations=['Unmodified numerical source helpers in current Octave; no original MATLAB runtime claim.',
                     '82 direct-FFT spectral features only; Welch moments, other helpers and full model execution remain outstanding.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases_passed':len(cases),'maximum_error':max(c['maximum_absolute_error'] for c in cases)}))


if __name__ == '__main__':
    main()
