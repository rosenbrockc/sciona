#!/usr/bin/env python3
"""Compare multi-order coefficients with the pinned source QR block in Octave."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from scipy.io import savemat,loadmat
from sciona.atoms.riemannian_bci.signal_processing.andriy_ar_coefficients import andriy_ar_coefficients


SOURCE_SHA='fdf46f8a38076e51809c2f0673458f6067a501a23318e3ee893b1ea39c2db61a'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=(args.reference_dir/'Andriy_code_ar.m').read_bytes()
    if hashlib.sha256(source).hexdigest()!=SOURCE_SHA:
        raise ValueError('source drift')
    text=source.decode()
    block=text[text.index('% Now compute the regression matrix'):text.index('if ~pt\n')]
    cases=[]
    with TemporaryDirectory(prefix='sciona-ar-coefficients-') as temporary:
        directory=Path(temporary)
        # Only the model property setter is substituted. The source QR,
        # pseudoinverse, reversal and all nine polynomial assignments are intact.
        (directory/'pvset.m').write_text("function s=pvset(s,name,value); assert(strcmp(name,'a')); s=struct('a',value); end;\n")
        prelude="function out=source_numerical(data,maxsize); n=9; Ne=1; pt1=1; approach='yw'; y={[zeros(n,1);data(:);zeros(n,1)]}; Ncaps=length(y{1}); th=cell(1,n);\n"
        ending="\nout=zeros(9,10); for q=1:9; out(q,1:q+1)=th{q}.a; end; end;\n"
        (directory/'source_numerical.m').write_text(prelude+block+ending)
        (directory/'check.m').write_text("load('input.mat'); chunks=[17,1111,10000]; result=zeros(rows(windows),9,10,3); for j=1:3; for k=1:rows(windows); result(k,:,:,j)=source_numerical(windows(k,:)',9*chunks(j)); end; end; reference_version=version; save('-mat7-binary','output.mat','result','reference_version');\n")
        for seed in [611,612,613]:
            for samples in [30,1920,3840]:
                rng=np.random.default_rng(seed)
                x=rng.normal(size=(3,samples))
                x[1]+=4
                x[2]=0
                savemat(directory/'input.mat',dict(windows=x))
                run=subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','check.m'],cwd=directory,capture_output=True,text=True,timeout=60)
                if run.returncode:
                    raise RuntimeError(run.stderr)
                reference=loadmat(directory/'output.mat')
                actual=andriy_ar_coefficients(x)
                for chunk_index,chunk in enumerate([17,1111,10000]):
                    expected=reference['result'][:,:,:,chunk_index]
                    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-11)
                    cases.append(dict(seed=seed,samples=samples,qr_chunk_rows=chunk,
                        maximum_absolute_error=float(np.max(abs(actual-expected)))))
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_sha256=SOURCE_SHA,
        extracted_block_sha256=hashlib.sha256(block.encode()).hexdigest(),
        provider_sha256=hashlib.sha256(Path(inspect.getsourcefile(andriy_ar_coefficients)).read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_ar_coefficients.py').read_bytes()).hexdigest(),
        octave_version=str(reference['reference_version'].reshape(-1)[0]),numpy_version=np.__version__,cases=cases,
        limitations=['Numerical QR block only; explicit zero-padding and numeric single-series setup replace iddata/idprep preprocessing.',
                     'Model polynomial property setter replaced with a struct setter; model metadata and covariance handling outside the block are not validated.',
                     'Three explicit QR chunk lengths tested; historical automatic idmsize choice unproven.',
                     'No System Identification Toolbox prediction, initial-state estimation or fit-score parity claim.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases_passed=len(cases),maximum_error=max(c['maximum_absolute_error'] for c in cases))))


if __name__=='__main__':
    main()
