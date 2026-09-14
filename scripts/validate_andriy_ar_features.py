#!/usr/bin/env python3
"""Check source AR feature flow using explicit state-space initial estimation."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import scipy
from scipy.io import savemat,loadmat
from sciona.atoms.riemannian_bci.signal_processing.andriy_ar_features import andriy_documented_ar_features
from sciona.atoms.riemannian_bci.signal_processing.andriy_ar_coefficients import andriy_ar_coefficients


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    hashes=json.loads((root/'docs/reviews/andriy_ar_feature_source_hashes.json').read_text())
    sources={}
    for name,sha in hashes.items():
        source=(args.reference_dir/('Andriy_code_'+name+'.m')).read_bytes()
        if hashlib.sha256(source).hexdigest()!=sha:
            raise ValueError('source drift')
        sources[name]=source.decode()
    cases=[]
    with TemporaryDirectory(prefix='sciona-ar-features-') as temporary:
        directory=Path(temporary)
        for name in ['ar_prediction_error','SPC_extract_featuresAR']:
            (directory/(name+'.m')).write_text(sources[name])
        (directory/'pvset.m').write_text("function s=pvset(s,name,value); assert(strcmp(name,'a')); s=struct('a',value); end;\n")
        block=sources['ar'][sources['ar'].index('% Now compute the regression matrix'):sources['ar'].index('if ~pt\n')]
        prelude="function th=ar(data,n,approach); assert(n==9); maxsize=9999; Ne=1; pt1=1; approach=lower(approach); y={[zeros(n,1);data(:);zeros(n,1)]}; Ncaps=length(y{1}); th=cell(1,n);\n"
        (directory/'ar.m').write_text(prelude+block+'\nend;\n')
        # Reference uses explicit shift-state propagation and solves for x0.
        # It does not call or emulate MATLAB's unavailable predict internals.
        predictor="""function fit=compare_new(data,model)
a=model.a; p=length(a)-1; F=zeros(p); F(2:p,1:p-1)=eye(p-1); K=zeros(p,1); K(1)=1; C=-a(2:end);
O=zeros(length(data),p); state=zeros(p,1); predicted=zeros(size(data)); power=eye(p);
for t=1:length(data)
 O(t,:)=C*power; power=power*F;
 predicted(t)=C*state; state=F*state+K*data(t);
end
s=svd(O); tol=p*eps*s(1); x0=pinv(O,tol)*(data-predicted);
state=x0; predicted=zeros(size(data));
for t=1:length(data)
 predicted(t)=C*state; state=F*state+K*data(t);
end
y={data}; yhh={predicted};
"""
        score=sources['compare_new'][sources['compare_new'].index('%Compute fit.'):]
        (directory/'compare_new.m').write_text(predictor+score+'\nend;\n')
        (directory/'check.m').write_text("load('input.mat'); result=zeros(rows(windows),9); for k=1:rows(windows); result(k,:)=SPC_extract_featuresAR(windows(k,:)')'; end; reference_version=version; save('-mat7-binary','output.mat','result','reference_version');\n")
        for seed in [631,632,633]:
            for samples in [41,3840,7680]:
                rng=np.random.default_rng(seed)
                x=rng.normal(size=(5,samples))
                x[1]=scipy.signal.lfilter([1.],[1.,-.8],x[1])
                x[2]=0
                x[3,:samples//2]=0
                x[4,-(samples//2):]=0
                savemat(directory/'input.mat',dict(windows=x))
                run=subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','check.m'],cwd=directory,capture_output=True,text=True,timeout=60)
                if run.returncode:
                    raise RuntimeError(run.stderr)
                reference=loadmat(directory/'output.mat')
                actual=andriy_documented_ar_features(x)
                np.testing.assert_allclose(actual,reference['result'],rtol=1e-8,atol=1e-8)
                cases.append(dict(seed=seed,samples=samples,
                    maximum_absolute_error=float(np.nanmax(abs(actual-reference['result'])))))
    providers=[Path(inspect.getsourcefile(f)) for f in [andriy_ar_coefficients,andriy_documented_ar_features]]
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_hashes=hashes,
        provider_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in providers},
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_ar_features.py').read_bytes()).hexdigest(),
        octave_version=str(reference['reference_version'].reshape(-1)[0]),numpy_version=np.__version__,scipy_version=scipy.__version__,cases=cases,
        initial_state_contract='Least-squares one-step prediction error; history-shift realization, pseudoinverse cutoff p*eps*smax.',
        documentation_url='https://www.mathworks.com/help/ident/ref/predictoptions.html',
        limitations=['Source window split, masks, sentinels, QR block and fit formula retained; toolbox model and prediction calls replaced explicitly.',
                     'State-space reference solves initial history then propagates states; provider projects onto the observable output subspace.',
                     'Historical MATLAB state realization, rank decisions, idprep and predict numerical behavior are unproven.',
                     'Finite-input contract excludes source NaN/inf repair. Low-range RNG side effects are not reproduced. Full model execution remains separate.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases_passed=len(cases),maximum_error=max(c['maximum_absolute_error'] for c in cases))))


if __name__=='__main__':
    main()
