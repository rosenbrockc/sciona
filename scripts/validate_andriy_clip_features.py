#!/usr/bin/env python3
"""Validate complete clip features with pinned extraction and assembly loops."""
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
from sciona.atoms.riemannian_bci.signal_processing.andriy_clip_features import andriy_documented_clip_features


def prepare_reference(directory, sources):
    for name,source in sources.items():
        if name.startswith('Andriy_code_'):
            (directory/name.removeprefix('Andriy_code_')).write_text(source)
    (directory/'nanmean.m').write_text("function y=nanmean(x,dim); if nargin<2; dim=find(size(x)~=1,1); if isempty(dim); dim=1; end; end; good=~isnan(x); x(~good)=0; y=sum(x,dim)./sum(good,dim); end;\n")
    (directory/'combnk.m').write_text("function y=combnk(x,k); y=nchoosek(x,k); end;\n")
    (directory/'pvset.m').write_text("function s=pvset(s,name,value); assert(strcmp(name,'a')); s=struct('a',value); end;\n")
    source=sources['Andriy_code_ar.m']
    block=source[source.index('% Now compute the regression matrix'):source.index('if ~pt\n')]
    prelude="function th=ar(data,n,approach); assert(n==9); maxsize=9999; Ne=1; pt1=1; approach=lower(approach); y={[zeros(n,1);data(:);zeros(n,1)]}; Ncaps=length(y{1}); th=cell(1,n);\n"
    (directory/'ar.m').write_text(prelude+block+'\nend;\n')
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
    source=sources['Andriy_code_compare_new.m']
    (directory/'compare_new.m').write_text(predictor+source[source.index('%Compute fit.'):]+ '\nend;\n')
    source=sources['Andriy_code_iwmf_and_bw_low.m']
    old='[fft_result,freq] = pwelch(data,[],[],N,Fs);'
    assert source.count(old)==1
    (directory/'iwmf_and_bw_low.m').write_text(source.replace(old,'[fft_result,freq] = pwelch(data,hamming(floor(length(data)/4.5)),floor(floor(length(data)/4.5)/2),N,Fs);'))
    script="pkg local_list '/private/tmp/sciona-octave/octave_packages'; pkg load signal; pwelch('R12+'); load('input.mat'); windL=30; windS=30; srateNew=256; dataStruct.data=signal256;\n"
    for filename,marker,stop_marker in [
        ('Andriy_FE_main_F.m','        feat = NaN','        %seq'),
        ('Andriy_FE_main_AR.m','        featAR = NaN','        if exist'),
        ('Andriy_FE_main_CONN.m','        featCon = NaN','        if exist')]:
        source=sources[filename]
        start=source.index(marker)
        stop=source.index(stop_marker,start)
        script+=source[start:stop]+'\n'
    script+='srateNew=128; dataStruct.data=csp128;\n'
    source=sources['Andriy_FE_main_CSP_AR.m']
    start=source.index('        featARCSP = NaN')
    script+=source[start:source.index('        if exist',start)]+'\n'
    source=sources['Andriy_data_preprocess.m']
    start=source.rindex('        clear tmp;')
    block=source[start:source.index('    newF_T(cur:end,:) = [];',start)]
    assert block.endswith('    end\n')
    script+='cur=1; goodidx=[]; newF_T=zeros(size(feat,2),1965);\n'+block[:-len('    end\n')]
    script+="\nvalid=false(size(feat,2),1); valid(goodidx)=true; reference_version=version; save('-mat7-binary','output.mat','newF_T','valid','reference_version');\n"
    (directory/'check.m').write_text(script)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    hashes=json.loads((root/'docs/reviews/andriy_clip_source_hashes.json').read_text())
    sources={}
    for name,sha in hashes.items():
        source=(args.reference_dir/name).read_bytes()
        if hashlib.sha256(source).hexdigest()!=sha:
            raise ValueError('source drift: '+name)
        sources[name]=source.decode()
    cases=[]
    with TemporaryDirectory(prefix='sciona-clip-features-') as temporary:
        directory=Path(temporary)
        prepare_reference(directory,sources)
        for seed,mode in [(651,'ordinary'),(652,'gates'),(653,'missing'),(654,'full_clip')]:
            rng=np.random.default_rng(seed)
            count=19 if mode=='full_clip' else 2
            signal=rng.normal(size=(16,count*7680+17))
            csp=rng.normal(size=(1,count*3840+7))
            if mode=='gates':
                signal[0,:7680]=np.linspace(-500,500,7680)
                signal[1,:7680]=0
                csp[0,:3840]*=1000
            elif mode=='missing':
                signal[:,:7680]=0
                signal[0,:]=0
                csp[0,:3840]=0
            savemat(directory/'input.mat',dict(signal256=signal,csp128=csp))
            run=subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','check.m'],cwd=directory,capture_output=True,text=True,timeout=600)
            if run.returncode:
                raise RuntimeError(run.stderr)
            reference=loadmat(directory/'output.mat')
            actual,valid=andriy_documented_clip_features(signal,csp)
            expected=reference['newF_T']
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
            np.testing.assert_array_equal(valid,reference['valid'].reshape(-1).astype(bool))
            finite=np.isfinite(actual)&np.isfinite(expected)
            print(json.dumps(dict(completed_case=mode,windows=count)),flush=True)
            cases.append(dict(seed=seed,mode=mode,windows=count,valid_windows=int(valid.sum()),
                maximum_finite_error=float(np.max(abs(actual[finite]-expected[finite])))))
    provider=Path(inspect.getsourcefile(andriy_documented_clip_features))
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_hashes=hashes,
        provider_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in provider.parent.glob('andriy_*.py')},
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_clip_features.py').read_bytes()).hexdigest(),
        octave_version=str(reference['reference_version'].reshape(-1)[0]),numpy_version=np.__version__,scipy_version=scipy.__version__,cases=cases,
        limitations=['Pinned caller loops, extraction helpers and assembly block; explicit documented Welch and least-squares AR reference substitutions.',
                     'Current SciPy/Octave filtering differs numerically; rtol/atol 1e-5 across all 1965 columns, NaN masks and validity exact.',
                     'Preprocessing/resampling, CSP fit/projection, training selection, joint normalization and models remain separate.',
                     'Historical MATLAB toolbox behavior and low-range RNG side effects remain unproven.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases_passed=len(cases),maximum_error=max(c['maximum_finite_error'] for c in cases))))


if __name__=='__main__':
    main()
