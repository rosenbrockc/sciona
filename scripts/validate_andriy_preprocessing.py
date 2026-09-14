#!/usr/bin/env python3
"""Validate dual-rate clip preprocessing and source CSP candidate selection."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import scipy
from scipy.io import savemat,loadmat
from scipy.signal import firwin
from sciona.atoms.riemannian_bci.signal_processing.andriy_preprocessing import andriy_documented_preprocess_clip,andriy_csp_training_classes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'docs/reviews/andriy_clip_source_hashes.json').read_text())
    names=['Andriy_FE_main_F.m','Andriy_FE_main_CSP_AR.m']
    sources={}
    for name in names:
        data=(args.reference_dir/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=manifest[name]:
            raise ValueError('source drift')
        sources[name]=data.decode()
    source=sources[names[0]]
    filtering=source[source.index('        % Highpass Filter coefficient'):source.index('        dataStruct.data = (resample')]
    source=sources[names[1]]
    selection=source[source.index('        if min(range(dataStruct.data\')) > 0'):source.index('    filtcoef = csp')]
    assert selection.endswith('    end\n')
    selection=selection[:-len('    end\n')]
    cases=[]
    with TemporaryDirectory(prefix='sciona-andriy-preprocess-') as temporary:
        directory=Path(temporary)
        script="pkg local_list '/private/tmp/sciona-octave/octave_packages'; pkg load signal; load('input.mat'); dataStruct.data=segment; dataStruct.iEEGsamplingRate=fs;\n"+filtering
        script+="high=resample(dataStruct.data',p256,q256,b256)'; low=resample(dataStruct.data',p128,q128,b128)'; reference_version=version; save('-mat7-binary','output.mat','high','low','reference_version');\n"
        (directory/'preprocess.m').write_text(script)
        for seed,fs in [(691,128),(692,400)]:
            segment=np.random.default_rng(seed).normal(size=(16,600*fs))
            payload=dict(segment=segment,fs=float(fs))
            for target in [128,256]:
                divisor=math.gcd(target,fs);p,q=target//divisor,fs//divisor
                coefficients=np.ones(1) if p==q else p*firwin(20*max(p,q)+1,1/max(p,q),window=('kaiser',5.))
                payload.update({f'p{target}':float(p),f'q{target}':float(q),f'b{target}':coefficients})
            savemat(directory/'input.mat',payload)
            run=subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','preprocess.m'],cwd=directory,capture_output=True,text=True,timeout=300)
            if run.returncode: raise RuntimeError(run.stderr)
            reference=loadmat(directory/'output.mat')
            high,low=andriy_documented_preprocess_clip(segment,fs)
            np.testing.assert_allclose(high,reference['high'],rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(low,reference['low'],rtol=1e-11,atol=1e-12)
            cases.append(dict(stage='preprocessing',sampling_frequency=fs,duration_seconds=600,
                maximum_error=float(max(np.max(abs(high-reference['high'])),np.max(abs(low-reference['low']))))))
            print(json.dumps(cases[-1]),flush=True)
        script="load('selection.mat'); preD=[]; intD=[]; for i2=3:numel(candidates); dataStruct.sequence=sequences(i2); if dataStruct.sequence>1 && dataStruct.sequence<6; continue; end; dataStruct.data=candidates{i2};\n"+selection
        script+="\nend; save('-mat7-binary','selected.mat','preD','intD');\n"
        (directory/'select.m').write_text(script)
        for seed in [693,694,695]:
            rng=np.random.default_rng(seed)
            candidates=[rng.normal(size=(16,3840+i)) for i in range(8)]
            candidates[5][0]=0
            sequences=np.array([6,1,1,6,3,6,6,1])
            cells=np.empty((1,len(candidates)),dtype=object)
            for i,x in enumerate(candidates): cells[0,i]=x
            savemat(directory/'selection.mat',dict(candidates=cells,sequences=sequences))
            run=subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','select.m'],cwd=directory,capture_output=True,text=True,timeout=60)
            if run.returncode: raise RuntimeError(run.stderr)
            expected=loadmat(directory/'selected.mat')
            late,early=andriy_csp_training_classes(candidates,sequences)
            np.testing.assert_array_equal(late,expected['preD'])
            np.testing.assert_array_equal(early,expected['intD'])
            cases.append(dict(stage='selection',seed=seed,exact_class_inputs=True))
    provider=Path(inspect.getsourcefile(andriy_documented_preprocess_clip))
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_hashes={name:manifest[name] for name in names},
        provider_sha256={name:hashlib.sha256((provider.parent/name).read_bytes()).hexdigest() for name in ['andriy_preprocessing.py','andriy_filter.py','andriy_resampling.py']},
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_preprocessing.py').read_bytes()).hexdigest(),
        octave_version=str(reference['reference_version'].reshape(-1)[0]),numpy_version=np.__version__,scipy_version=scipy.__version__,cases=cases,
        limitations=['Source filtering and selection blocks; resampling uses explicit documented-design coefficients supplied to both runtimes.',
                     'Historical MATLAB default resampling is unproven; filesystem discovery/order must be supplied by caller.',
                     'Selection inputs are already preprocessed; full training/prediction pipeline integration remains separate.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
