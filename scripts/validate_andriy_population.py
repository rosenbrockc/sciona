#!/usr/bin/env python3
"""Compare raw population inputs with staged source preprocessing/extraction."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import scipy
from scipy.io import savemat,loadmat
from scipy.signal import firwin
from sciona.atoms.riemannian_bci.signal_processing.andriy_population import andriy_documented_population
from validate_andriy_clip_features import prepare_reference


def run_octave(directory,script,timeout=600):
    run=subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history',script],cwd=directory,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=timeout)
    if run.returncode:
        raise RuntimeError(run.stderr)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    hashes=json.loads((root/'docs/reviews/andriy_clip_source_hashes.json').read_text())
    csp_hashes=json.loads((root/'docs/reviews/andriy_csp_source_hashes.json').read_text())
    for name,sha in csp_hashes.items(): hashes['Andriy_code_'+name+'.m']=sha
    normalization=json.loads((root/'docs/reviews/andriy_population_inputs_parity.json').read_text())
    hashes.update(normalization['source_hashes'])
    sources={}
    for name,sha in hashes.items():
        data=(args.reference_dir/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=sha: raise ValueError('source drift: '+name)
        sources[name]=data.decode()
    fs=400
    rng=np.random.default_rng(721)
    candidates=[rng.normal(size=(16,600*fs))*np.linspace(1+i*.2,2+i*.1,16)[:,None] for i in range(4)]
    sequences=np.array([6,1,1,6])
    clips=[rng.normal(size=(16,600*fs)) for _ in range(4)]
    groups=[[clips[0]],[clips[1]],[clips[2]],[clips[3]]]
    with TemporaryDirectory(prefix='sciona-population-reference-') as temporary:
        directory=Path(temporary)
        for name in ['csp','spatFilt','normalise_tr_mv','normalise_te_mv']:
            (directory/(name+'.m')).write_text(sources['Andriy_code_'+name+'.m'])
        for index in range(4):
            child=directory/f'clip_{index+1}'
            child.mkdir()
            prepare_reference(child,sources)
        source=sources['Andriy_FE_main_F.m']
        filtering=source[source.index('        % Highpass Filter coefficient'):source.index('        dataStruct.data = (resample')]
        function="function [high,low]=source_preprocess(segment,fs,p256,q256,b256,p128,q128,b128); dataStruct.data=segment; dataStruct.iEEGsamplingRate=fs;\n"+filtering
        function+="high=resample(dataStruct.data',p256,q256,b256)'; low=resample(dataStruct.data',p128,q128,b128)'; end;\n"
        (directory/'source_preprocess.m').write_text(function)
        payload=dict(fs=float(fs),sequences=sequences)
        for name,values in [('candidates',candidates),('clips',clips)]:
            cells=np.empty((1,len(values)),dtype=object)
            for i,value in enumerate(values): cells[0,i]=value
            payload[name]=cells
        for target in [128,256]:
            divisor=math.gcd(target,fs);p,q=target//divisor,fs//divisor
            payload.update({f'p{target}':float(p),f'q{target}':float(q),f'b{target}':p*firwin(20*max(p,q)+1,1/max(p,q),window=('kaiser',5.))})
        savemat(directory/'raw.mat',payload)
        source=sources['Andriy_FE_main_CSP_AR.m']
        selected=source[source.index('        if min(range(dataStruct.data\')) > 0'):source.index('    filtcoef = csp')]
        assert selected.endswith('    end\n')
        script="pkg local_list '/private/tmp/sciona-octave/octave_packages'; pkg load signal; load('raw.mat'); preD=[]; intD=[];\nfor i2=3:numel(candidates); dataStruct.sequence=sequences(i2); if dataStruct.sequence>1 && dataStruct.sequence<6; continue; end; [unused,low]=source_preprocess(candidates{i2},fs,p256,q256,b256,p128,q128,b128); dataStruct.data=low;\n"
        script+=selected[:-len('    end\n')]+"\nend; filters=csp(preD,intD);\n"
        script+="for k=1:numel(clips); [signal256,low]=source_preprocess(clips{k},fs,p256,q256,b256,p128,q128,b128); csp128=spatFilt(low,filters,1); save('-mat7-binary',sprintf('clip_%d/input.mat',k),'signal256','csp128'); end;\n"
        (directory/'preprocess.m').write_text(script)
        run_octave(directory,'preprocess.m')
        print('Source preprocessing and CSP fit complete',flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending={pool.submit(run_octave,directory/f'clip_{i+1}','check.m'):i for i in range(4)}
            for future in as_completed(pending):
                future.result()
                print(f'Source clip {pending[future]+1}/4 complete',flush=True)
        source=sources['Andriy_data_preprocess.m']
        start=source.index('    nFeatI = round(nFeat/3);')
        block=source[start:source.index("    save(['train_ALL'",start)]
        script="load('clip_1/output.mat'); newF_P=newF_T(valid,:); load('clip_2/output.mat'); newF_P1=newF_T(valid,:); load('clip_3/output.mat'); newF_I=newF_T(valid,:); load('clip_4/output.mat'); goodidx=find(valid); nFeat=1965;\n"+block
        script+="train=[newF_P;newF_P1;newF_I]; prediction=newF_T; labels=(datalabels+1)/2; reference_version=version; save('-mat7-binary','expected.mat','train','labels','prediction','valid','reference_version');\n"
        (directory/'normalize.m').write_text(script)
        run_octave(directory,'normalize.m')
        reference=loadmat(directory/'expected.mat')
        actual=andriy_documented_population(candidates,sequences,*groups,fs)
        errors={}
        for value,name in [(actual[0],'train'),(actual[2],'prediction')]:
            expected=reference[name]
            np.testing.assert_allclose(value,expected,rtol=1e-4,atol=1e-4)
            finite=np.isfinite(value)&np.isfinite(expected)
            errors[name]=float(np.max(abs(value[finite]-expected[finite])))
        np.testing.assert_array_equal(actual[1],reference['labels'].reshape(-1))
        np.testing.assert_array_equal(actual[3],reference['valid'].reshape(-1).astype(bool))
        assert actual[0].shape==(57,1965) and actual[2].shape==(19,1965)
        assert np.all(np.isfinite(actual[0])) and np.all(np.isfinite(actual[2]))
    provider=Path(inspect.getsourcefile(andriy_documented_population))
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_hashes=hashes,
        provider_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in provider.parent.glob('andriy_*.py')},
        validation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'scripts/validate_andriy_clip_features.py']},
        test_sha256=hashlib.sha256((root/'tests/test_andriy_population.py').read_bytes()).hexdigest(),
        octave_version=str(reference['reference_version'].reshape(-1)[0]),numpy_version=np.__version__,scipy_version=scipy.__version__,
        source_sampling_frequency=fs,clip_duration_seconds=600,candidate_clips=4,group_clips=4,
        training_rows=57,prediction_rows=19,maximum_errors=errors,exact_labels=True,exact_validity=True,
        limitations=['One raw synthetic population; all four source group slots exercised. Group membership/file order supplied explicitly.',
                     'Documented resampling/Welch and estimated-state AR adaptations remain; no historical MATLAB equivalence claim.',
                     'Normalized feature tolerance 1e-4 reflects composition of source/SciPy filter differences; labels and validity exact.',
                     'R models, other populations and complete ensemble are not part of this validation.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(raw_population_passed=True,maximum_errors=errors)))


if __name__=='__main__': main()
