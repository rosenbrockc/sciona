#!/usr/bin/env python3
"""Compare population grouping and normalization with pinned source arithmetic."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from scipy.io import savemat,loadmat
from sciona.atoms.riemannian_bci.signal_processing.andriy_population_inputs import andriy_population_inputs


HASHES={
    'Andriy_data_preprocess.m':'7270d94f782ccb0850a8b1241386e52232c3fb2cfb916d1602610b8afb15cf14',
    'Andriy_code_normalise_tr_mv.m':'a6bf8370a9aac17d28e676a1cb9f38d494d1a18b7b631358337c18561ad476c9',
    'Andriy_code_normalise_te_mv.m':'2164f3ee11d67c9a81064a52a7151fc3832c085533a8a44351ce308bb742742b',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    hashes=dict(HASHES)
    for name,sha in hashes.items():
        if hashlib.sha256((args.reference_dir/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('source drift')
    source=(args.reference_dir/'Andriy_data_preprocess.m').read_text()
    start=source.index('    nFeatI = round(nFeat/3);')
    block=source[start:source.index("    save(['train_ALL'",start)]
    cases=[]
    with TemporaryDirectory(prefix='sciona-population-inputs-') as temporary:
        directory=Path(temporary)
        for name in ['normalise_tr_mv','normalise_te_mv']:
            (directory/(name+'.m')).write_bytes((args.reference_dir/('Andriy_code_'+name+'.m')).read_bytes())
        prelude="load('input.mat'); nFeat=1965; newF_P=p(all(~isnan(p),2),:); newF_P1=p1(all(~isnan(p1),2),:); newF_I=n(all(~isnan(n),2),:); newF_T=t; goodidx=find(all(~isnan(t),2));\n"
        ending="\ntrain=[newF_P;newF_P1;newF_I]; prediction=newF_T; labels=(datalabels+1)/2; valid=false(rows(t),1); valid(goodidx)=true; reference_version=version; save('-mat7-binary','output.mat','train','prediction','labels','valid','reference_version');\n"
        (directory/'check.m').write_text(prelude+block+ending)
        for seed,auxiliary,constant in [(711,True,False),(712,False,False),(713,True,True)]:
            rng=np.random.default_rng(seed)
            groups=[[rng.normal(size=(19,1965)) for _ in range(count)] for count in [2,int(auxiliary),2,2]]
            groups[0][0][0,0]=np.nan
            groups[2][1][3,1]=np.nan
            groups[3][0][0,0]=np.nan
            if constant:
                for group in groups:
                    for x in group: x[:,10]=1
                groups[3][0][0,10]=2
            arrays=[np.concatenate(group,axis=0) if group else np.empty((0,1965)) for group in groups]
            savemat(directory/'input.mat',dict(zip(['p','p1','n','t'],arrays)))
            run=subprocess.run(['/opt/homebrew/bin/octave-cli','--quiet','--no-history','check.m'],cwd=directory,capture_output=True,text=True,timeout=60)
            if run.returncode: raise RuntimeError(run.stderr)
            reference=loadmat(directory/'output.mat')
            train,labels,prediction,valid=andriy_population_inputs(*groups)
            errors=[]
            for actual,name in [(train,'train'),(prediction,'prediction')]:
                expected=reference[name]
                np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-13)
                finite=np.isfinite(actual)&np.isfinite(expected)
                errors.append(float(np.max(abs(actual[finite]-expected[finite]))))
            np.testing.assert_array_equal(labels,reference['labels'].reshape(-1))
            np.testing.assert_array_equal(valid,reference['valid'].reshape(-1).astype(bool))
            cases.append(dict(seed=seed,auxiliary_group_present=auxiliary,constant_column=constant,
                retained_training_rows=len(labels),prediction_rows=len(valid),maximum_finite_error=max(errors)))
    provider=Path(inspect.getsourcefile(andriy_population_inputs))
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_hashes=hashes,
        provider_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [provider,provider.parent/'andriy_normalization.py']},
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_population_inputs.py').read_bytes()).hexdigest(),
        octave_version=str(reference['reference_version'].reshape(-1)[0]),numpy_version=np.__version__,cases=cases,
        limitations=['Already assembled clip features and caller-supplied safe group membership; no file discovery or membership inference.',
                     'Reference applies source non-NaN row criterion then runs unchanged three-chunk normalization and label construction.',
                     'Current Octave, not historical MATLAB; raw preprocessing/model pipeline integration remains separate.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases_passed=len(cases),maximum_error=max(c['maximum_finite_error'] for c in cases))))


if __name__=='__main__': main()
