#!/usr/bin/env python3
"""Compare the feature-selected R SVM branch with pinned source calculations."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from scipy.io import savemat,loadmat
from scipy.stats import rankdata
from sciona.atoms.ml.liblinear.andriy_r_svm import andriy_r_svm_segment_probabilities


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--library',required=True,type=Path)
    parser.add_argument('--rscript',default='/opt/homebrew/bin/Rscript')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source_bytes=(args.reference_dir/'Andriy_mod_svm_5_7.R').read_bytes()
    source=source_bytes.decode().replace('\r\n','\n')
    audit=json.loads((root/'docs/reviews/andriy_r_contract_audit.json').read_text())
    for name,sha in audit['compiled_library_sha256'].items():
        if hashlib.sha256((args.library/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('compiled R library drift: '+name)
    expected_sha=audit['source_sha256']['Andriy_mod_svm_5_7.R']
    if hashlib.sha256(source_bytes).hexdigest()!=expected_sha:
        raise ValueError('source drift')
    source=source[:source.index('submission <-')]
    source=source.replace('train.y <- data$datalabels','train.y <- as.numeric(data$datalabels)')
    source='\n'.join(line for line in source.splitlines() if not line.lstrip().startswith(('save(', 'testid =')))
    source=source.replace('  model1 <- xgb.train(', '  param$nthread <- 1\n  model1 <- xgb.train(')
    source=source.replace('importance <- xgb.importance(model = model1)', 'importance <- xgb.importance(model = model1, feature_names=as.character(seq_len(ncol(train))))\n  stopifnot(nrow(importance)>=300)\n  counts <- c(counts,nrow(importance))')
    sigmoid='  dec1_2 =  1/(1 + exp(-dec1_2$decisionValues[,1]))'
    assert source.count(sigmoid)==1
    source='all_windows <- NULL\ncounts <- NULL\n'+source.replace(sigmoid,sigmoid+'\n  all_windows <- c(all_windows,dec1_2)')
    source+="\nwriteMat('reference.mat',scores=testpreds2,windows=all_windows,counts=counts)\n"
    train_list=[];label_list=[];prediction_list=[];valid_list=[]
    env=dict(os.environ,R_LIBS_USER=str(args.library),SCIONA_RSCRIPT=args.rscript,OMP_NUM_THREADS='1')
    with TemporaryDirectory(prefix='sciona-andriy-r-svm-parity-') as temporary:
        directory=Path(temporary)
        (directory/'reference.R').write_text(source)
        for population in range(3):
            rng=np.random.default_rng(671+population)
            train=rng.normal(size=(512,1965))
            labels=np.r_[np.ones(256),np.zeros(256)]
            train[:,:1000]+=np.where(labels,.15,-.15)[:,None]
            prediction=rng.normal(size=(114,1965))
            prediction[:,:1000]+=np.repeat([-.8,.8,-.6,.6,-.8,0.],19)[:,None]
            valid=np.ones(114,dtype=bool)
            prediction[4*19,:1000]+=3
            valid[4*19]=False
            valid[5*19:]=False
            prediction[5*19:]=np.nan
            train_list.append(train);label_list.append(labels);prediction_list.append(prediction);valid_list.append(valid)
            savemat(directory/f'train_ALL{population+1}.mat',dict(newF_P=train[:128],newF_P1=train[128:256],newF_I=train[256:],
                datalabels=np.where(labels,1,-1)[None,:],newF_T=prediction,goodidx=(np.flatnonzero(valid)+1)[:,None]))
        # R.matlab maps underscores in MATLAB variable names to dots.
        run=subprocess.run([args.rscript,'--vanilla','reference.R'],cwd=directory,env=env,capture_output=True,text=True,timeout=1200)
        if run.returncode:
            raise RuntimeError(run.stderr)
        reference=loadmat(directory/'reference.mat')
        old={key:os.environ.get(key) for key in ['R_LIBS_USER','SCIONA_RSCRIPT']}
        try:
            os.environ.update({key:env[key] for key in old})
            scores,windows=andriy_r_svm_segment_probabilities(train_list,label_list,prediction_list,valid_list)
        finally:
            for key,value in old.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value
        np.testing.assert_array_equal(windows,reference['windows'].reshape(-1))
        np.testing.assert_array_equal(scores,reference['scores'].reshape(-1))
        mask=np.concatenate(valid_list)
        expected=np.where(mask,windows,0).reshape(-1,19).max(axis=1)
        np.testing.assert_array_equal(scores,expected)
        np.testing.assert_array_equal(scores[[5,11,17]],0)
        assert np.min(windows[[4*19,10*19,16*19]])>.8
        assert np.max(scores[[4,10,16]])<.2
        assert np.mean(scores[[1,3,7,9,13,15]])-np.mean(scores[[0,2,6,8,12,14]])>.5
        used_features=reference['counts'].reshape(-1).astype(int).tolist()
    provider=Path(inspect.getsourcefile(andriy_r_svm_segment_probabilities))
    report=dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658',source_sha256=expected_sha,
        provider_sha256=hashlib.sha256(provider.read_bytes()).hexdigest(),validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_r_svm.py').read_bytes()).hexdigest(),
        r_version=audit['r_version'],package_versions=audit['package_versions'],compiled_library_sha256=audit['compiled_library_sha256'],
        populations=3,selectors_per_population=1,rounds_per_selector=1000,selected_features=300,used_features=used_features,prediction_segments=18,exact_window_probabilities=True,exact_scores=True,
        score_spread=float(np.ptp(scores)),
        limitations=['Current pinned R engine; no historical 2016 XGBoost parity claim.',
                     'Source-shaped MATLAB row-vector labels explicitly converted to an R numeric vector for the current XGBoost API.',
                     'Source file I/O adapted to synthetic inputs; model saving/submission identifiers omitted; serial nthread=1 added.',
                     'Explicit one-based feature names supplied to current XGBoost importance API; source feature-index expression otherwise retained.',
                     'One three-population synthetic classification case; feature extraction, normalization and complete ensemble are separate.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(exact_scores=True,score_spread=report['score_spread'])))


if __name__=='__main__':
    main()
