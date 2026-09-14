#!/usr/bin/env python3
"""Run pinned MATLAB feature assembly in Octave on synthetic feature tensors."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from scipy.io import savemat, loadmat
from sciona.atoms.riemannian_bci.signal_processing.andriy_feature_assembly import andriy_assemble_clip_features

HASHES = {
    'Andriy_data_preprocess.m': '7270d94f782ccb0850a8b1241386e52232c3fb2cfb916d1602610b8afb15cf14',
    'Andriy_code_mean_nan.m': 'f3998cbc9dd08b97e9c827b07712de1fe5e338af8ec46fb6bf1a73b80bfa5adf',
    'Andriy_code_sumskipnan.m': 'ac6cf310c625b7c53f850d9994491735d1fa2967a5289bb942b4435033849b3e',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--octave', default='/opt/homebrew/bin/octave-cli')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    for name, sha in HASHES.items():
        if hashlib.sha256((args.reference_dir/name).read_bytes()).hexdigest() != sha:
            raise ValueError('source drift: '+name)
    source = (args.reference_dir/'Andriy_data_preprocess.m').read_text()
    start = source.rindex('        clear tmp;')
    stop = source.index('    newF_T(cur:end,:) = [];', start)
    block = source[start:stop]
    assert block.endswith('    end\n')
    block = block[:-len('    end\n')]  # enclosing file-iteration loop is supplied synthetically
    cases = []
    with TemporaryDirectory(prefix='sciona-andriy-assembly-') as temporary:
        directory = Path(temporary)
        for name in ['mean_nan', 'sumskipnan']:
            (directory/(name+'.m')).write_bytes((args.reference_dir/('Andriy_code_'+name+'.m')).read_bytes())
        script = "load('input.mat');\ncur=1; goodidx=[]; newF_T=zeros(size(feat,2),1965);\n"+block
        script += "\nvalid=false(size(feat,2),1); valid(goodidx)=true; reference_version=version;\nsave('-mat7-binary','output.mat','newF_T','valid','reference_version');\n"
        (directory/'check.m').write_text(script)
        for seed, windows, missing in [(9911, 3, False), (9912, 19, False), (9913, 19, True)]:
            rng = np.random.default_rng(seed)
            main = rng.uniform(0, 5, size=(16, windows, 102))
            ar, csp, con = rng.normal(size=(16, windows, 9)), rng.normal(size=(2, windows, 9)), rng.normal(size=(windows, 180))
            main[0, 1, 20] = np.nan; ar[1, 1, 2] = np.nan; csp[0, 1, 1] = np.nan; con[1, 1] = np.nan
            main[1, :, 21] = -1
            if missing:
                ar[0, :, 0] = np.nan
            savemat(directory/'input.mat', dict(feat=main, featAR=ar, featARCSP=csp, featCon=con))
            run = subprocess.run([args.octave, '--quiet', '--no-gui', '--no-history', 'check.m'], cwd=directory, capture_output=True, text=True, timeout=90)
            if run.returncode:
                raise RuntimeError(run.stderr)
            reference = loadmat(directory/'output.mat')
            actual, valid = andriy_assemble_clip_features(main, ar, csp, con)
            expected = reference['newF_T']
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
            np.testing.assert_array_equal(valid, reference['valid'].reshape(-1).astype(bool))
            finite = np.isfinite(actual) & np.isfinite(expected)
            cases.append(dict(seed=seed, windows=windows, wholly_missing_feature=missing, validity_exact=True,
                              maximum_finite_error=float(np.max(abs(actual[finite]-expected[finite])))))
        version = str(reference['reference_version'].reshape(-1)[0])
    root = Path(__file__).resolve().parents[1]
    report = dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658', source_hashes=HASHES,
        provider_sha256=hashlib.sha256(Path(inspect.getsourcefile(andriy_assemble_clip_features)).read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((root/'tests/test_andriy_feature_assembly.py').read_bytes()).hexdigest(),
        octave_version=version, numpy_version=np.__version__, cases=cases,
        limitations=['Unchanged per-clip source prediction assembly block and NaN helpers run in current Octave, not original MATLAB.',
                     'Synthetic inputs cover imputation, log transform, column-major assembly and validity; feature extraction/model execution remains outstanding.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases_passed':len(cases), 'maximum_finite_error':max(c['maximum_finite_error'] for c in cases)}))


if __name__ == '__main__':
    main()
