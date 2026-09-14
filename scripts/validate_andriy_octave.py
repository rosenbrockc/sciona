#!/usr/bin/env python3
"""Cross-language checks of source filtering and normalization on synthetic data."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import scipy
from scipy.io import savemat, loadmat
from sciona.atoms.riemannian_bci.signal_processing.andriy_filter import andriy_filter_trim
from sciona.atoms.riemannian_bci.signal_processing.andriy_normalization import andriy_joint_normalization

HASHES = {
    'Andriy_FE_main_F.m': 'b53474aefc981102cc39cd550cc7d1d63210825003cd046e396555588bff2515',
    'Andriy_code_normalise_tr_mv.m': 'a6bf8370a9aac17d28e676a1cb9f38d494d1a18b7b631358337c18561ad476c9',
    'Andriy_code_normalise_te_mv.m': '2164f3ee11d67c9a81064a52a7151fc3832c085533a8a44351ce308bb742742b',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--octave', default='/opt/homebrew/bin/octave-cli')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name, digest in HASHES.items():
        if hashlib.sha256((args.reference_dir/name).read_bytes()).hexdigest() != digest:
            raise ValueError('source drift: '+name)
    source = (args.reference_dir/'Andriy_FE_main_F.m').read_text()
    filtering = source[source.index('        % Highpass Filter coefficient'):source.index("        dataStruct.data = (resample")]
    cases = []
    with TemporaryDirectory(prefix='sciona-andriy-octave-') as temporary:
        directory = Path(temporary)
        for name in ['normalise_tr_mv', 'normalise_te_mv']:
            (directory/(name+'.m')).write_bytes((args.reference_dir/('Andriy_code_'+name+'.m')).read_bytes())
        script = "load('input.mat');\ndataStruct.data = segment;\ndataStruct.iEEGsamplingRate = fs;\n" + filtering + "\nfiltered = dataStruct.data;\n"
        script += "fit = [training' prediction(logical(valid),:)'];\n[unused, mu, sd] = normalise_tr_mv(fit);\nnormalized_training = normalise_te_mv(training',mu,sd);\nnormalized_prediction = normalise_te_mv(prediction',mu,sd);\nreference_version = version;\nsave('-mat7-binary','output.mat','filtered','normalized_training','normalized_prediction','reference_version');\n"
        (directory/'check.m').write_text(script)
        for seed, fs, duration in [(9811, 200, 9), (9812, 400, 9), (9813, 400, 600)]:
            rng = np.random.default_rng(seed)
            segment = rng.normal(size=(2, fs*duration))
            training, prediction = rng.normal(size=(21, 7)), rng.normal(size=(11, 7))
            valid = np.arange(11) % 3 != 0
            prediction[0, 0] = np.nan
            savemat(directory/'input.mat', dict(segment=segment, fs=float(fs), training=training, prediction=prediction, valid=valid))
            run = subprocess.run([args.octave, '--quiet', '--no-gui', 'check.m'], cwd=directory, capture_output=True, text=True, timeout=180)
            if run.returncode:
                raise RuntimeError(run.stderr)
            reference = loadmat(directory/'output.mat')
            actual = andriy_filter_trim(segment, fs)
            np.testing.assert_allclose(actual, reference['filtered'], rtol=1e-11, atol=1e-12)
            a, b = andriy_joint_normalization(training, prediction, valid)
            np.testing.assert_allclose(a, reference['normalized_training'], rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(b, reference['normalized_prediction'], rtol=1e-12, atol=1e-13)
            cases.append(dict(seed=seed, sampling_frequency=fs, duration_seconds=duration,
                              filter_max_abs_error=float(np.max(abs(actual-reference['filtered']))),
                              normalization_max_abs_error=float(max(np.max(abs(a-reference['normalized_training'])), np.nanmax(abs(b-reference['normalized_prediction']))))))
        reference_version = str(reference['reference_version'].reshape(-1)[0])
    report = dict(source_revision='00f937cc7710977dc812d9fc675864e2b8288658', source_hashes=HASHES,
        provider_hashes={f.__name__:hashlib.sha256(Path(inspect.getsourcefile(f)).read_bytes()).hexdigest() for f in [andriy_filter_trim, andriy_joint_normalization]},
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_versions={'numpy':np.__version__, 'scipy':scipy.__version__, 'octave':reference_version}, cases=cases,
        limitations=['Unmodified source numerical filter block and normalization function bodies run in current Octave, not original MATLAB.',
                     'Synthetic inputs only; no resampling, full feature extraction or model validation implied.',
                     'Source function/file name mismatches retained; Octave accepts these with warnings.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases_passed':len(cases),'octave_version':reference_version,'maximum_filter_error':max(c['filter_max_abs_error'] for c in cases)}))


if __name__ == '__main__':
    main()
