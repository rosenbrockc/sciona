#!/usr/bin/env python3
"""Compare the Feng filtering provider with the pinned public source function."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import scipy
import scipy.signal
from sciona.atoms.riemannian_bci.signal_processing.feng_filter import feng_resample_filter

SOURCE_SHA = 'c9981942933fed2afaee9d9e3be90ea24d7167a3d08eeaf7e668e245bc822167'


def load_reference(reference_dir):
    path = reference_dir / 'Feng_preprocessors_filtering.py'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError('pinned filtering source differs')
    # Only the numerical function is loaded: file IO and Python 2 orchestration
    # are excluded. Its body requires no compatibility or algorithm changes.
    body = data.decode().split('def filter(', 1)[1].split('def process_file_filter', 1)[0]
    namespace = {'np': np, 'sc': scipy, 'scipy': scipy}
    exec(compile('def filter(' + body, str(path), 'exec'), namespace)
    return namespace['filter']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = load_reference(args.reference_dir)
    cases = []
    for seed in [8121, 8122, 8123, 8124, 8125]:
        for duration in [6, 600]:
            rng = np.random.default_rng(seed)
            x = rng.normal(size=(200 * duration + 13, 2))
            before = x.copy()
            expected = reference(x.astype(np.float32), 400, duration, .1, 180)
            actual = feng_resample_filter(x, duration)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(x, before)
            cases.append(dict(seed=seed, duration_seconds=duration, exact=True,
                              max_abs_error=float(np.max(np.abs(actual-expected)))))
    root = Path(__file__).resolve().parents[1]
    provider = Path(inspect.getsourcefile(feng_resample_filter))
    tests = root / 'tests/test_feng_resample_filter.py'
    report = dict(source_commit='00f937cc7710977dc812d9fc675864e2b8288658',
        source_sha256=SOURCE_SHA, provider_sha256=hashlib.sha256(provider.read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(tests.read_bytes()).hexdigest(),
        libraries={'numpy': np.__version__, 'scipy': scipy.__version__}, cases=cases,
        cases_passed=len(cases), limitations=[
            'Unmodified numerical source body with current NumPy/SciPy; not a historical library environment.',
            'Synthetic signals only; no clinical or predictive-performance claim.',
            'Explicit raw-input float32 conversion matches source file loader; numerical parity covers resampling and zero-state causal filtering.',
            'Provider implementation evidence only; complete model graph and catalog approval remain outstanding.'])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'cases_passed': len(cases), 'all_exact': True}))


if __name__ == '__main__':
    main()
