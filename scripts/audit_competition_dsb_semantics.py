"""Source-pinned synthetic counterexamples for competition implementation coverage.

Only explicitly selected public source code is read. No real data or checkpoints
are loaded. This diagnostic does not publish or approve any catalog artifact.
"""
import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def checked_source(root, pins, name):
    raw = (root / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pins['files'][name]:
        raise ValueError('Pinned source changed: ' + name)
    return raw.decode()


def source_oversampling(source, labels):
    start = source.index('            self.bboxes = []')
    end = source.index('\n\n        self.crop =', start)
    block = textwrap.dedent(source[start:end])
    state = SimpleNamespace()
    scope = dict(np=np, self=state, labels=labels, sizelim=6., sizelim2=30., sizelim3=40.)
    exec(compile(block, '<pinned-public-oversampling-loop>', 'exec'), scope)
    return state.bboxes


def audit(source_root, root):
    pins = json.loads((root / 'docs/reviews/competition_dsb_source_pins.json').read_text())
    sources = {name: checked_source(source_root, pins, name) for name in pins['files']}
    from sciona.visualizer.runner import _ensure_atoms_imported
    _ensure_atoms_imported()
    from sciona.atoms.dl.training.atoms import size_aware_nodule_oversampling, softmax_temperature_proposal_sampling

    # Coordinates and dimensions below are wholly synthetic, including boundary cases.
    sizes = np.array([5., 6., 7., 30., 31., 40., 41.])
    labels = np.zeros((len(sizes), 4))
    labels[:, 3] = sizes
    original = source_oversampling(sources['data_detector.py'], [labels])
    current = size_aware_nodule_oversampling(labels, diameter_column=3)
    source_counts = [int(np.sum(original[:, -1] == size)) for size in sizes]
    current_counts = [int(np.sum(current[:, -1] == size)) for size in sizes]
    if source_counts != [0, 0, 1, 1, 3, 3, 7]:
        raise ValueError('Reviewed source sampling behavior changed')

    # Execute the pinned source's probability construction with an RNG recorder.
    sampling = sources['data_classifier.py']
    block = sampling[sampling.index('def sampleone('):]
    capture = {}
    def choice(target, size, replace, p):
        capture['p'] = p.copy()
        return np.array([0])
    proxy = SimpleNamespace(**{name: getattr(np, name) for name in ['max', 'ones_like', 'sum', 'arange', 'exp']})
    proxy.random = SimpleNamespace(choice=choice)
    scope = dict(np=proxy)
    exec(compile(block, '<pinned-public-proposal-sampler>', 'exec'), scope)
    scores = np.array([0., -1000., -2000.])
    scope['sampleone']([0, 1, 2], scores, 1.)
    probability_floor_preserved = bool(np.all(capture['p'] > 0))
    try:
        softmax_temperature_proposal_sampling(scores, k=2, temperature=1., random_state=0)
        current_sampling_error = None
    except ValueError as error:
        current_sampling_error = str(error)

    providers = [root.parent / 'sciona-atoms-dl/src/sciona/atoms/dl' / part / 'atoms.py'
                 for part in ['detection', 'training', 'loss']]
    return dict(approved=False, read_only=True, source_version='6625a41e-b5a7-5ee2-86e5-1d2c60d6d185',
                source_commit=pins['commit'], source_files_verified=len(sources), synthetic_only=True,
                oversampling=dict(source_counts=source_counts, current_counts=current_counts,
                                  equivalent=source_counts == current_counts,
                                  source_loop_executed=True),
                sampling=dict(source_probability_floor_preserved=probability_floor_preserved,
                              current_error=current_sampling_error,
                              equivalent=False if current_sampling_error else 'not_established'),
                provider_sha256={str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in providers},
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = audit(args.source_root, root)
    (root / 'docs/reviews/competition_dsb_semantic_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
