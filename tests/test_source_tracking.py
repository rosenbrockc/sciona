"""Synthetic original-flow recovery, isolation and fail-closed input tests."""
import sys
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.particle_tracking.source_tracking.atoms import find_tracks
from sciona.atoms.particle_tracking.source_tracking.loader import load_source_modules
from sciona.atoms.particle_tracking.source_tracking import loader
from scripts.validate_tracking_detector_geometry import synthetic_geometry


def fixture():
    source = load_source_modules()['trackml_solution.geometry']
    geometry = synthetic_geometry(source)
    modules = geometry.detectors_df[['volume_id', 'layer_id', 'cx', 'cy', 'cz', 'module_hv']].to_numpy()
    hits = np.column_stack([modules[:, :2], np.ones(len(modules)), modules[:, 2:5]])
    extra = []
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        for layer, radius in enumerate([20., 40., 60.]):
            phase = 2*np.arcsin(radius/400.)
            x, y = 200*np.sin(phase), 200*(1-np.cos(phase))
            extra.append([source.CylindersSpec.cylinder_volume_ids[0], layer, 1,
                          x*np.cos(angle)-y*np.sin(angle), x*np.sin(angle)+y*np.cos(angle), 10*phase])
    return modules, np.concatenate([hits, extra])


def assert_recovered(labels):
    assert labels.shape == (160,) and labels.dtype == np.int64
    assert (labels != 0).sum() == 48
    recovered_ids = []
    for start in range(136, 160, 3):
        identity = labels[start]
        assert identity > 0
        np.testing.assert_array_equal(np.flatnonzero(labels == identity), np.arange(start, start+3))
        recovered_ids.append(identity)
    assert len(set(recovered_ids)) == 8


@pytest.mark.parametrize('rounds,limit', [(1, 1000), (3, 1)])
def test_full_synthetic_recovery_and_input_preservation(rounds, limit):
    modules, hits = fixture()
    before = modules.copy(), hits.copy()
    labels = find_tracks(modules, hits, extension_steps=3, commitment_rounds=rounds, commitment_limit=limit)
    assert_recovered(labels)
    np.testing.assert_array_equal(modules, before[0])
    np.testing.assert_array_equal(hits, before[1])


def test_concurrent_invocations_do_not_share_source_state_or_global_apis():
    modules, hits = fixture()
    before = {k: v for k, v in sys.modules.items() if k.startswith(('trackml.', 'trackml_solution.'))}
    pandas_conversion = getattr(pd.DataFrame, 'as_matrix', None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda limit: find_tracks(modules, hits, extension_steps=3,
                            commitment_rounds=3, commitment_limit=limit), [1, 1000]))
    for labels in results:
        assert_recovered(labels)
    after = {k: v for k, v in sys.modules.items() if k.startswith(('trackml.', 'trackml_solution.'))}
    assert before == after
    assert getattr(pd.DataFrame, 'as_matrix', None) is pandas_conversion


@pytest.mark.parametrize('argument,value', [
    ('extension_steps', 1), ('extension_steps', 101), ('extension_steps', 3.0),
    ('extension_steps', True), ('commitment_rounds', 0), ('commitment_rounds', 101),
    ('commitment_limit', 0), ('commitment_limit', 1000001)])
def test_invalid_options(argument, value):
    with pytest.raises(ValueError, match='integer tracking option'):
        find_tracks(np.ones((1, 6)), np.ones((1, 6)), **{argument: value})


@pytest.mark.parametrize('bad', [np.zeros((0, 6)), np.zeros((2, 5)), np.zeros(6),
                               np.full((2, 6), np.nan), np.full((2, 6), np.inf),
                               np.ones((2, 6), dtype=complex), np.ones((2, 6), dtype=bool)])
@pytest.mark.parametrize('position', [0, 1])
def test_invalid_matrices(bad, position):
    args = [np.ones((2, 6)), np.ones((2, 6))]
    args[position] = bad
    with pytest.raises(ValueError):
        find_tracks(*args)


@pytest.mark.parametrize('position,column,value', [(0, 0, .5), (1, 1, -1), (1, 2, 32768),
                                                   (0, 5, 0), (1, 3, 1e100)])
def test_invalid_field_values(position, column, value):
    args = [np.ones((2, 6)), np.ones((2, 6))]
    args[position][0, column] = value
    with pytest.raises(ValueError):
        find_tracks(*args)


def test_modules_are_distinct_with_independent_defaults():
    one, two = load_source_modules(), load_source_modules()
    name = 'trackml_solution.algorithm'
    one[name].Algorithm.default_params['commit__nmax'] = -99
    assert two[name].Algorithm.default_params['commit__nmax'] == 1000


def test_default_iteration_count():
    assert_recovered(find_tracks(*fixture()))


@pytest.mark.parametrize('tamper', ['manifest', 'source', 'source_and_manifest'])
def test_packaged_integrity_fails_closed(tmp_path, monkeypatch, tamper):
    import hashlib
    directory = Path(loader.__file__).parent
    for p in directory.iterdir():
        if p.is_file():
            shutil.copyfile(p, tmp_path/p.name)
    monkeypatch.setattr(loader, '__file__', str(tmp_path/'loader.py'))
    manifest_path = tmp_path/'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    name = loader.ORDER[-1]
    if tamper in ['source', 'source_and_manifest']:
        source_path = tmp_path/name
        source_path.write_bytes(source_path.read_bytes()+b'\n# changed\n')
    if tamper == 'source_and_manifest':
        manifest[name]['sha256'] = hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    elif tamper == 'manifest':
        manifest_path.write_bytes(manifest_path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='integrity check failed'):
        loader.load_source_modules()
