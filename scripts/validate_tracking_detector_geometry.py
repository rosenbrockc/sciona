#!/usr/bin/env python3
"""Exercise pinned detector construction using wholly synthetic module geometry."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd


def synthetic_geometry(source):
    # These generated positions are mathematical fixtures, not copied modules,
    # events, observations or configuration from any detector dataset.
    rows = []
    axes = [(1., 0.), (0., 1.), (-1., 0.), (0., -1.)]
    for layer, radius in enumerate([20., 40., 60.]):
        for z in [-radius/4, radius/4]:
            for ux, uy in axes:
                rows.append(dict(volume_id=source.CylindersSpec.cylinder_volume_ids[0], layer_id=layer,
                    cx=radius*ux, cy=radius*uy, cz=z, module_hv=1.))
    for layer, z in enumerate([-50., -30., 30., 50.]):
        for radius in [10., 15., 18., 30., 33., 50., 53.]:
            for ux, uy in axes:
                rows.append(dict(volume_id=source.CapsSpec.cap_volume_ids[0], layer_id=layer,
                    cx=radius*ux, cy=radius*uy, cz=z, module_hv=2.))
    geometry = source.GeometrySpec()
    geometry.detectors_df = pd.DataFrame(rows)
    return geometry


def validate(root, source_dir):
    pins = json.loads((root/'docs/reviews/tracking_source_pins.json').read_text())
    path = source_dir/'trackml_solution/geometry.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != pins['source_files']['trackml_solution/geometry.py']:
        raise ValueError('Source hash differs')
    spec = importlib.util.spec_from_file_location('reviewed_detector_geometry', path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    geometry = synthetic_geometry(source)
    before = geometry.detectors_df.copy(deep=True)
    detector = source.DetectorSpec(geometry)
    pd.testing.assert_frame_equal(geometry.detectors_df, before)
    np.testing.assert_allclose(detector.cylinders.cyl_rsqr, [400., 1600., 3600.])
    np.testing.assert_allclose(detector.cylinders.cyl_absz_max, [6., 11., 16.])
    np.testing.assert_allclose(detector.caps.cap_z, [-50., -30., 30., 50.])
    np.testing.assert_allclose(detector.caps.ring_gap_r2_min, [12., 20., 35.])
    np.testing.assert_allclose(detector.caps.ring_gap_r2_max, [13., 28., 48.])
    np.testing.assert_allclose(detector.caps.ring_overlap_r2_min, [16., 31., 51.])
    np.testing.assert_allclose(detector.caps.ring_overlap_r2_max, [17., 32., 52.])
    # Reordering module rows must preserve the generated geometry.
    shuffled = source.GeometrySpec()
    shuffled.detectors_df = before.sample(frac=1., random_state=2).reset_index(drop=True)
    repeated = source.DetectorSpec(shuffled)
    for a, b in [(detector.cylinders.cyl_rsqr, repeated.cylinders.cyl_rsqr),
                 (detector.caps.cap_z, repeated.caps.cap_z),
                 (detector.caps.ring_gap_r2_min, repeated.caps.ring_gap_r2_min)]:
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    # Use the constructed detector in the unmodified original intersector.
    arrays = [np.array([value]) for value in [5., 0., 0., 1., 15., 0., 10., 20.]]
    intersection = source.Intersector(detector).findNextHelixIntersection(*arrays,
        cyl_pre_move=1e-6, cap_pre_move=1e-6, missable=True)
    x, y, z, cylinder, layer, phase, distance, _ = intersection
    if not cylinder[0] or layer[0] != 0 or not (distance[0] > 0 and np.isfinite(distance[0])):
        raise ValueError('Expected first synthetic cylinder crossing')
    np.testing.assert_allclose(x*x+y*y, [400.], rtol=1e-12)
    if abs(z[0]) > detector.cylinders.cyl_absz_max[0] or phase[0] <= 0:
        raise ValueError('Crossing violates finite layer or direction')
    return dict(approved=False, synthetic_only=True, source_commit=pins['commit'],
        source_sha256=pins['source_files']['trackml_solution/geometry.py'],
        checks=dict(detector_construction=True, cylinder_radii_and_lengths=True, cap_clustering=True,
                    cap_gaps_and_overlaps=True, input_preserved=True, row_permutation_invariant=True,
                    combined_intersector_crossing=True),
        scope='Compatible generated geometry and one finite-layer intersection. No actual detector data, complete tracking execution or publication approval.',
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(Path(__file__).resolve().parents[1], args.source_dir)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['checks']))
