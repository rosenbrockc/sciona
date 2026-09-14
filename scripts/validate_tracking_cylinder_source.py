#!/usr/bin/env python3
"""Compare corrected geometry to pinned original source on synthetic cylinders."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sciona.visualizer.runner import _ensure_atoms_imported


def validate(root, source_dir):
    pins = json.loads((root/'docs/reviews/tracking_source_pins.json').read_text())
    path = source_dir/'trackml_solution/geometry.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != pins['source_files']['trackml_solution/geometry.py']:
        raise ValueError('Source geometry pin differs')
    spec = importlib.util.spec_from_file_location('reviewed_tracking_geometry', path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    _ensure_atoms_imported()
    from sciona.atoms.particle_tracking.track_matching import cylinder_geometry as provider
    compared, missing = 0, 0
    maximum = 0.
    for target in [2., 5., 8., 16.]:
        intersector = source.CylinderIntersector(SimpleNamespace(cyl_rsqr=np.array([target])))
        for direction in [-1., 1.]:
            for initial in [.2, 1., 3., 5.]:
                values = {k: np.array([v]) for k, v in dict(x0=2+np.cos(initial), y0=np.sin(initial), z0=0.,
                    hel_xm=2., hel_ym=0., hel_r=1., hel_pitch=2*np.pi, sign_uz=direction, target_r2sqr=target).items()}
                actual = provider.next_cylinder_intersection(**values)
                expected = intersector.intersectHelices(values['x0'], values['y0'], values['z0'], values['sign_uz'],
                    values['hel_xm'], values['hel_ym'], values['hel_r'], values['hel_pitch'], pre_move=1e-6, missable=False)
                valid = expected[3] >= 0
                np.testing.assert_array_equal(actual[4], valid)
                if valid[0]:
                    for a, b in zip(actual[:4], [*expected[:3], expected[4]]):
                        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
                        maximum = max(maximum, float(np.max(np.abs(a-b))))
                    compared += 1
                else:
                    if not all(np.isnan(a).all() for a in expected[:3]):
                        raise ValueError('Source missing-intersection semantics differ')
                    if not all(np.array_equal(a, [0.]) for a in actual[:4]):
                        raise ValueError('Corrected missing-intersection placeholders differ')
                    missing += 1
    values = {k: np.array([v]) for k, v in dict(x0=3., y0=0., z0=0., hel_xm=2., hel_ym=0., hel_r=1.,
        hel_pitch=2*np.pi, sign_uz=-1., target_r2sqr=9.).items()}
    actual = provider.next_cylinder_intersection(**values)
    expected = source.CylinderIntersector(SimpleNamespace(cyl_rsqr=np.array([9.]))).intersectHelices(
        values['x0'], values['y0'], values['z0'], values['sign_uz'], values['hel_xm'], values['hel_ym'],
        values['hel_r'], values['hel_pitch'], pre_move=1e-6, missable=False)
    if expected[3][0] != -1 or not actual[4][0]:
        raise ValueError('Reviewed reverse-tangent discrepancy changed')
    np.testing.assert_allclose(actual[3], [-2*np.pi], rtol=1e-8)
    return dict(approved=False, synthetic_only=True, source_commit=pins['commit'], source_sha256=pins['source_files']['trackml_solution/geometry.py'],
        compared_reachable_cases=compared, compared_unreachable_cases=missing, maximum_absolute_error=maximum,
        reverse_tangent_correction='Pinned source reports no next intersection; corrected geometry advances one negative turn, validated analytically.',
        missing_intersection_mapping='Source NaN coordinates/-1 layer/infinite arc length become explicit false validity mask with ignored zero placeholders.',
        scope='One synthetic cylinder, pre_move=1e-6, missable=False, no corrector. No finite-layer/gap/complete-track parity claim.',
        provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(Path(__file__).resolve().parents[1], args.source_dir)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
