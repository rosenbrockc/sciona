#!/usr/bin/env python3
"""Record pinned source control-flow requirements without event data or IO."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def audit(root, source_dir):
    pins = json.loads((root/'docs/reviews/tracking_source_pins.json').read_text())
    trees, syntax_failures = {}, []
    for filename, sha in pins['source_files'].items():
        content = (source_dir/filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError('Source pin differs')
        if filename.endswith('.py'):
            try:
                trees[filename] = ast.parse(content)
            except SyntaxError as error:
                syntax_failures.append(dict(source_file=filename, line=error.lineno))
    if any(name not in trees for name in ['trackml_solution/algorithm.py', 'trackml_solution/geometry.py']):
        raise ValueError('Required tracking source does not parse')
    algorithm = next(n for n in trees['trackml_solution/algorithm.py'].body if isinstance(n, ast.ClassDef) and n.name == 'Algorithm')
    entry = next(n for n in algorithm.body if isinstance(n, ast.FunctionDef) and n.name == 'findTracks')
    computation = {'setupRun', 'chooseLikelyFirstHits', 'chooseLikelySecondHits', 'chooseLikelyNextHits',
        'findPairs', 'fitTracks', 'evaluateTracks', 'dropRedundantTracks', 'filterInvalidTrackCandidates',
        'nonPhysicalPostprocessOddHits'}
    observed = {n.func.attr for n in ast.walk(entry) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == 'self'}
    if not computation.issubset(observed):
        raise ValueError('Reviewed tracking entry flow changed')
    stages = [
        ('detector geometry', ['GeometrySpec', 'CylindersSpec', 'CapsSpec', 'DetectorSpec']),
        ('layer intersections', ['CylinderIntersector', 'CapIntersector', 'Intersector']),
    ]
    classes = {n.name for n in trees['trackml_solution/geometry.py'].body if isinstance(n, ast.ClassDef)}
    if any(not set(names).issubset(classes) for _, names in stages):
        raise ValueError('Reviewed geometry class inventory changed')
    return dict(approved=False, source_commit=pins['commit'], source_files_verified=len(pins['source_files']),
        source_entry='Algorithm.findTracks', required_computations=sorted(computation),
        current_python_syntax_failures=syntax_failures,
        source_geometry_classes={label: names for label, names in stages},
        missing_template_stage='Detector geometry construction and reachability preprocessing.',
        integration_requirements=[
            'Original detector geometry construction has fixed topology assumptions; exercise compatible synthetic geometry before claiming generic support.',
            'Layer selection includes finite cylinder lengths, cap radial gaps, premove distance, and optional corrections.',
            'Candidate seeds, repeated extension, ranking, pruning, and commitment must carry consistent state across rounds.',
            'Rebuilt neighborhoods must exclude committed observations in subsequent rounds.',
            'The new fixed-cylinder helper has a required validity mask and an explicit reverse-tangent correction; all consumers need corresponding handling.',
            'Optional calibrated layer functions and cell corrections need explicit configuration and evidence, not silent omission.',
        ],
        scope='Static source-flow audit only. No full execution, detector data, event data, scores or publication approval.',
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1], args.source_dir)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(source_files_verified=result['source_files_verified'], required_computations=len(result['required_computations']))))
