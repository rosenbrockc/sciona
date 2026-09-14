"""Read-only dependency resolution audit for the combined reference runtime."""
import base64
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


DECLARED_ROOTS = ['numpy', 'scipy', 'scikit-learn', 'xgboost', 'icontract']
IMPORT_REPORT = Path('docs/reviews/competition_flavours_import_origins.json')
OBSERVED_IMPORT_ROOTS = sorted(json.loads(IMPORT_REPORT.read_text())['packages'])
ROOTS = DECLARED_ROOTS + OBSERVED_IMPORT_ROOTS


def main():
    pending = [(name, frozenset()) for name in ROOTS]
    seen, packages, failures = set(), {}, []
    while pending:
        name, extras = pending.pop()
        name = canonicalize_name(name)
        state = (name, extras)
        if state in seen:
            continue
        seen.add(state)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            failures.append({'package': name, 'reason': 'missing'})
            continue
        selected = packages.setdefault(name, {'version': distribution.version, 'active_requirements': []})
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker and not any(requirement.marker.evaluate({'extra': extra}) for extra in ({''} | set(extras))):
                continue
            if value not in selected['active_requirements']:
                selected['active_requirements'].append(value)
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                failures.append({'package': name, 'requirement': value, 'reason': 'missing'})
                continue
            if installed not in requirement.specifier:
                failures.append({'package': name, 'requirement': value, 'installed': installed, 'reason': 'version mismatch'})
            pending.append((requirement.name, frozenset(requirement.extras)))
    report = {'approved': False, 'dependency_constraints_satisfied': not failures,
        'roots': ROOTS, 'declared_roots': DECLARED_ROOTS, 'observed_import_roots': OBSERVED_IMPORT_ROOTS,
        'python_version': sys.version.split()[0], 'packages': dict(sorted(packages.items())),
        'failures': failures,
        'scope': 'Installed numerical and observed shared-runner dependency constraint closure. Import observation covers startup and optimizer warmup, not every optional lazy branch. File origins, publisher authentication and license notices remain separate evidence.',
        'sha256': {'scripts/audit_flavours_environment.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), str(IMPORT_REPORT): hashlib.sha256(IMPORT_REPORT.read_bytes()).hexdigest()}}
    Path('docs/reviews/competition_flavours_environment_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'packages': len(packages), 'failures': failures}))


if __name__ == '__main__':
    main()
