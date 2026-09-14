"""Candidate dependency pins and bundled notice inventory; no installations."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path


def main():
    audit_path = Path('docs/reviews/competition_cassava_environment_audit.json')
    audit = json.loads(audit_path.read_text())
    assert audit['dependency_constraints_satisfied'] and audit['declared_headless_binary_matches']
    for path, expected in audit['sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    packages = {}
    for name, selected in audit['packages'].items():
        distribution = metadata.distribution(name)
        assert distribution.version == selected['version']
        license_files = {}
        for item in distribution.files or []:
            leaf = Path(str(item)).name.lower()
            if (any(word in leaf for word in ['license', 'licence', 'copying', 'notice'])
                    and ('dist-info/' in str(item) or len(Path(str(item)).parts) <= 3)):
                path = Path(distribution.locate_file(item))
                if path.is_file():
                    license_files[str(item)] = hashlib.sha256(path.read_bytes()).hexdigest()
        packages[name] = {'version': distribution.version,
            'license_expression': distribution.metadata.get('License-Expression'),
            'license_classifiers': [value for value in distribution.metadata.get_all('Classifier', []) if value.startswith('License ::')],
            'license_files': license_files}
    requirements = Path('requirements/cassava-reference.txt')
    lines = ['# Candidate combined reference environment; publication qualification pending.',
             '# Python 3.13 / macOS arm64 dependency resolution observed by audit_cassava_environment.py.']
    lines += [f'{name}=={details["version"]}' for name, details in packages.items()]
    requirements.write_text('\n'.join(lines) + '\n')
    paths = [str(requirements), str(audit_path), 'scripts/inventory_cassava_dependencies.py']
    report = {'approved': False,
        'scope': 'Candidate dependency notice inventory; declarations and bundled notices still require review before publication. No dependency installation or active environment mutation performed.',
        'packages': packages, 'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_cassava_dependency_notice_inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'packages': len(packages), 'notice_files': sum(len(item['license_files']) for item in packages.values()),
                      'without_bundled_notices': [name for name, item in packages.items() if not item['license_files']]}))


if __name__ == '__main__':
    main()
