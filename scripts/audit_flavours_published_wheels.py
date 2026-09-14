"""Check pinned dependencies against compatible publisher-index wheel metadata."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def main():
    audit_path = Path('docs/reviews/competition_flavours_environment_audit.json')
    audit = json.loads(audit_path.read_text())
    ranks = {tag: index for index, tag in enumerate(sys_tags())}
    python = sys.version.split()[0]
    def inspect(item):
        name, selected = item
        version = selected['version']
        url = f'https://pypi.org/pypi/{name}/{version}/json'
        try:
            with urlopen(url, timeout=45) as response:
                data = json.load(response)
            candidates, rejected = [], []
            for artifact in data['urls']:
                if artifact['packagetype'] != 'bdist_wheel':
                    continue
                _, _, _, tags = parse_wheel_filename(artifact['filename'])
                compatible = tags & ranks.keys()
                if not compatible:
                    continue
                if artifact.get('requires_python') and python not in SpecifierSet(artifact['requires_python']):
                    continue
                if artifact.get('yanked'):
                    rejected.append({'filename': artifact['filename'], 'reason': artifact.get('yanked_reason')})
                    continue
                candidates.append((min(ranks[tag] for tag in compatible), artifact))
            candidates.sort(key=lambda entry: (entry[0], entry[1]['filename']))
            result = {'version': version, 'index_url': url, 'compatible_unyanked_wheels': len(candidates),
                      'compatible_yanked_wheels': rejected}
            if candidates:
                artifact = candidates[0][1]
                result['selected_wheel'] = {'filename': artifact['filename'], 'url': artifact['url'],
                    'sha256': artifact['digests']['sha256'], 'bytes': artifact['size'],
                    'requires_python': artifact.get('requires_python')}
            return name, result
        except Exception as error:
            return name, {'version': version, 'index_url': url, 'error_type': type(error).__name__}
    with ThreadPoolExecutor(max_workers=4) as pool:
        packages = dict(pool.map(inspect, [(name, row) for name, row in audit['packages'].items()]))
    unresolved = [name for name, details in packages.items() if 'selected_wheel' not in details]
    report = {'approved': False, 'all_wheel_pins_have_compatible_unyanked_wheels': not unresolved,
        'python_version': python, 'packages': packages, 'unresolved': unresolved,
        'source_only_packages': {},
        'scope': 'Compatible wheel availability and publisher-index hashes only; wheel downloads, installed-file comparison to publisher wheels and fresh installation are not yet verified.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
            str(audit_path), 'scripts/audit_flavours_published_wheels.py']}}
    Path('docs/reviews/competition_flavours_published_wheel_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'packages': len(packages), 'unresolved': {name: packages[name] for name in unresolved}}))


if __name__ == '__main__':
    main()
