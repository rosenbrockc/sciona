"""Acquire hash-verified public dependency wheels without installing them."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import requests


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    audit_path = Path('docs/reviews/competition_flavours_published_wheel_audit.json')
    audit = json.loads(audit_path.read_text())
    assert audit['all_wheel_pins_have_compatible_unyanked_wheels']
    root = Path('/private/tmp/sciona_flavours_publisher_wheels')
    root.mkdir(exist_ok=True)
    caches = [Path('/private/tmp/sciona_aptos_publisher_wheels'), Path('/private/tmp/sciona_cassava_publisher_wheels'), Path('/private/tmp/sciona_flavours_opencv_wheel')]
    def acquire(item):
        name, details = item
        artifact = details['selected_wheel']
        target = root / artifact['filename']
        try:
            cached = next((cache / artifact['filename'] for cache in caches if (cache / artifact['filename']).is_file()), root / 'no_cached_wheel')
            if target.exists():
                source = 'existing verified artifact'
            elif cached.is_file() and cached.stat().st_size == artifact['bytes'] and digest(cached) == artifact['sha256']:
                shutil.copyfile(cached, target)
                source = 'verified local publisher cache'
            else:
                partial = target.with_suffix('.partial')
                with requests.get(artifact['url'], stream=True, timeout=(30, 60)) as response:
                    response.raise_for_status()
                    with partial.open('wb') as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
                assert partial.stat().st_size == artifact['bytes'] and digest(partial) == artifact['sha256']
                partial.rename(target)
                source = 'publisher download'
            assert target.stat().st_size == artifact['bytes'] and digest(target) == artifact['sha256']
            print('VERIFIED', name, flush=True)
            return name, {'passed': True, 'source': source, 'sha256': artifact['sha256']}
        except Exception as error:
            return name, {'passed': False, 'error_type': type(error).__name__}
    with ThreadPoolExecutor(max_workers=4) as pool:
        packages = dict(pool.map(acquire, audit['packages'].items()))
    report = {'approved': False, 'catalog_mutations': 0,
        'all_wheels_acquired': all(row['passed'] for row in packages.values()), 'packages': packages,
        'scope': 'Public publisher artifacts acquired and hash checked; no installation or redistribution.',
        'sha256': {str(audit_path): digest(audit_path), str(Path(__file__).relative_to(Path.cwd())): digest(Path(__file__))}}
    Path('docs/reviews/competition_flavours_wheel_acquisition.json').write_text(json.dumps(report, indent=2) + '\n')
    print('ACQUISITION COMPLETE', report['all_wheels_acquired'], flush=True)


if __name__ == '__main__':
    main()
