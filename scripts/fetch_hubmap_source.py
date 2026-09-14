"""Retrieve only hash-pinned public software for the HuBMAP source comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(root, destination):
    pins = json.loads((root / 'docs/reviews/competition_hubmap_source_pins.json').read_text())
    if pins['repository'] != 'https://github.com/tikutikutiku/kaggle-hubmap':
        raise ValueError('Unexpected software repository')
    downloaded = 0
    for name, expected in pins['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not (relative.suffix == '.py' or relative.name in {'LICENSE', 'LICENSE.txt', 'requirements.txt'}):
            raise ValueError('Only pinned source code, licenses and requirements allowed')
        path = destination / relative
        if path.is_file():
            data = path.read_bytes()
        else:
            url = 'https://raw.githubusercontent.com/tikutikutiku/kaggle-hubmap/' + pins['commit'] + '/' + name
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            downloaded += 1
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Pinned software source mismatch: ' + name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return dict(verified_files=len(pins['files']), downloaded_files=downloaded, source_commit=pins['commit'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch(Path(__file__).resolve().parents[1], args.destination)))
