#!/usr/bin/env python3
"""Fetch pinned public algorithm/network source only; never fetch input data."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(pins, destination):
    result = {}
    for variant in ['targeted', 'untargeted']:
        pin = pins[variant]
        if len(pin['commit']) != 40 or any(c not in '0123456789abcdef' for c in pin['commit']):
            raise ValueError('Immutable commit required')
        base = pin['repository'].replace('https://github.com/', 'https://raw.githubusercontent.com/')+'/'+pin['commit']
        pending = [pin['source_file'], 'nets/__init__.py', 'LICENSE']
        files = {}
        while pending:
            filename = pending.pop()
            if filename in files:
                continue
            parts = Path(filename).parts
            if Path(filename).is_absolute() or '..' in parts:
                raise ValueError('Unsafe source path')
            request = urllib.request.Request(base+'/'+filename, headers={'User-Agent': 'sciona-source-review'})
            data = urllib.request.urlopen(request, timeout=30).read()
            sha = hashlib.sha256(data).hexdigest()
            if filename == pin['source_file'] and sha != pin['sha256']:
                raise ValueError('Pinned entrypoint hash differs')
            output = destination/variant/filename
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            files[filename] = sha
            if not filename.endswith('.py'):
                continue
            for node in ast.walk(ast.parse(data)):
                if isinstance(node, ast.ImportFrom) and node.module == 'nets':
                    for alias in node.names:
                        if not alias.name.isidentifier():
                            raise ValueError('Unsupported network import')
                        pending.append('nets/'+alias.name+'.py')
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('nets.'):
                    pending.append(node.module.replace('.', '/')+'.py')
        result[variant] = dict(commit=pin['commit'], source_files=files)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root/'docs/reviews/competition_gradient_source_pins.json').read_text())
    result = fetch(pins, args.destination)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({variant: len(record['source_files']) for variant, record in result.items()}))
