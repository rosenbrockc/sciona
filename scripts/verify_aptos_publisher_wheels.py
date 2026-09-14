"""Compare installed payloads with SHA-256 authenticated publisher wheels."""
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
import sys


def main():
    audit_path = Path('docs/reviews/competition_aptos_published_wheel_audit.json')
    audit = json.loads(audit_path.read_text())
    assert audit['all_wheel_pins_have_compatible_unyanked_wheels']
    root = Path('/private/tmp/sciona_aptos_publisher_wheels')
    packages = {}
    for name, details in audit['packages'].items():
        artifact = details['selected_wheel']
        wheel = root / artifact['filename']
        assert wheel.stat().st_size == artifact['bytes']
        with wheel.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == artifact['sha256']
        distribution = metadata.distribution(name)
        assert distribution.version == details['version']
        matched, skipped, failures, verified_scripts = 0, [], [], []
        with zipfile.ZipFile(wheel) as archive:
            records = [entry for entry in archive.namelist()
                       if entry.endswith('.dist-info/RECORD') and len(PurePosixPath(entry).parts) == 2]
            assert len(records) == 1
            for entry, encoded_hash, size in csv.reader(io.StringIO(archive.read(records[0]).decode())):
                if not encoded_hash:
                    continue
                parts = PurePosixPath(entry).parts
                assert not PurePosixPath(entry).is_absolute() and '..' not in parts
                if parts[0].endswith('.data'):
                    if len(parts) >= 3 and parts[1] in ('purelib', 'platlib'):
                        relative = '/'.join(parts[2:])
                    elif len(parts) >= 3 and parts[1] == 'data':
                        suffix = '/'.join(parts[2:])
                        destinations = [str(item) for item in distribution.files or []
                                        if str(item) == suffix or str(item).endswith('/' + suffix)]
                        if len(destinations) != 1:
                            skipped.append(entry)
                            continue
                        relative = destinations[0]
                    elif len(parts) == 3 and parts[1] == 'scripts':
                        destinations = [item for item in distribution.files or []
                                        if str(item).endswith('/bin/' + parts[2])]
                        if len(destinations) != 1:
                            skipped.append(entry)
                            continue
                        installed = Path(distribution.locate_file(destinations[0]))
                        source = archive.read(entry)
                        actual = installed.read_bytes()
                        source_header, _, source_body = source.partition(b'\n')
                        actual_header, _, actual_body = actual.partition(b'\n')
                        interpreter = Path(actual_header[2:].decode()) if actual_header.startswith(b'#!') else None
                        if (source_header != b'#!python' or source_body != actual_body
                                or interpreter is None or not interpreter.is_file()
                                or interpreter.parent.resolve() != Path(sys.executable).parent.resolve()
                                or not interpreter.samefile(sys.executable)):
                            failures.append({'wheel_path': entry, 'reason': 'script body or environment shebang mismatch'})
                        else:
                            verified_scripts.append({'wheel_path': entry, 'comparison': 'Exact body; installer shebang resolves to the current virtual-environment interpreter'})
                            matched += 1
                        continue
                    else:
                        skipped.append(entry)
                        continue
                else:
                    relative = entry
                installed = Path(distribution.locate_file(relative))
                if not installed.is_file():
                    failures.append({'wheel_path': entry, 'reason': 'missing installed payload'})
                    continue
                algorithm, expected = encoded_hash.split('=', 1)
                with installed.open('rb') as stream:
                    actual = base64.urlsafe_b64encode(hashlib.file_digest(stream, algorithm).digest()).decode().rstrip('=')
                if actual != expected:
                    failures.append({'wheel_path': entry, 'reason': 'installed payload differs'})
                else:
                    matched += 1
        packages[name] = {'version': distribution.version, 'publisher_wheel_sha256': artifact['sha256'],
            'matched_payload_files': matched, 'verified_script_shebang_rewrites': verified_scripts,
            'unverified_install_scheme_files': skipped, 'failures': failures}
        print(f'{name}: {matched} matched, {len(failures)} differences, {len(skipped)} scheme entries', flush=True)
    report = {'approved': False, 'all_compared_payloads_match': all(not p['failures'] for p in packages.values()),
        'all_install_scheme_files_compared': all(not p['unverified_install_scheme_files'] for p in packages.values()),
        'packages': packages, 'source_only_packages': audit['source_only_packages'],
        'scope': 'Publisher-index SHA-256 authenticated wheels and installed library/data payload comparison. Script bodies match exactly with documented installer shebang rewrites to this virtual environment. Data destinations require unique installed RECORD suffix matches. Other installer-generated files are outside wheel payloads; unresolved schemes are explicit. Fresh installation is not claimed.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
            str(audit_path), 'scripts/verify_aptos_publisher_wheels.py', 'requirements/aptos-reference-macos-arm64.lock']}}
    Path('docs/reviews/competition_aptos_publisher_payload_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Publisher payload comparison complete', flush=True)


if __name__ == '__main__':
    main()
