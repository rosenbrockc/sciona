"""Authenticate the exact Homebrew install-name relocation and signed pages."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def commands(blob):
    header = struct.unpack_from('<8I', blob)
    if header[0] != 0xfeedfacf:
        raise ValueError('Expected little-endian64bit Mach-O')
    offset = 32
    result = {}
    for _ in range(header[4]):
        kind, size = struct.unpack_from('<II', blob, offset)
        if size < 8 or offset+size > 32+header[5]:
            raise ValueError('Invalid Mach-O command bounds')
        result.setdefault(kind, []).append((offset,size))
        offset += size
    if offset != 32+header[5]: raise ValueError('Load-command length mismatch')
    return result


def signed_pages(blob, signature):
    offset, size = signature
    magic, length, count = struct.unpack_from('>III',blob,offset)
    if magic != 0xfade0cc0 or length != size:
        raise ValueError('Unexpected signature format')
    directories = []
    for i in range(count):
        kind, relative = struct.unpack_from('>II',blob,offset+12+8*i)
        if kind == 0: directories.append(offset+relative)
    if len(directories) != 1: raise ValueError('Expected one primary code directory')
    start = directories[0]
    fields = struct.unpack_from('>9I',blob,start)
    magic, length, version, flags, hashes, ident, special, slots, limit = fields
    hash_size, hash_type, platform, page_power = blob[start+36:start+40]
    if magic != 0xfade0c02 or hash_size != 32 or hash_type != 2 or limit != offset:
        raise ValueError('Unexpected code hashing scheme')
    page_size = 1 << page_power
    if slots != (limit+page_size-1)//page_size:
        raise ValueError('Code slot count mismatch')
    for slot in range(slots):
        expected = hashlib.sha256(blob[slot*page_size:min((slot+1)*page_size,limit)]).digest()
        stored = blob[start+hashes+slot*32:start+hashes+(slot+1)*32]
        if expected != stored: raise ValueError('Signed page mismatch')
    return start+hashes, slots


def main():
    report_path=Path('docs/reviews/competition_flavours_native_publisher.json')
    report=json.loads(report_path.read_text())
    source=Path('/private/tmp/sciona_flavours_native/bottle-libomp.dylib').read_bytes()
    installed_path=Path('/opt/homebrew/opt/libomp/lib/libomp.dylib')
    installed=installed_path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=report['bottle_library_sha256']:
        raise ValueError('Authenticated bottle payload drift')
    if len(source)!=len(installed) or commands(source)!=commands(installed):
        raise ValueError('Unexpected structural changes')
    table=commands(source)
    (identity, size), = table[0xd]
    name_offset=struct.unpack_from('<I',source,identity+8)[0]
    old=source[identity+name_offset:identity+size].split(b'\0')[0]
    if old!=b'@@HOMEBREW_PREFIX@@/opt/libomp/lib/libomp.dylib':
        raise ValueError('Unexpected original install name')
    new=b'/opt/homebrew/opt/libomp/lib/libomp.dylib\0'
    expected=bytearray(source)
    expected[identity+name_offset:identity+size]=new.ljust(size-name_offset,b'\0')
    (sig_cmd,_),=table[0x1d]
    signature=struct.unpack_from('<II',source,sig_cmd+8)
    first_hash,slots=signed_pages(source,signature)
    installed_hash,installed_slots=signed_pages(installed,signature)
    if (first_hash,slots)!=(installed_hash,installed_slots):
        raise ValueError('Signature structure differs')
    # Only page0 contains the install-name change. All other signature bytes
    # must be untouched, including requirements and signature metadata.
    expected[first_hash:first_hash+32]=installed[first_hash:first_hash+32]
    if bytes(expected)!=installed:
        raise ValueError('Changes exceed install name and its signed page hash')
    subprocess.run(['codesign','--verify','--strict',str(installed_path)],check=True,capture_output=True)
    report.update(relocation_verified=True,signed_pages_verified_per_binary=slots,
                  changes='Exact Homebrew placeholder-to-prefix install name and SHA256signed page0hash only',
                  codesign_strict_verification='passed',
                  relocation_verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS exact relocation and',slots,'signed pages per binary')


if __name__=='__main__': main()
