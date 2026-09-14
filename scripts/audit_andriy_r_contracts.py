#!/usr/bin/env python3
"""Run the synthetic R API audit and bind its evidence to source/runtime hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rscript',default=shutil.which('Rscript'))
    parser.add_argument('--library',required=True,type=Path)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if not args.rscript:
        parser.error('Rscript must be installed or supplied explicitly')
    root=Path(__file__).resolve().parents[1]
    script=root/'scripts/audit_andriy_r_contracts.R'
    env=dict(os.environ,R_LIBS_USER=str(args.library),OMP_NUM_THREADS='1')
    with TemporaryDirectory(prefix='sciona-r-audit-') as temporary:
        output=Path(temporary)/'audit.json'
        subprocess.run([args.rscript,'--vanilla',str(script),str(output)],env=env,check=True,timeout=120)
        report=json.loads(output.read_text())
    report['validation_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),script]}
    report['source_revision']='00f937cc7710977dc812d9fc675864e2b8288658'
    report['source_sha256']={name:hashlib.sha256((args.reference_dir/name).read_bytes()).hexdigest() for name in ['Andriy_mod_glmnet_5_3.R','Andriy_mod_svm_5_7.R','Andriy_mod_xgb_7_5.R']}
    report['compiled_library_sha256']={str(p.relative_to(args.library)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.library.glob('*/libs/*.so')}
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
