"""Record installed Otto dependencies and retain their distributed notices.

This is inventory evidence, not clean-install or publication qualification.
Runtime locations are command-line arguments and are not stored in the report.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import zipfile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
ROOT=Path(__file__).resolve().parents[1]


def sha(data):return hashlib.sha256(data).hexdigest()


def closure():
    pending=['numpy','scipy','scikit-learn','xgboost','h2o','threadpoolctl','six'];versions={}
    while pending:
        name=canonicalize_name(pending.pop())
        if name in versions:continue
        dist=metadata.distribution(name);versions[name]=dist.version
        for raw in dist.requires or []:
            r=Requirement(raw)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            if not r.specifier.contains(metadata.version(r.name),prereleases=True):
                raise ValueError('Incompatible declared dependency: '+r.name)
            pending.append(r.name)
    return dict(sorted(versions.items()))


def main():
    parser=argparse.ArgumentParser()
    for name in ('r-home','r-library','java-home','lasagne','theano','libfm','rsofia'):
        parser.add_argument('--'+name,required=True,type=Path)
    cli=parser.parse_args();versions=closure();licenses={};origins={};dedup={}
    destination=ROOT/'docs/licenses';destination.mkdir(exist_ok=True)
    def retain(origin,data):
        digest=sha(data)
        if digest not in dedup:
            name='Otto-notice-'+digest[:16]+'.txt'
            (destination/name).write_bytes(data);dedup[digest]=name;licenses[name]=digest
        origins[origin]=dedup[digest]
    def notice(name):
        base=Path(name).name.lower()
        return any(t in base for t in ('license','copying','notice','copyright')) and not base.endswith(('.py','.pyc','.class'))
    for name in versions:
        dist=metadata.distribution(name)
        for file in sorted(dist.files or [],key=str):
            p=Path(dist.locate_file(file))
            if p.is_file() and notice(str(file)):
                retain('python/'+name+'/'+str(file),p.read_bytes())
    h2o=metadata.distribution('h2o');jars={}
    for file in h2o.files or []:
        if file.suffix=='.jar':
            path=Path(h2o.locate_file(file));jars[file.name]=sha(path.read_bytes())
            with zipfile.ZipFile(path) as archive:
                for name in sorted(archive.namelist()):
                    if not name.endswith('/') and notice(name):retain('h2o-jar/'+name,archive.read(name))
    for label,root,file in [('Lasagne',cli.lasagne,'LICENSE'),('Theano',cli.theano,'LICENSE.txt'),
                           ('libFM',cli.libfm,'license.txt'),('RSofia',cli.rsofia,'inst/sofia-ml-read-only/sofia-ml/COPYING')]:
        retain(label+'/'+file,(root/file).read_bytes())
    for name in ('GPL-2','GPL-3','LGPL-2','LGPL-2.1','LGPL-3'):
        retain('R/'+name,(cli.r_home/'share/licenses'/name).read_bytes())
    for file in sorted((cli.java_home/'legal').rglob('*')):
        if file.is_file():retain('OpenJDK/'+str(file.relative_to(cli.java_home/'legal')),file.read_bytes())
    commits={};source_hashes={}
    for name in ('lasagne','theano','libfm','rsofia'):
        root=getattr(cli,name)
        commits[name]=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        files=subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0')
        source_hashes[name]={f:sha((root/f).read_bytes()) for f in files if f and (root/f).is_file() and Path(f).suffix in ('.py','.R','.cpp','.h','.c')}
    r_packages={}
    for name in ('randomForest','RSofia','Rcpp'):
        description=(cli.r_library/name/'DESCRIPTION').read_text()
        fields=dict(line.split(': ',1) for line in description.splitlines() if ': ' in line and not line.startswith((' ','\t')))
        r_packages[name]={k:fields[k] for k in ('Version','License')}
        binaries={str(p.relative_to(cli.r_library/name)):sha(p.read_bytes()) for p in (cli.r_library/name/'libs').rglob('*') if p.is_file()}
        r_packages[name]['binary_sha256']=binaries
    requirements=ROOT/'requirements/otto-execution.txt'
    requirements.write_text(''.join(name+'=='+version+'\n' for name,version in versions.items()))
    report=dict(status='inventory_complete',approved=False,catalog_mutations=0,
        python_version=platform.python_version(),platform=platform.system(),machine=platform.machine(),
        dependency_versions=versions,declared_python_constraints_satisfied=True,
        requirements_sha256=sha(requirements.read_bytes()),license_sha256=licenses,license_origins=origins,
        source_commits=commits,source_code_sha256=source_hashes,r_packages=r_packages,h2o_jar_sha256=jars,
        libfm_binary_sha256=sha((cli.libfm/'bin/libFM').read_bytes()),
        java_release=(cli.java_home/'release').read_text(),
        compatibility_patch_sha256={p.name:sha(p.read_bytes()) for p in (ROOT/'docs/reviews').glob('otto_*compatibility.patch')},
        limits=['Inventory only: imported runtime identity and complete native dynamic-library closure still require verification.',
                'Source checkouts are qualified separately from installed Python packages; no clean-install or GPU claim.',
                'Shared graph runner dependencies are outside these algorithm requirements.'])
    (ROOT/'docs/reviews/competition_otto_environment_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],python_packages=len(versions),distinct_notices=len(licenses),notice_origins=len(origins),source_checkouts=len(commits))))


if __name__=='__main__':main()
