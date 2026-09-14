"""Record the installed CPU Amex dependency closure and retain notices."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
ROOT=Path(__file__).resolve().parents[1]
ROOT_PACKAGES=('numpy','scipy','scikit-learn','torch','lightgbm','pandas')


def dependency_closure():
    pending=list(ROOT_PACKAGES);seen={}
    while pending:
        name=canonicalize_name(pending.pop())
        if name in seen:continue
        distribution=metadata.distribution(name);seen[name]=distribution.version
        for raw in distribution.requires or []:
            r=Requirement(raw)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            if not r.specifier.contains(metadata.version(r.name),prereleases=True):raise ValueError('Incompatible runtime dependency: '+r.name)
            pending.append(r.name)
    return dict(sorted(seen.items()))


def main():
    import numpy,scipy,sklearn,torch,lightgbm,pandas
    versions=dependency_closure()
    for name,module in [('numpy',numpy),('scipy',scipy),('scikit-learn',sklearn),('torch',torch),('lightgbm',lightgbm),('pandas',pandas)]:
        if Version(str(module.__version__))!=Version(versions[name]):raise ValueError('Imported version differs')
        expected=Path(metadata.distribution(name).locate_file(module.__name__)).resolve()
        if not Path(module.__file__).resolve().is_relative_to(expected):raise ValueError('Imported location differs')
    notices={};origins={};packages_without_notice=[]
    for name in versions:
        dist=metadata.distribution(name);count=0
        for file in sorted(dist.files or [],key=str):
            base=file.name.lower()
            if not any(word in base for word in ('license','copying','notice','copyright')) or file.suffix in ('.py','.pyc','.class','.h'):continue
            path=Path(dist.locate_file(file))
            if not path.is_file():continue
            data=path.read_bytes();digest=hashlib.sha256(data).hexdigest();target='Amex-notice-'+digest[:16]+'.txt'
            (ROOT/'docs/licenses'/target).write_bytes(data)
            notices[target]=digest;origins[name+'/'+str(file)]=target;count+=1
        if not count:packages_without_notice.append(name)
    if packages_without_notice:raise ValueError('Missing distributed notices: '+','.join(packages_without_notice))
    requirements=ROOT/'requirements/amex-execution.txt'
    requirements.write_text(''.join(name+'=='+version+'\n' for name,version in versions.items()))
    report=dict(status='passed',approved=False,catalog_mutations=0,dependency_versions=versions,
        declared_dependency_constraints_satisfied=True,actual_root_imports_verified=True,
        python_version=platform.python_version(),platform=platform.system(),machine=platform.machine(),execution_device='cpu',
        requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
        license_sha256=notices,license_origins=origins,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Provisioned CPU environment and declared Python dependency closure; no clean-install or cross-platform qualification.',
                'Shared graph runner dependencies are exercised by graph validation, not exhaustively audited here.',
                'Native dynamic-loading and operating-system behavior are qualified only by actual synthetic execution.'])
    (ROOT/'docs/reviews/competition_amex_environment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',dependencies=len(versions),distinct_notices=len(notices),notice_origins=len(origins))))


if __name__=='__main__':main()
