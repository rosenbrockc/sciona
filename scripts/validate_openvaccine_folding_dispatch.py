"""Pinned Arnie command construction; subprocess is an explicit test double."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    cache=Path('/private/tmp/sciona_openvaccine_arnie')
    manifest=json.loads((cache/'manifest.json').read_text())
    ns=dict(np=np,DEBUG=False,package_locs={'contrafold_2':'synthetic-engine'},
            filename=lambda:'synthetic-temporary',
            convert_dbn_to_contrafold_input=lambda *args:None,
            os=SimpleNamespace(remove=lambda path:None,path=SimpleNamespace(isdir=lambda path:True)))
    commands=[]
    class Process:
        returncode=0
        def communicate(self):return b'........\n',b''
    def popen(command,**kwargs):commands.append(command);return Process()
    ns['sp']=SimpleNamespace(Popen=popen,PIPE=-1)
    for file,name in [('mfe.py','mfe_contrafold_'),('pfunc.py','pfunc_contrafold_')]:
        raw=(cache/file).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==next(p['sha256'] for p in manifest['files'] if p['software_path'].endswith('/'+file))
        tree=ast.parse(raw)
        definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[definition],type_ignores=[]),'<pinned-arnie-command>','exec'),ns)
    assert ns['mfe_contrafold_']('A'*8,param_file='synthetic-parameters')=='........'
    assert '--viterbi' not in commands[-1]
    assert commands[-1][-2:]==['--params','synthetic-parameters']
    ns['mfe_contrafold_']('A'*8,param_file='synthetic-parameters',viterbi=True)
    assert commands[-1][-1]=='--viterbi'
    ns['pfunc_contrafold_']('A'*8,param_file='synthetic-parameters',bpps=True)
    assert commands[-1][-3:]==['--posteriors','0.0000000001','synthetic-temporary.posteriors']
    report=dict(status='passed',source_commit=manifest['commit'],source_pins=manifest['files'],
                default_viterbi=False,explicit_viterbi_forwarded=True,explicit_parameters_forwarded=True,
                posterior_cutoff=1e-10,command_cases=3,engine_executed=False,
                scope='Pinned Arnie subprocess argument construction only; explicit subprocess test double. No folding engine, parameters or biological data executed.',
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_folding_dispatch.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_pins'}))


if __name__=='__main__':main()
