"""Reproduce edge cases in the linked notebook; no reuse license assumed."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='dcf21b634705e83e3fbfcde448146619688449c53ad5a3e4c8bd4bdce9a34bd9'
    ns=dict(np=np,pd=pd,torch=torch,PERIOD=1,SR=10,TTA=1,device='cpu',INV_BIRD_CODE={0:'synthetic'})
    functions=[]
    for text in json.loads(raw):
        try:tree=ast.parse(text)
        except SyntaxError:continue
        functions.extend(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prediction_for_clip')
    assert len(functions)==1
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<reviewed-inference-function>','exec'),ns)
    class Model:
        def __init__(self,single):self.single=single;self.calls=0
        def eval(self):return self
        def __call__(self,inputs):
            self.calls+=1
            frames=torch.zeros((1,1,4,1))
            if self.single:frames[0,0,1,0]=.8
            return dict(framewise_output=frames,clipwise_output=torch.ones((1,1,1)))
    frame=pd.DataFrame(dict(site=['synthetic'],audio_id=['synthetic']))
    try:
        ns['prediction_for_clip'](frame,np.zeros(9),Model(True),.3,.3)
    except ValueError as error:
        assert 'zero-size array' in str(error)
    else:raise AssertionError('Singleton event did not reproduce empty slice failure')
    model=Model(False)
    result=ns['prediction_for_clip'](frame,np.zeros(10),model,.3,.3)
    assert model.calls==2 and result.empty and len(result.columns)==0
    report=dict(singleton_event_empty_slice_reproduced=True,exact_multiple_adds_zero_chunk=True,
        empty_detection_result_has_no_columns=True,notebook_code_sha256=hashlib.sha256(raw).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Synthetic predictions with shortened chunk constants reproduce original event/chunk behavior. No full inference or notebook reuse-license claim.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
