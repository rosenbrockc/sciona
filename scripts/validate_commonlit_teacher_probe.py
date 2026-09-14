"""Review teacher heads and singleton evaluation with synthetic tensors only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


class SyntheticHiddenStates(torch.nn.Module):
    def forward(self,input_ids,attention_mask):
        hidden=input_ids.float()[:,:,None].expand(-1,-1,768)/10
        return SimpleNamespace(hidden_states=(hidden,))


class SyntheticRegressor(torch.nn.Module):
    def forward(self,**kwargs):
        return (torch.ones(kwargs['input_ids'].shape[0],1),)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=(args.source/'teacher.code.json').read_bytes()
    manifest=json.loads((args.source/'teacher_review.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==manifest['code_sha256']
    cells={c['cell_index']:c['source'] for c in json.loads(raw)}
    ns={'torch':torch,'nn':torch.nn,'ROBERTA_PATH':'synthetic-model',
        'AutoConfig':SimpleNamespace(from_pretrained=lambda *_:{}),
        'AutoModel':SimpleNamespace(from_pretrained=lambda *a,**kw:SyntheticHiddenStates())}
    definition=[n for n in ast.parse(cells[9]).body if isinstance(n,ast.ClassDef) and n.name=='LitModel']
    assert len(definition)==1
    exec(compile(ast.Module(body=definition,type_ignores=[]),'<source-attention-head>','exec'),ns)
    torch.manual_seed(1729)
    model=ns['LitModel']()
    ids=torch.tensor([[1,2,3,4]])
    mask=torch.tensor([[1,1,0,0]])
    hidden=model.roberta(ids,mask).hidden_states[-1]
    weights=model.attention(hidden)
    assert torch.all(weights[:,2:]>0)
    torch.testing.assert_close(weights.sum(dim=1),torch.ones(1,1))
    actual=model(ids,mask)
    expected=model.regressor((weights*hidden).sum(dim=1))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    source=cells[16].replace('.cuda()',".to('cpu')")
    definition=[n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Evaluator']
    assert len(definition)==1
    exec(compile(ast.Module(body=definition,type_ignores=[]),'<source-evaluator-cpu>','exec'),ns)
    evaluator=ns['Evaluator'](SyntheticRegressor())
    def batch(count):return {k:torch.ones(count,4,dtype=torch.long) for k in ('input_ids','attention_mask','token_type_ids')}
    assert evaluator.evaluate([batch(2)],None)==[1.,1.]
    try:evaluator.evaluate([batch(1)],None)
    except TypeError as error:assert 'iterable' in str(error)
    else:raise AssertionError('Singleton source evaluator failure not reproduced')
    args.output.write_text(json.dumps({'teacher_version':3,'teacher_code_sha256':manifest['code_sha256'],
        'checks':{'attention_head_matches_weighted_hidden_states':True,'padding_has_positive_pooling_weight':True,
                  'finite_head_gradients':True,'two_row_evaluation_passes':True,'singleton_evaluation_typeerror_reproduced':True},
        'scope':'Original teacher attention head with synthetic hidden states, and evaluator with CPU transfer adaptation and synthetic regressor. No full RoBERTa/teacher training or publication claim.',
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('Teacher head checked; unmasked pooling and singleton evaluator failure reproduced')


if __name__=='__main__':main()
