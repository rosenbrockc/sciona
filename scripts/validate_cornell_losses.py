"""Independent scalar and gradient checks for both Cornell source losses."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.source/'manifest.json').read_text())
    pins={p['software_path']:p['sha256'] for p in manifest['pins']}
    ns=dict(torch=torch,nn=nn,EPSILON_FP16=1e-5)
    for name in ('sed_scaled_pos_neg_focal_loss','sed_scaled_pos_neg_focal_loss_augd'):
        path='src/loss/'+name+'.py';raw=(args.source/path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==pins[path]
        classes=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef)]
        exec(compile(ast.Module(body=classes,type_ignores=[]),path,'exec'),ns)
    target=torch.tensor([[[1.,.6,0.,.2]],[[0.,1.,.3,0.]]],dtype=torch.float64)
    secondary=torch.tensor([[[0.,.6,0.,0.]],[[0.,0.,.3,0.]]],dtype=torch.float64)
    probabilities=torch.tensor([[[.7,.4,.2,.9]],[[.1,.8,.5,.3]]],dtype=torch.float64)
    draws=torch.tensor([[.1,.5,.8,.9],[.7,.2,.1,.4]],dtype=torch.float64)
    reports=[]
    for augmented in (False,True):
        cls=ns['SedScaledPosNegFocalLossAugd' if augmented else 'SedScaledPosNegFocalLoss']
        for factor in (0.,.4,1.):
            for gamma in (0.,2.):
                prediction=probabilities.clone().requires_grad_()
                criterion=cls(gamma=gamma,alpha_0=.7,alpha_1=1.3,secondary_factor=factor)
                with patch.object(torch,'rand',return_value=draws):
                    actual,_=criterion(prediction,dict(all_labels=target,secondary_labels=secondary))
                gradient=torch.autograd.grad(actual,prediction)[0]
                ref_prediction=probabilities.clone().requires_grad_();terms=[]
                for i,(p,y,s) in enumerate(zip(ref_prediction.flatten(),target.flatten(),secondary.flatten())):
                    bce=-(y*torch.log(p)+(1-y)*torch.log(1-p))
                    weight=(1.3 if y>0 else .7)
                    weight*=factor if s>0 else 1.
                    if augmented and y-s>0 and draws.flatten()[i]<=.3:weight=0.
                    terms.append(weight*(1-torch.exp(-bce))**gamma*bce)
                expected=torch.stack(terms).mean()
                ref_gradient=torch.autograd.grad(expected,ref_prediction)[0]
                torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12,msg=f"augmented={augmented}, factor={factor}, gamma={gamma}: {actual.item()} vs {expected.item()}")
                torch.testing.assert_close(gradient,ref_gradient,rtol=1e-12,atol=1e-12)
                if factor==0:assert (gradient[secondary>0]==0).all()
                reports.append(dict(augmented=augmented,secondary_factor=factor,gamma=gamma,scalar_and_gradient_match=True))
    augmented=ns['SedScaledPosNegFocalLossAugd'](secondary_factor=1)
    augmented.eval()
    with patch.object(torch,'rand',return_value=torch.zeros_like(draws)):
        dropped,_=augmented(probabilities,dict(all_labels=target,secondary_labels=secondary))
    with patch.object(torch,'rand',return_value=torch.ones_like(draws)):
        retained,_=augmented(probabilities,dict(all_labels=target,secondary_labels=secondary))
    assert dropped<retained
    report=dict(source_commit=manifest['commit'],cases=reports,
        both_losses_secondary_factor_respected=True,
        augmented_loss_secondary_factor_respected=True,
        augmented_loss_primary_dropout_active_in_eval=True,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Original source semantics, synthetic soft labels and deterministic random masks; no correction or full training claim.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(reports),scalar_and_gradient_checks='passed',eval_dropout_reproduced=True)))


if __name__=='__main__':main()
