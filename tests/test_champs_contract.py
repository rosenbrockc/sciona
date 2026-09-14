import copy

import pytest

from sciona.champs_contract import prepare
from sciona.champs_ensemble import MODEL_ORDER


def payload():
    molecule={'key':'synthetic','elements':['O','H','H'],'coordinates':[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
              'couplings':[{'key':'synthetic-request','atoms':[1,2],'type':'2JHH','target':.5}]}
    inference=copy.deepcopy(molecule)
    del inference['couplings'][0]['target']
    return {'version':1,'training':[molecule],'inference':[inference],'selection':'full',
            'models':{name:{'epochs':1,'options':{}} for name in MODEL_ORDER}}


def test_private_keys_get_separate_internal_ids_and_inputs_are_copied():
    raw=payload()
    prepared=prepare(raw)
    assert prepared.inference_keys=={0:'synthetic-request'}
    assert prepared.inference_couplings.id.tolist()==[0]
    raw['training'][0]['coordinates'][0][0]=99.
    assert prepared.training_atoms.x.iloc[0]==0.


@pytest.mark.parametrize('bad', ['inference_target','duplicate_pair','nonfinite','missing_model','invalid_option','boolean_index','extra_path'])
def test_invalid_inputs_rejected(bad):
    raw=payload()
    if bad=='inference_target': raw['inference'][0]['couplings'][0]['target']=1.
    elif bad=='duplicate_pair':
        row=copy.deepcopy(raw['training'][0]['couplings'][0]);row['key']='another'
        raw['training'][0]['couplings'].append(row)
    elif bad=='nonfinite': raw['training'][0]['coordinates'][0][0]=float('nan')
    elif bad=='missing_model': del raw['models']['model_H']
    elif bad=='invalid_option': raw['models']['model_H']['options']={'unknown':1}
    elif bad=='boolean_index': raw['training'][0]['couplings'][0]['atoms'][0]=True
    elif bad=='extra_path': raw['source_path']='untrusted'
    with pytest.raises(ValueError): prepare(raw)
