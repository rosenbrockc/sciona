import io
import pytest
import torch
from sciona.hubmap_checkpoint import checkpoint_bytes,restore_checkpoint


def test_tensor_only_roundtrip_is_independent_snapshot():
    model=torch.nn.Linear(2,1)
    expected={k:v.clone() for k,v in model.state_dict().items()}
    payload=checkpoint_bytes(model)
    with torch.no_grad():model.weight.add_(1)
    restore_checkpoint(model,payload)
    assert all(torch.equal(v,expected[k]) for k,v in model.state_dict().items())
    restored=torch.load(io.BytesIO(payload),weights_only=True)
    assert set(restored)==set(expected) and all(isinstance(v,torch.Tensor) for v in restored.values())


@pytest.mark.parametrize('bad',['nan','dtype','shape','key'])
def test_invalid_state_does_not_partially_load(bad):
    model=torch.nn.Linear(2,1)
    before={k:v.clone() for k,v in model.state_dict().items()}
    state={k:v.clone() for k,v in before.items()}
    state['weight'].add_(1)
    if bad=='nan':state['bias'].fill_(float('nan'))
    if bad=='dtype':state['bias']=state['bias'].double()
    if bad=='shape':state['bias']=torch.zeros(3)
    if bad=='key':state['extra']=torch.zeros(1)
    stream=io.BytesIO();torch.save(state,stream)
    with pytest.raises(ValueError):restore_checkpoint(model,stream.getvalue())
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
