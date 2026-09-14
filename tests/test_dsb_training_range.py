import io
import pytest
import torch

from sciona.dsb_checkpoint import checkpoint_bytes
from sciona.dsb_network import CaseNet
from sciona.dsb_runtime import restore_training_runtime
from sciona.dsb_training import run_epoch_lifecycle
from sciona.dsb_training_range import run_training_range


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal_tree(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b


@pytest.mark.parametrize('start,end',[(49,51),(159,161)])
def test_range_matches_persistent_manual_epoch_loop(start,end):
    torch.set_num_threads(1);torch.manual_seed(41)
    payload=checkpoint_bytes(CaseNet(2),start-1)
    actual=restore_training_runtime(payload,topk=2)
    expected=restore_training_runtime(payload,topk=2)
    images=torch.randn(2,1,16,16,16);coords=torch.randn(2,3,4,4,4)
    labels=torch.zeros(2,4,4,4,3,5);labels[...,0]=-1;labels[:,1,1,1,0,0]=1
    det=(images,labels,coords)
    case=(images[None],coords[None],torch.ones(1,2),torch.ones(1,1))
    training=lambda task,profile:[det if task=='detector' else case]
    validation=lambda task:[det if task=='detector' else case]
    torch.manual_seed(53)
    reports=[]
    for epoch in range(start,end+1):
        result=run_epoch_lifecycle(**expected,training_factory=training,validation_factory=validation,
            epoch=epoch,classifier_variant=4,freeze_batchnorm=True)
        reports.append(dict(epoch=epoch,training=result['training'],validation=result['validation']))
    expected_rng=torch.get_rng_state()
    torch.manual_seed(53);calls=[]
    result=run_training_range(actual,training,validation,end_epoch=end,classifier_variant=4,
        freeze_batchnorm=True,prepare_epoch=calls.append)
    assert calls==list(range(start,end+1))
    assert result['epochs']==reports
    assert sum(p['learning_rate']==0 for r in reports for p in r['training'])==1
    for role in ['classifier','detector']:
        equal_tree(actual[role].state_dict(),expected[role].state_dict())
        equal_tree(actual[role+'_optimizer'].state_dict(),expected[role+'_optimizer'].state_dict())
    torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0,atol=0)
    checkpoint=torch.load(io.BytesIO(result['checkpoint']),weights_only=True)
    assert checkpoint['epoch']==end
    equal_tree(checkpoint['state_dict'],expected['classifier'].state_dict())


def test_invalid_range_rejected_before_training_or_callback():
    torch.set_num_threads(1)
    runtime=restore_training_runtime(checkpoint_bytes(CaseNet(2),120),topk=2)
    calls=[]
    with pytest.raises(ValueError):
        run_training_range(runtime,None,None,end_epoch=181,classifier_variant=4,prepare_epoch=calls.append)
    assert calls==[]
    assert not runtime['classifier_optimizer'].state and not runtime['detector_optimizer'].state
