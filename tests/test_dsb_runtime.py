import numpy as np
import pytest
import torch

from sciona.dsb_checkpoint import checkpoint_bytes
from sciona.dsb_network import DetectorNet, CaseNet
from sciona.dsb_pipeline import infer_preprocessed
from sciona.dsb_runtime import restore_training_runtime, restore_inference_runtime
from sciona.dsb_training import run_epoch_lifecycle


def test_warm_restart_shares_models_and_recreates_both_optimizers():
    torch.set_num_threads(1);torch.manual_seed(11)
    model=CaseNet(2)
    runtime=restore_training_runtime(checkpoint_bytes(model,120),topk=2)
    assert runtime['start_epoch']==121
    assert runtime['classifier'].NoduleNet is runtime['detector']
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,runtime['classifier'].state_dict()[key],rtol=0,atol=0)
    for role in ['detector','classifier']:
        optimizer=runtime[role+'_optimizer']
        assert not optimizer.state
        assert {id(p) for p in optimizer.param_groups[0]['params']}=={id(p) for p in runtime[role].parameters()}
        assert optimizer.param_groups[0]['momentum']==.9
        assert optimizer.param_groups[0]['weight_decay']==1e-4


def test_explicit_checkpoint_pair_reproduces_connected_inference():
    torch.set_num_threads(1);torch.manual_seed(12)
    detector,classifier=DetectorNet().eval(),CaseNet(2).eval()
    with torch.no_grad():
        detector.output[-1].weight.zero_();detector.output[-1].bias.zero_()
        detector.output[-1].bias[::5]=2.
    runtime=restore_inference_runtime(checkpoint_bytes(detector,30),
                                      checkpoint_bytes(classifier,121),topk=2)
    assert runtime['detector'] is not runtime['classifier'].NoduleNet
    assert not runtime['detector'].training and not runtime['classifier'].training
    volume=np.full((1,16,16,16),100,dtype=np.uint8)
    options=dict(tile_side=16,margin=0,topk=2,crop_size=16)
    expected=infer_preprocessed(volume,detector,classifier,**options)
    actual=infer_preprocessed(volume,**runtime,**options)
    assert len(actual['proposals'])>=2
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])


def test_wrong_checkpoint_role_is_rejected():
    torch.set_num_threads(1)
    blob=checkpoint_bytes(CaseNet(2),1)
    with pytest.raises(ValueError,match='state keys'):
        restore_inference_runtime(blob,blob,topk=2)


def test_restored_training_epoch_checkpoint_flows_into_inference():
    torch.set_num_threads(1);torch.manual_seed(13)
    initial=CaseNet(2)
    runtime=restore_training_runtime(checkpoint_bytes(initial,159),topk=2)
    images=torch.randn(1,2,1,16,16,16);coords=torch.randn(1,2,3,4,4,4)
    batch=(images,coords,torch.ones(1,2),torch.ones(1,1))
    result=run_epoch_lifecycle(**runtime,training_factory=lambda task,profile:[batch],
        validation_factory=lambda task:[batch],epoch=160,classifier_variant=4,
        freeze_batchnorm=True)
    assert not torch.equal(initial.fc2.weight,runtime['classifier'].fc2.weight)
    standalone=DetectorNet().eval()
    with torch.no_grad():
        standalone.output[-1].weight.zero_();standalone.output[-1].bias.zero_()
        standalone.output[-1].bias[::5]=2.
    inference=restore_inference_runtime(checkpoint_bytes(standalone,140),
                                        result['checkpoint'],topk=2)
    volume=np.full((1,16,16,16),100,dtype=np.uint8)
    options=dict(tile_side=16,margin=0,topk=2,crop_size=16)
    expected=infer_preprocessed(volume,standalone,runtime['classifier'],**options)
    actual=infer_preprocessed(volume,**inference,**options)
    for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
