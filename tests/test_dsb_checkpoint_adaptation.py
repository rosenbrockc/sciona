import io
import torch
from sciona.dsb_checkpoint import checkpoint_bytes
from sciona.dsb_checkpoint_adaptation import adapt_detector_checkpoint
from sciona.dsb_network import DetectorNet,Net,CaseNet


def test_detector_transfer_preserves_all_tensors_and_initialization_rng():
    torch.set_num_threads(1);torch.manual_seed(61)
    detector=DetectorNet()
    payload=checkpoint_bytes(detector,100)
    torch.manual_seed(67)
    # Exact operation order from pinned adapt_ckpt.py, with metadata-free IO.
    feature=Net();feature.load_state_dict(detector.state_dict())
    expected=CaseNet(2,nodulenet=feature)
    expected_rng=torch.get_rng_state()
    torch.manual_seed(67)
    result=torch.load(io.BytesIO(adapt_detector_checkpoint(payload,topk=2)),weights_only=True)
    assert result['epoch']==0
    for key,value in expected.state_dict().items():
        torch.testing.assert_close(value,result['state_dict'][key],rtol=0,atol=0)
    for key,value in detector.state_dict().items():
        torch.testing.assert_close(value,result['state_dict']['NoduleNet.'+key],rtol=0,atol=0)
    torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0,atol=0)
