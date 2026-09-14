import pytest
from sciona.hubmap_network import UNET_SERESNEXT101


@pytest.mark.parametrize('options', [
    {'load_weights': True}, {'clf_threshold': .5}, {'resolution': (31,32)},
    {'resolution': (32,0)}, {'deepsupervision': 1}, {'clfhead': 'yes'},
])
def test_rejects_unsupported_contract_before_model_allocation(options):
    config=dict(resolution=(32,32),deepsupervision=True,clfhead=True,clf_threshold=None,load_weights=False)
    config.update(options)
    with pytest.raises(ValueError): UNET_SERESNEXT101(**config)
