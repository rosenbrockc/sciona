import pytest

from sciona.cassava_efficientnet_schedule import learning_rate


def test_warmup_peak_and_endpoints():
    rates = [learning_rate(i) for i in range(20)]
    assert rates[0] == pytest.approx(1e-6)
    assert rates[4] == pytest.approx(2e-4)
    assert rates[19] == pytest.approx(1e-6)
    assert rates[:5] == sorted(rates[:5])
    assert rates[4:] == sorted(rates[4:], reverse=True)


def test_refit_keeps_twenty_epoch_clock():
    # A 14-epoch cosine would end at 1e-6; the source retains a higher rate.
    assert learning_rate(13) == pytest.approx(6.975280905969273e-5)


@pytest.mark.parametrize('epoch', [-1, 20, 1.5, True])
def test_invalid_epoch(epoch):
    with pytest.raises(ValueError): learning_rate(epoch)
