import pytest

from sciona.grapheme_b7_schedule import warmup_linear_multiplier


def test_missing_warmup_cannot_silently_start_training():
    with pytest.raises(TypeError):
        warmup_linear_multiplier(0, total_steps=400)


@pytest.mark.parametrize('warmup', [None, True, 0, -1, 400, 401, float('nan'), float('inf')])
def test_invalid_warmup_rejected(warmup):
    with pytest.raises(ValueError):
        warmup_linear_multiplier(0, total_steps=400, warmup_steps=warmup)


@pytest.mark.parametrize('step', [-1, 401, .5, True])
def test_outside_training_lifecycle_rejected(step):
    with pytest.raises(ValueError):
        warmup_linear_multiplier(step, total_steps=400, warmup_steps=20)
