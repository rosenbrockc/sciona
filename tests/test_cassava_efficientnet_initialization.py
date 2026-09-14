import pytest

from sciona.cassava_efficientnet_initialization import initialize_population


def test_upstream_adapts_exactly_once(monkeypatch):
    import sciona.cassava_efficientnet_upstream as upstream
    import sciona.cassava_training_images as images
    calls = []
    result = object()
    def build(**kwargs):
        calls.append(kwargs)
        return result
    monkeypatch.setattr(upstream, 'build_adapted_model', build)
    monkeypatch.setattr(images, 'adapt_training_normalization', lambda *a, **k: pytest.fail('Double adaptation'))
    actual = initialize_population(weights='local', expected_sha256='0' * 64,
        weight_format='upstream', jpegs=[b'synthetic'], batch_size=1)
    assert actual is result
    assert calls == [dict(weights='local', expected_sha256='0' * 64, training_jpegs=[b'synthetic'], batch_size=1)]


def test_keras_preserves_verified_initialization_and_population(monkeypatch):
    import sciona.cassava_efficientnet_model as models
    import sciona.cassava_training_images as images
    result, calls = object(), []
    def build(**kwargs):
        calls.append(('build', kwargs))
        return result
    def adapt(model, jpegs, **kwargs):
        assert model is result
        calls.append(('adapt', jpegs, kwargs))
    monkeypatch.setattr(models, 'build_model', build)
    monkeypatch.setattr(images, 'adapt_training_normalization', adapt)
    assert initialize_population(weights=None, expected_sha256=None, weight_format='keras',
        jpegs=[b'synthetic'], batch_size=2) is result
    assert calls == [('build', dict(weights=None, expected_sha256=None)),
                     ('adapt', [b'synthetic'], dict(batch_size=2))]


def test_unknown_format_fails_closed():
    with pytest.raises(ValueError, match='weight format'):
        initialize_population(weights=None, expected_sha256=None, weight_format='automatic', jpegs=[], batch_size=1)
