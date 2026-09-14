"""Explicit pretrained format selection and one population adaptation."""


def initialize_population(*, weights, expected_sha256, weight_format, jpegs, batch_size):
    if weight_format == 'upstream':
        from sciona.cassava_efficientnet_upstream import build_adapted_model
        return build_adapted_model(weights=weights, expected_sha256=expected_sha256,
                                   training_jpegs=jpegs, batch_size=batch_size)
    if weight_format != 'keras':
        raise ValueError('Explicit keras or upstream weight format required')
    from sciona.cassava_efficientnet_model import build_model
    from sciona.cassava_training_images import adapt_training_normalization
    model = build_model(weights=weights, expected_sha256=expected_sha256)
    adapt_training_normalization(model, jpegs, batch_size=batch_size)
    return model
