"""Explicit uninitialized backbone/head construction for APTOS diagnostics.

The source names four architecture families and GeM pooling. A single linear
regression output is an explicit reconstruction choice consistent with the
reported scalar regression loss; exact winning head details are unproven.
This module never downloads or claims to load pretrained parameters.
"""
from torch import nn
from sciona.aptos_pooling import GeM

FAMILIES = {
    'inception_resnet_v2': ('inceptionresnetv2', 'avgpool_1a', 1536, 512),
    'inception_v4': ('inceptionv4', 'avg_pool', 1536, 512),
    'seresnext50': ('se_resnext50_32x4d', 'avg_pool', 2048, 512),
    'seresnext101': ('se_resnext101_32x4d', 'avg_pool', 2048, 384),
}


def build_uninitialized(family):
    """Replace the actual global pool and 1000-way head; preserve backbone ops.

    Returns the native model, producing an N-by-1 unbounded regression tensor.
    Every parameter, including GeM's exponent, remains trainable. These are
    diagnostic random parameters, not a qualified pretrained initialization.
    """
    if family not in FAMILIES:
        raise ValueError('A specified APTOS architecture family is required')
    import pretrainedmodels
    factory, pool_name, features, _ = FAMILIES[family]
    model = getattr(pretrainedmodels, factory)(num_classes=1000, pretrained=None)
    if (not hasattr(model, pool_name) or not isinstance(model.last_linear, nn.Linear)
            or model.last_linear.in_features != features or model.last_linear.out_features != 1000):
        raise ValueError('Legacy backbone pooling or head contract differs')
    setattr(model, pool_name, GeM())
    model.last_linear = nn.Linear(features, 1)
    return model
