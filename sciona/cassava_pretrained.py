"""Explicit, locally verified timm reconstruction backbones.

The caller must obtain expected hashes from reviewed artifact evidence. These
tags identify historical-reference candidates, not proven winner checkpoints.
No network downloads or mutable default pretrained configurations are used.
"""
import hashlib
from pathlib import Path
import re


REFERENCE_TAGS = {
    'resnext': 'resnext50_32x4d.ra_in1k',
    'vit': 'vit_base_patch16_384.orig_in21k_ft_in1k',
}


def load_reference_backbone(family, weights_path, *, expected_sha256):
    """Load the verified 1000-class backbone before replacing its task head."""
    if not isinstance(family, str) or family not in REFERENCE_TAGS:
        raise ValueError('An explicitly supported reference family is required')
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('A reviewed lowercase SHA-256 digest is required')
    path = Path(weights_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('A regular local weights file is required')
    # Deserialize the exact bytes verified, avoiding a second mutable file read.
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Reference weights integrity mismatch')
    from safetensors.torch import load
    import timm
    state = load(content)
    model = timm.create_model(REFERENCE_TAGS[family], pretrained=False)
    model.load_state_dict(state, strict=True)
    return model


def build_backbone(family, *, pretrained, weights_path=None, expected_sha256=None):
    """Explicit untrained diagnostic or reviewed local pretrained initialization."""
    if not isinstance(pretrained, bool):
        raise ValueError('Explicit pretrained boolean required')
    if pretrained:
        if weights_path is None or expected_sha256 is None:
            raise ValueError('Pretrained initialization requires local weights and reviewed SHA-256')
        return load_reference_backbone(family, weights_path, expected_sha256=expected_sha256)
    if weights_path is not None or expected_sha256 is not None:
        raise ValueError('Untrained diagnostics cannot silently ignore pretrained artifacts')
    if not isinstance(family, str) or family not in REFERENCE_TAGS:
        raise ValueError('An explicitly supported reference family is required')
    import timm
    return timm.create_model(REFERENCE_TAGS[family], pretrained=False)
