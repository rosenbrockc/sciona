"""Runtime-only alignment of source and prepared training image views."""
from dataclasses import dataclass, field
import hashlib

from sciona.cassava_fold_contract import _keys, members
from sciona.cassava_resnext_images import decode_rgb


@dataclass(frozen=True, repr=False)
class PopulationViews:
    source_images: tuple = field(repr=False)
    efficientnet_images: tuple = field(repr=False)


def _align(expected, rows, shape):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('Nonempty keyed image rows required')
    if any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in rows):
        raise ValueError('Each image row must contain a runtime identity and encoded bytes')
    keys = _keys([row[0] for row in rows])
    if set(keys) != set(expected):
        raise ValueError('Image identities must exactly match the fold plan')
    ordered, fingerprints = {}, set()
    for key, encoded in rows:
        if not isinstance(encoded, bytes) or not encoded:
            raise ValueError('Nonempty encoded image bytes required')
        pixels = decode_rgb(encoded)
        if pixels.shape != shape:
            raise ValueError('Image view has an unsupported spatial shape')
        fingerprint = hashlib.sha256(pixels.tobytes()).digest()
        if fingerprint in fingerprints:
            raise ValueError('Different runtime identities reference identical decoded images')
        fingerprints.add(fingerprint)
        ordered[key] = encoded
    return tuple(ordered[key] for key in expected)


def align_training_views(plan, *, source_rows, efficientnet_rows):
    """Align both family inputs to one reviewed fold plan.

The caller must establish provenance linking each prepared image to its source
identity. This checks key coverage and exact decoded duplicates, not semantic
equivalence of arbitrary preprocessing or near-duplicate physical images.
No guessed resize conversion replaces the source's supplied prepared records.
Fingerprints are transient and are never returned or logged.
"""
    members(plan, 0)
    return PopulationViews(
        _align(plan.keys, source_rows, (600, 800, 3)),
        _align(plan.keys, efficientnet_rows, (512, 512, 3)),
    )
