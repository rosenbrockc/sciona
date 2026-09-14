"""Compare exact source and upstream model tensors without publishing weights."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def tensors(path):
    values = {}
    def collect(name, value):
        if isinstance(value, h5py.Dataset):
            parts = name.split('/')
            assert len(parts) == 3
            # Keras auto-suffixed the inner scope; outer layer and variable
            # names are stable in both artifacts. Reject any ambiguous mapping.
            key = (parts[0], parts[-1])
            assert key not in values
            values[key] = value[()]
    with h5py.File(path) as model:
        model.visititems(collect)
    return values


def main():
    source_root = Path('/private/tmp/sciona_cassava_efficientnet_winner_reference')
    upstream_root = Path('/private/tmp/sciona_cassava_efficientnet_upstream_reference')
    source = json.loads((source_root / 'download_manifest.json').read_text())
    upstream = json.loads((upstream_root / 'manifest.json').read_text())
    for root, expected in [(source_root, source['model_sha256']), (upstream_root, upstream['sha256'])]:
        assert hashlib.sha256((root / 'model.h5').read_bytes()).hexdigest() == expected
    with (upstream_root / 'model.h5').open('rb') as stream:
        assert hashlib.file_digest(stream, 'md5').hexdigest() == '47c10902a4949eec589ab92fe1c35ed8'
    original, candidate = tensors(source_root / 'model.h5'), tensors(upstream_root / 'model.h5')
    assert len(candidate) == 608 and len(original) == 611
    assert set(candidate) <= set(original)
    assert set(original) - set(candidate) == {
        ('normalization_1', 'count:0'), ('normalization_1', 'mean:0'), ('normalization_1', 'variance:0')}
    for key, value in candidate.items():
        assert value.dtype == original[key].dtype and value.shape == original[key].shape
        np.testing.assert_array_equal(value, original[key])
    paths = ['scripts/validate_cassava_efficientnet_weight_origin.py',
        'docs/reviews/competition_cassava_efficientnet_pretrained_validation.json']
    report = {'approved': False, 'passed': True, 'model_software_only': True,
        'source_model_sha256': source['model_sha256'], 'upstream_artifact': upstream,
        'tensor_mapping': 'Outer layer name plus terminal variable name; ignore only auto-suffixed inner Keras scope, rejecting collisions.',
        'upstream_tensors': 608, 'source_tensors': 611, 'identical_parameter_tensors': 608,
        'source_only_tensors': 'Three normalization adaptation state tensors',
        'scope': 'Exact numerical backbone equivalence to public upstream release. Does not establish weight-specific redistribution terms. Reconstructing normalization and verifying post-adaptation lifecycle remain separate gates.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('docs/reviews/competition_cassava_efficientnet_weight_origin_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS all 608 upstream parameter tensors exactly match source; three additional normalization tensors', flush=True)


if __name__ == '__main__':
    main()
