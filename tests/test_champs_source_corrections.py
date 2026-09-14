import hashlib

import pytest

from sciona.champs_source_corrections import correct_triplet_layout


def test_packer_changes_only_second_bond_destination():
    source = b"x_triplet[i,:p,5] = torch.tensor(b_idx1.values)\nx_triplet[i,:p,5] = torch.tensor(b_idx2.values)"
    result = correct_triplet_layout(source, hashlib.sha256(source).hexdigest(), kind="packer")
    assert result.splitlines() == ["x_triplet[i,:p,5] = torch.tensor(b_idx1.values)",
                                   "x_triplet[i,:p,6] = torch.tensor(b_idx2.values)"]


@pytest.mark.parametrize("mode", ["hash", "missing", "duplicate", "unknown"])
def test_reject_unreviewed_source(mode):
    source = b"x_triplet[i,:p,5] = torch.tensor(b_idx2.values)"
    if mode == "missing":
        source = b"unrelated source"
    if mode == "duplicate":
        source += b"\n" + source
    digest = "0"*64 if mode == "hash" else hashlib.sha256(source).hexdigest()
    with pytest.raises(ValueError):
        correct_triplet_layout(source, digest, kind="unknown" if mode == "unknown" else "packer")
