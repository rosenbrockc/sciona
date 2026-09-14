"""Explicit CHAMPS triplet-layout corrections for the new execution version.

The original source cache stays unchanged. The seven-column layout is two
categorical types, three atom references, and two bond references. These changes
alter source behavior; no original-checkpoint prediction parity is claimed.
"""
import hashlib


def correct_triplet_layout(source: bytes, expected_sha256: str, *, kind: str) -> str:
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("CHAMPS source hash mismatch")
    text = source.decode("utf-8")
    if kind == "packer":
        replacements = {
            "x_triplet[i,:p,5] = torch.tensor(b_idx2.values)":
            "x_triplet[i,:p,6] = torch.tensor(b_idx2.values)",
        }
    elif kind == "graph":
        replacements = {
            "a1,a2,a3 = x_triplet[i,:,1], x_triplet[i,:,2], x_triplet[i,:,3]":
            "a1,a2,a3 = x_triplet[i,:,2], x_triplet[i,:,3], x_triplet[i,:,4]",
            "b1,b2 = x_triplet[i,:,4], x_triplet[i,:,5]":
            "b1,b2 = x_triplet[i,:,5], x_triplet[i,:,6]",
        }
    else:
        raise ValueError("Unknown CHAMPS correction kind")
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError("CHAMPS source no longer matches reviewed correction")
        text = text.replace(old, new)
    return text
