"""Tests for execution receipt generation, signing, and verification."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ageom.receipt import (
    ExecutionReceipt,
    SignedReceipt,
    canonicalize_receipt,
    generate_receipt,
    load_signed_receipt,
    save_signed_receipt,
    sign_receipt,
    verify_receipt,
)

HAS_SSH_KEYGEN = shutil.which("ssh-keygen") is not None


def _make_test_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create minimal test files for receipt generation."""
    cdg = tmp_path / "cdg.json"
    cdg.write_text('{"nodes": [], "edges": []}')
    split = tmp_path / "split.json"
    split.write_text('{"train": [], "test": []}')
    output = tmp_path / "output.json"
    output.write_text('{"predictions": [1, 2, 3]}')
    return cdg, split, output


class TestExecutionReceipt:
    def test_round_trip(self):
        r = ExecutionReceipt(
            bounty_id="b1",
            cdg_hash="abc",
            split_hash="def",
            output_hash="ghi",
            metric_name="loss",
            metric_value=0.42,
            timestamp="2025-01-01T00:00:00Z",
            ageom_version="0.1.0",
        )
        data = r.model_dump()
        restored = ExecutionReceipt.model_validate(data)
        assert restored == r

    def test_receipt_includes_bounty_id(self):
        r = ExecutionReceipt(
            bounty_id="bounty-123",
            cdg_hash="a",
            split_hash="b",
            output_hash="c",
            metric_name="loss",
            metric_value=0.5,
            timestamp="t",
            ageom_version="0.0.0",
        )
        assert r.bounty_id == "bounty-123"


class TestCanonicalizeReceipt:
    def test_deterministic(self):
        r = ExecutionReceipt(
            bounty_id="b1",
            cdg_hash="abc",
            split_hash="def",
            output_hash="ghi",
            metric_name="loss",
            metric_value=0.42,
            timestamp="2025-01-01T00:00:00Z",
            ageom_version="0.1.0",
        )
        a = canonicalize_receipt(r)
        b = canonicalize_receipt(r)
        assert a == b

    def test_sorted_keys(self):
        r = ExecutionReceipt(
            bounty_id="b1",
            cdg_hash="abc",
            split_hash="def",
            output_hash="ghi",
            metric_name="loss",
            metric_value=0.42,
            timestamp="t",
            ageom_version="0.0.0",
        )
        canonical = canonicalize_receipt(r).decode()
        # "ageom_version" should come before "bounty_id"
        assert canonical.index("ageom_version") < canonical.index("bounty_id")


class TestGenerateReceipt:
    def test_hashes_files(self, tmp_path: Path):
        cdg, split, output = _make_test_files(tmp_path)
        r = generate_receipt(
            bounty_id="b1",
            cdg_path=cdg,
            split_path=split,
            output_path=output,
            metric_name="loss",
            metric_value=0.5,
        )
        assert len(r.cdg_hash) == 64
        assert len(r.split_hash) == 64
        assert len(r.output_hash) == 64
        assert r.bounty_id == "b1"


class TestSaveLoadRoundTrip:
    def test_round_trip(self, tmp_path: Path):
        receipt = ExecutionReceipt(
            bounty_id="b1",
            cdg_hash="abc",
            split_hash="def",
            output_hash="ghi",
            metric_name="loss",
            metric_value=0.42,
            timestamp="t",
            ageom_version="0.0.0",
        )
        signed = SignedReceipt(receipt=receipt, signature="SIG_BLOCK")
        path = tmp_path / "receipt.json"
        save_signed_receipt(signed, path)

        loaded = load_signed_receipt(path)
        assert loaded.receipt == receipt
        assert loaded.signature == "SIG_BLOCK"


@pytest.mark.skipif(not HAS_SSH_KEYGEN, reason="ssh-keygen not available")
class TestSignAndVerify:
    def _generate_keypair(self, tmp_path: Path) -> tuple[Path, Path]:
        """Generate an ephemeral ed25519 keypair."""
        key_path = tmp_path / "test_key"
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-q"],
            check=True,
        )
        return key_path, key_path.with_suffix(".pub")

    def _create_allowed_signers(
        self, tmp_path: Path, pub_key_path: Path
    ) -> Path:
        """Create an allowed_signers file for the test key."""
        pub_key = pub_key_path.read_text().strip()
        signers = tmp_path / "allowed_signers"
        signers.write_text(f"ageom-receipt {pub_key}\n")
        return signers

    def test_sign_and_verify(self, tmp_path: Path):
        key_path, pub_path = self._generate_keypair(tmp_path)
        signers = self._create_allowed_signers(tmp_path, pub_path)

        cdg, split, output = _make_test_files(tmp_path)
        receipt = generate_receipt(
            bounty_id="b1",
            cdg_path=cdg,
            split_path=split,
            output_path=output,
            metric_name="loss",
            metric_value=0.5,
        )
        signed = sign_receipt(receipt, key_path)
        assert signed.signature  # non-empty
        assert verify_receipt(signed, signers)

    def test_verify_wrong_key_fails(self, tmp_path: Path):
        (tmp_path / "k1").mkdir()
        (tmp_path / "k2").mkdir()
        key1, pub1 = self._generate_keypair(tmp_path / "k1")
        key2, pub2 = self._generate_keypair(tmp_path / "k2")
        # Allowed signers only has key2
        signers = self._create_allowed_signers(tmp_path, pub2)

        cdg, split, output = _make_test_files(tmp_path)
        receipt = generate_receipt(
            bounty_id="b1",
            cdg_path=cdg,
            split_path=split,
            output_path=output,
            metric_name="loss",
            metric_value=0.5,
        )
        signed = sign_receipt(receipt, key1)  # signed with key1
        assert not verify_receipt(signed, signers)  # signers expects key2
