"""CLI commands for execution receipt signing and verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_receipt_sign(args: argparse.Namespace) -> None:
    """Sign an execution receipt."""
    from ageom.receipt import generate_receipt, save_signed_receipt, sign_receipt

    cdg_path = Path(args.cdg)
    split_path = Path(args.split)
    output_path = Path(args.output)
    key_path = Path(args.key)

    for label, p in [("CDG", cdg_path), ("split", split_path), ("output", output_path)]:
        if not p.exists():
            print(f"Error: {label} file not found: {p}", file=sys.stderr)
            sys.exit(1)

    if not key_path.exists():
        print(f"Error: SSH key not found: {key_path}", file=sys.stderr)
        sys.exit(1)

    receipt = generate_receipt(
        bounty_id=args.bounty_id,
        cdg_path=cdg_path,
        split_path=split_path,
        output_path=output_path,
        metric_name=args.metric_name or "loss",
        metric_value=float(args.metric_value) if args.metric_value else 0.0,
    )
    signed = sign_receipt(receipt, key_path)

    receipt_path = Path(args.receipt_output) if args.receipt_output else Path("receipt.json")
    save_signed_receipt(signed, receipt_path)
    print(f"Signed receipt saved to {receipt_path}")


def _cmd_receipt_verify(args: argparse.Namespace) -> None:
    """Verify a signed execution receipt."""
    from ageom.receipt import load_signed_receipt, verify_receipt

    receipt_path = Path(args.receipt)
    signers_path = Path(args.allowed_signers)

    if not receipt_path.exists():
        print(f"Error: receipt not found: {receipt_path}", file=sys.stderr)
        sys.exit(1)
    if not signers_path.exists():
        print(f"Error: allowed signers file not found: {signers_path}", file=sys.stderr)
        sys.exit(1)

    signed = load_signed_receipt(receipt_path)
    if verify_receipt(signed, signers_path):
        print("Verification: PASS")
    else:
        print("Verification: FAIL", file=sys.stderr)
        sys.exit(1)
