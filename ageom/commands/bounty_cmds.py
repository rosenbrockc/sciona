"""CLI commands for the Dead-End Flare / bounty system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_bounty_generate(args: argparse.Namespace) -> None:
    """Generate Dead-End Flare from a completed optimization run directory.

    Reads trial_history.json from --run-dir.
    """
    from ageom.principal.flare import FlarePayload, generate_flare, write_flare_config

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    history_path = run_dir / "trial_history.json"
    if not history_path.exists():
        print(f"Error: trial_history.json not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    history = json.loads(history_path.read_text(encoding="utf-8"))
    domain_tags = getattr(args, "domain_tags", None) or []

    # Reconstruct a minimal final_state dict from available data
    final_state: dict = {
        "trial_history": history,
        "goal": "",
        "metric": "",
        "best_loss": float("inf"),
    }
    # Try to find best loss from history
    for entry in history:
        loss = entry.get("loss", float("inf"))
        if loss < final_state["best_loss"]:
            final_state["best_loss"] = loss

    flare = generate_flare(final_state, domain_tags=domain_tags)

    output_path = Path(args.output) if args.output else run_dir / "flare.yml"
    write_flare_config(flare, output_path)
    print(f"Flare saved to {output_path}")
