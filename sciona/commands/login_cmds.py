"""CLI commands for platform authentication."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _get_config_path() -> Path:
    """Return the path to the CLI config file."""
    return Path.home() / ".sciona" / "config.json"


def _save_token(token: str, api_url: str) -> Path:
    """Persist the JWT token to ~/.sciona/config.json."""
    config_path = _get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config: dict = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    config["access_token"] = token
    config["api_url"] = api_url
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def _load_token() -> tuple[str, str]:
    """Load the JWT token and API URL from config. Returns (token, api_url)."""
    config_path = _get_config_path()
    if not config_path.exists():
        return "", ""
    try:
        config = json.loads(config_path.read_text())
        return config.get("access_token", ""), config.get("api_url", "")
    except (json.JSONDecodeError, OSError):
        return "", ""


async def _cmd_login(args: argparse.Namespace) -> None:
    """Authenticate with the SCIONA platform via GitHub device flow."""
    try:
        import httpx
    except ImportError:
        print("Error: httpx is required for login. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)

    api_url = (args.api_url or "https://api.sciona.dev").rstrip("/")

    async with httpx.AsyncClient() as client:
        # Start device flow
        resp = await client.get(f"{api_url}/auth/github/device")
        if resp.status_code != 200:
            print(f"Error: failed to start device flow: {resp.text}", file=sys.stderr)
            sys.exit(1)

        data = resp.json()
        user_code = data["user_code"]
        device_code = data["device_code"]
        verification_uri = data["verification_uri"]
        interval = data.get("interval", 5)
        expires_in = data.get("expires_in", 900)

        print(f"\nOpen {verification_uri} and enter code: {user_code}\n")
        print("Waiting for authorization...")

        deadline = time.time() + expires_in
        while time.time() < deadline:
            await __import__("asyncio").sleep(interval)

            poll_resp = await client.post(
                f"{api_url}/auth/github/device/poll",
                params={"device_code": device_code},
            )

            if poll_resp.status_code != 200:
                continue

            poll_data = poll_resp.json()
            if "access_token" in poll_data:
                config_path = _save_token(poll_data["access_token"], api_url)
                print(f"Authenticated! Token saved to {config_path}")
                return

            status = poll_data.get("status", "")
            if status == "slow_down":
                interval = poll_data.get("interval", interval + 5)

        print("Error: authorization timed out", file=sys.stderr)
        sys.exit(1)
