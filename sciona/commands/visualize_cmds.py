"""Command for browser-based CDG visualization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socketserver
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from pathlib import Path


def _cmd_visualize(args: argparse.Namespace) -> None:
    """Open browser-based CDG visualization."""

    static_dir = Path(__file__).resolve().parent.parent / "static"
    if not static_dir.exists():
        print(f"Error: static directory not found at {static_dir}", file=sys.stderr)
        sys.exit(1)

    # API mode: start FastAPI with uvicorn
    if getattr(args, "api", False):
        try:
            import uvicorn
        except ImportError:
            print(
                "Error: uvicorn not installed. Install with: pip install 'sciona[visualizer]'",
                file=sys.stderr,
            )
            sys.exit(1)

        port = args.port or 8080
        url = f"http://127.0.0.1:{port}"
        print(f"Starting CDG Visualizer API at {url}")
        print(f"Telemetry dashboard: {url}/dashboard.html")
        print("Press Ctrl+C to stop")

        threading.Thread(
            target=webbrowser.open, args=(url,), daemon=True
        ).start()

        uvicorn.run(
            "sciona.visualizer_api:app",
            host="127.0.0.1",
            port=port,
            log_level="info",
            reload=getattr(args, "reload", False),
        )
        return

    default_cdg = static_dir / "default_cdg.json"

    # If a CDG file was provided, validate and copy it
    if args.cdg_file:
        cdg_path = Path(args.cdg_file)
        if not cdg_path.exists():
            print(f"Error: CDG file not found: {cdg_path}", file=sys.stderr)
            sys.exit(1)
        try:
            with open(cdg_path) as f:
                data = json.load(f)
            if not isinstance(data.get("nodes"), list):
                print("Error: CDG JSON must contain a 'nodes' array", file=sys.stderr)
                sys.exit(1)
            if not isinstance(data.get("edges"), list):
                print("Error: CDG JSON must contain an 'edges' array", file=sys.stderr)
                sys.exit(1)
        except json.JSONDecodeError as exc:
            print(f"Error: invalid JSON in {cdg_path}: {exc}", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(str(cdg_path), str(default_cdg))

    try:
        if args.no_serve:
            # Open file:// directly
            index_html = static_dir / "index.html"
            url = index_html.as_uri()
            print(f"Opening {url}")
            webbrowser.open(url)
        else:
            # Start local HTTP server
            original_dir = os.getcwd()
            os.chdir(str(static_dir))

            handler = SimpleHTTPRequestHandler

            with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
                port = httpd.server_address[1]
                url = f"http://127.0.0.1:{port}/index.html"
                print(f"Serving CDG visualizer at {url}")
                print("Press Ctrl+C to stop")

                # Open browser in a thread so we don't block the server
                threading.Thread(
                    target=webbrowser.open, args=(url,), daemon=True
                ).start()

                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\nShutting down server")
                finally:
                    os.chdir(original_dir)
    finally:
        # Clean up default_cdg.json
        if default_cdg.exists():
            default_cdg.unlink()
