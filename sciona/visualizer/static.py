"""Static file mounting for the visualizer app."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles that disables browser caching."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        original_send = send

        async def send_with_no_cache(message: dict) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"cache-control", b"no-cache, no-store, must-revalidate"))
                message["headers"] = headers
            await original_send(message)

        await super().__call__(scope, receive, send_with_no_cache)


def mount_static_assets(app: FastAPI) -> None:
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/", NoCacheStaticFiles(directory=str(static_dir), html=True), name="static")
