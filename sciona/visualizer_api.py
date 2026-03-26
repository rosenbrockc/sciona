"""Compatibility wrapper for the visualizer FastAPI app."""

from __future__ import annotations

from sciona.visualizer.app import app, create_app

__all__ = ["app", "create_app"]
