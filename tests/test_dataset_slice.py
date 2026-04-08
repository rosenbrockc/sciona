from __future__ import annotations

from sciona.principal.dataset_slice import apply_relative_dataset_slice


def test_apply_relative_dataset_slice_anchors_to_loaded_collection_start() -> None:
    calls: list[tuple[float | None, float | None]] = []

    class _Collection:
        def __init__(self) -> None:
            self.data = None

        def autoload(self):
            self.data = type("Data", (), {"min": 42.5})()
            return self.data

        def slice(self, start=None, stop=None):
            calls.append((start, stop))

    collection = _Collection()
    apply_relative_dataset_slice(collection, start_s=5.0, stop_s=65.0)

    assert calls == [(47.5, 107.5)]


def test_apply_relative_dataset_slice_falls_back_when_no_anchor_exists() -> None:
    calls: list[tuple[float | None, float | None]] = []

    class _Collection:
        data = None

        def autoload(self):
            self.data = type("Data", (), {"min": None})()
            return self.data

        def slice(self, start=None, stop=None):
            calls.append((start, stop))

    collection = _Collection()
    apply_relative_dataset_slice(collection, start_s=None, stop_s=300.0)

    assert calls == [(None, 300.0)]


def test_apply_relative_dataset_slice_defaults_start_to_anchor_for_stop_only() -> None:
    calls: list[tuple[float | None, float | None]] = []

    class _Collection:
        data = None

        def autoload(self):
            self.data = type("Data", (), {"min": 100.0})()
            return self.data

        def slice(self, start=None, stop=None):
            calls.append((start, stop))

    collection = _Collection()
    apply_relative_dataset_slice(collection, stop_s=30.0)

    assert calls == [(100.0, 130.0)]
