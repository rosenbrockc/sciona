"""Tests for CDG visualization: static files and CLI subcommand."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from sciona.cli import main

STATIC_DIR = Path(__file__).resolve().parent.parent / "sciona" / "static"


@pytest.fixture
def sample_cdg(tmp_path):
    """Write a minimal valid CDG JSON and return its path."""
    data = {
        "nodes": [
            {
                "node_id": "root",
                "name": "Root Task",
                "description": "Top-level goal",
                "concept_type": "divide_and_conquer",
                "status": "decomposed",
                "children": ["child1"],
                "depth": 0,
            },
            {
                "node_id": "child1",
                "parent_id": "root",
                "name": "Child Step",
                "description": "A leaf step",
                "concept_type": "sorting",
                "status": "atomic",
                "matched_primitive": "merge_sort",
                "type_signature": "list[int] -> list[int]",
                "inputs": [
                    {"name": "arr", "type_desc": "list[int]", "constraints": ""}
                ],
                "outputs": [
                    {"name": "sorted", "type_desc": "list[int]", "constraints": ""}
                ],
                "depth": 1,
                "children": [],
            },
        ],
        "edges": [
            {
                "source_id": "root",
                "target_id": "child1",
                "output_name": "data",
                "input_name": "arr",
                "source_type": "list[int]",
                "target_type": "list[int]",
                "requires_glue": False,
            },
        ],
        "metadata": {
            "goal": "Sort an array",
            "paradigm": "divide_and_conquer",
            "thread_id": "test-thread-abc",
        },
    }
    path = tmp_path / "cdg.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def bad_cdg(tmp_path):
    """Write a CDG JSON missing the 'nodes' key."""
    data = {"edges": [], "metadata": {}}
    path = tmp_path / "bad_cdg.json"
    path.write_text(json.dumps(data))
    return path


class TestStaticFilesExist:
    def test_index_html_exists(self):
        assert (STATIC_DIR / "index.html").is_file()

    def test_style_css_exists(self):
        assert (STATIC_DIR / "style.css").is_file()

    def test_app_js_exists(self):
        assert (STATIC_DIR / "app.js").is_file()


class TestStaticFileContents:
    def test_index_html_has_cytoscape_cdn(self):
        html = (STATIC_DIR / "index.html").read_text()
        assert "cytoscape" in html
        assert "dagre" in html
        assert "app.js" in html

    def test_index_html_has_drop_zone(self):
        html = (STATIC_DIR / "index.html").read_text()
        assert "drop-zone" in html

    def test_index_html_has_detail_panel(self):
        html = (STATIC_DIR / "index.html").read_text()
        assert "detail-panel" in html

    def test_app_js_has_status_colors(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert "CONCEPT_FAMILY" in js
        assert "FAMILY_COLORS" in js
        assert "pending" in js
        assert "decomposed" in js
        assert "atomic" in js
        assert "rejected" in js
        assert "high_risk" in js

    def test_app_js_has_concept_shapes(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert "STATUS_SHAPES" in js
        assert "divide_and_conquer" in js
        assert "sorting" in js

    def test_style_css_has_status_classes(self):
        css = (STATIC_DIR / "style.css").read_text()
        assert ".status-pending" in css
        assert ".status-decomposed" in css
        assert ".status-atomic" in css
        assert ".status-rejected" in css
        assert ".status-high_risk" in css


class TestCLIParserAcceptsVisualize:
    def test_visualize_no_args(self):
        """Parser should accept 'visualize' with no arguments."""
        with patch("sys.argv", ["sciona", "visualize", "--no-serve"]):
            with patch("sciona.cli._cmd_visualize") as mock_cmd:
                main()
                mock_cmd.assert_called_once()
                args = mock_cmd.call_args[0][0]
                assert args.command == "visualize"
                assert args.cdg_file is None
                assert args.port == 0
                assert args.no_serve is True

    def test_visualize_with_cdg_file(self, sample_cdg):
        """Parser should accept a positional cdg_file argument."""
        with patch("sys.argv", ["sciona", "visualize", str(sample_cdg), "--no-serve"]):
            with patch("sciona.cli._cmd_visualize") as mock_cmd:
                main()
                args = mock_cmd.call_args[0][0]
                assert args.cdg_file == str(sample_cdg)

    def test_visualize_with_port(self):
        """Parser should accept --port."""
        with patch("sys.argv", ["sciona", "visualize", "--port", "8080"]):
            with patch("sciona.cli._cmd_visualize") as mock_cmd:
                main()
                args = mock_cmd.call_args[0][0]
                assert args.port == 8080


class TestCDGCopyAndCleanup:
    def test_copies_cdg_to_default(self, sample_cdg):
        """When a CDG file is given, it should be copied to static/default_cdg.json."""
        default_cdg = STATIC_DIR / "default_cdg.json"
        assert not default_cdg.exists(), "default_cdg.json should not exist before test"

        with patch("sys.argv", ["sciona", "visualize", str(sample_cdg), "--no-serve"]):
            with patch("webbrowser.open"):
                main()

        # After the command, default_cdg.json should have been cleaned up
        assert (
            not default_cdg.exists()
        ), "default_cdg.json should be cleaned up after exit"

    def test_rejects_invalid_cdg(self, bad_cdg):
        """Should exit with error if CDG JSON is missing 'nodes'."""
        with patch("sys.argv", ["sciona", "visualize", str(bad_cdg), "--no-serve"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_rejects_nonexistent_file(self):
        """Should exit with error if CDG file doesn't exist."""
        with patch(
            "sys.argv", ["sciona", "visualize", "/nonexistent/cdg.json", "--no-serve"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_rejects_invalid_json(self, tmp_path):
        """Should exit with error if file is not valid JSON."""
        bad_file = tmp_path / "not_json.json"
        bad_file.write_text("this is not json {{{")
        with patch("sys.argv", ["sciona", "visualize", str(bad_file), "--no-serve"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestNoServeMode:
    def test_opens_file_url(self, sample_cdg):
        """--no-serve should open a file:// URL."""
        with patch("sys.argv", ["sciona", "visualize", str(sample_cdg), "--no-serve"]):
            with patch("webbrowser.open") as mock_open:
                main()
                mock_open.assert_called_once()
                url = mock_open.call_args[0][0]
                assert url.startswith("file://")
                assert "index.html" in url
