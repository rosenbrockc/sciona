import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is required for static visualizer JS checks")
def test_static_visualizer_js_syntax():
    subprocess.run(
        [NODE, "frontend/scripts/check_static_visualizer.mjs"],
        cwd=ROOT,
        check=True,
    )


@pytest.mark.skipif(NODE is None, reason="node is required for static visualizer JS checks")
def test_static_visualizer_js_smoke():
    subprocess.run(
        [NODE, "--test", "frontend/tests/static_visualizer.test.mjs"],
        cwd=ROOT,
        check=True,
    )
