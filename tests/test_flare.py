"""Tests for the Dead-End Flare protocol."""

from __future__ import annotations

from pathlib import Path

import pytest

from sciona.principal.flare import (
    BestStructure,
    FlarePayload,
    generate_flare,
    write_flare_config,
)


def _minimal_final_state(
    *, goal: str = "test goal", best_loss: float = 0.42
) -> dict:
    """Build a minimal final_state dict for testing."""
    return {
        "goal": goal,
        "metric": "latency",
        "best_loss": best_loss,
        "trial_history": [
            {
                "trial": 1,
                "loss": 0.5,
                "structure": {
                    "node_count": 4,
                    "edge_count": 3,
                    "topo_hash": "abc123",
                    "primitive_signature": "sig1",
                    "atomic_primitives": {"n1": "atom_a", "n2": "atom_b"},
                },
            },
            {
                "trial": 2,
                "loss": 0.42,
                "structure": {
                    "node_count": 5,
                    "edge_count": 4,
                    "topo_hash": "def456",
                    "primitive_signature": "sig2",
                    "atomic_primitives": {"n1": "atom_a", "n3": "atom_c"},
                },
            },
        ],
    }


class TestFlarePayloadSchema:
    def test_round_trip(self):
        payload = FlarePayload(
            goal="test",
            objective="latency",
            execution_metric="latency",
            best_metric_value=0.5,
            metric_name="latency",
        )
        data = payload.model_dump()
        restored = FlarePayload.model_validate(data)
        assert restored == payload

    def test_defaults(self):
        payload = FlarePayload(
            goal="g",
            objective="o",
            execution_metric="e",
            best_metric_value=1.0,
            metric_name="m",
        )
        assert payload.domain_tags == []
        assert payload.atoms_tried == []
        assert payload.best_structure.node_count == 0


class TestGenerateFlare:
    def test_from_state(self):
        state = _minimal_final_state()
        flare = generate_flare(state)
        assert flare.goal == "test goal"
        assert flare.best_metric_value == 0.42
        assert "atom_a" in flare.atoms_tried
        assert "atom_b" in flare.atoms_tried
        assert "atom_c" in flare.atoms_tried
        assert flare.max_graph_nodes == 5
        assert flare.max_graph_edges == 4

    def test_best_structure_from_lowest_loss(self):
        state = _minimal_final_state()
        flare = generate_flare(state)
        # Trial 2 has lower loss
        assert flare.best_structure.topo_hash == "def456"
        assert flare.best_structure.node_count == 5

    def test_domain_tags_passthrough(self):
        state = _minimal_final_state()
        flare = generate_flare(state, domain_tags=["crystallography"])
        assert flare.domain_tags == ["crystallography"]

    def test_omits_private_data(self):
        state = _minimal_final_state()
        # Add private data that should NOT appear in the flare
        state["trial_history"][0]["ucb_score"] = 1.5
        state["trial_history"][0]["gradient_scores"] = [0.1, 0.2]
        state["sources_path"] = "/private/sources.yml"

        flare = generate_flare(state)
        flare_dict = flare.model_dump()
        flare_str = str(flare_dict)

        assert "ucb" not in flare_str.lower()
        assert "gradient" not in flare_str.lower()
        assert "sources.yml" not in flare_str
        assert "/private" not in flare_str

    def test_empty_history(self):
        state = {
            "goal": "empty",
            "metric": "latency",
            "best_loss": float("inf"),
            "trial_history": [],
        }
        flare = generate_flare(state)
        assert flare.goal == "empty"
        assert flare.atoms_tried == []
        assert flare.best_structure.node_count == 0
        assert flare.max_graph_nodes == 0


class TestWriteFlareConfig:
    def test_writes_yaml(self, tmp_path: Path):
        payload = FlarePayload(
            goal="test goal",
            objective="latency",
            execution_metric="latency",
            best_metric_value=0.42,
            metric_name="latency",
            domain_tags=["crystallography"],
            atoms_tried=["atom_a", "atom_b"],
            best_structure=BestStructure(
                node_count=5,
                edge_count=4,
                topo_hash="abc123",
            ),
        )
        out = write_flare_config(payload, tmp_path / "flare.yml")
        assert out.exists()
        content = out.read_text()
        assert "goal:" in content
        assert "test goal" in content
        assert "crystallography" in content
        assert "atom_a" in content
        assert "node_count: 5" in content

    def test_read_back(self, tmp_path: Path):
        payload = FlarePayload(
            goal="round-trip",
            objective="memory",
            execution_metric="memory",
            best_metric_value=1.23,
            metric_name="memory",
        )
        out = write_flare_config(payload, tmp_path / "flare.yml")
        content = out.read_text()
        assert "round-trip" in content
        assert "1.23" in content
