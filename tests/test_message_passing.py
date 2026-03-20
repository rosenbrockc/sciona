"""Tests for recursive factor graph / message-passing support.

Covers:
- _generate_message_passing_witness() witness templates
- _detect_message_passing_cycle() cycle detection
- _simulate_message_passing_iterative() convergence and deadlock
- GhostSimReport cyclic fields
- generate_atom_wrappers() memoization preamble
- repair_message_cycle routing
"""

from __future__ import annotations


from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.ingester.emitter import (
    _MESSAGE_PASSING_CONCEPT_TYPES,
    _generate_message_passing_witness,
    generate_atom_wrappers,
    generate_ghost_witnesses,
)
from sciona.ingester.models import MacroAtomSpec
from sciona.synthesizer.ghost_sim import (
    GhostSimReport,
    _detect_message_passing_cycle,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mp_atom(name: str, inputs: list[IOSpec], outputs: list[IOSpec]) -> MacroAtomSpec:
    return MacroAtomSpec(
        name=name,
        description=f"Test message-passing atom: {name}",
        concept_type=ConceptType.MESSAGE_PASSING,
        inputs=inputs,
        outputs=outputs,
    )


def _bp_atoms() -> list[MacroAtomSpec]:
    """Build the four canonical belief propagation atoms."""
    return [
        _mp_atom(
            "Variable to Factor",
            [
                IOSpec(name="incoming_messages", type_desc="dict[str, ndarray]"),
                IOSpec(name="memo_state", type_desc="dict[str, ndarray]"),
            ],
            [IOSpec(name="var_messages", type_desc="dict[str, ndarray]")],
        ),
        _mp_atom(
            "Factor to Variable",
            [
                IOSpec(name="var_messages", type_desc="dict[str, ndarray]"),
                IOSpec(name="factor_potentials", type_desc="dict[str, ndarray]"),
                IOSpec(name="memo_state", type_desc="dict[str, ndarray]"),
            ],
            [IOSpec(name="factor_messages", type_desc="dict[str, ndarray]")],
        ),
        _mp_atom(
            "Marginal Computation",
            [
                IOSpec(name="factor_messages", type_desc="dict[str, ndarray]"),
                IOSpec(name="var_messages", type_desc="dict[str, ndarray]"),
            ],
            [IOSpec(name="marginals", type_desc="dict[str, ndarray]")],
        ),
        _mp_atom(
            "Memoization State",
            [
                IOSpec(name="var_messages", type_desc="dict[str, ndarray]"),
                IOSpec(name="factor_messages", type_desc="dict[str, ndarray]"),
            ],
            [
                IOSpec(name="memo_state", type_desc="dict[str, ndarray]"),
                IOSpec(name="converged", type_desc="bool"),
            ],
        ),
    ]


def _bp_nodes_and_edges():
    """Build AlgorithmicNode + DependencyEdge lists for a cyclic BP graph."""
    var_to_factor = AlgorithmicNode(
        node_id="variable_to_factor",
        name="Variable to Factor",
        description="Var-to-factor messages",
        concept_type=ConceptType.MESSAGE_PASSING,
        status=NodeStatus.ATOMIC,
        inputs=[
            IOSpec(name="incoming_messages", type_desc="dict[str, ndarray]"),
            IOSpec(name="memo_state", type_desc="dict[str, ndarray]"),
        ],
        outputs=[IOSpec(name="var_messages", type_desc="dict[str, ndarray]")],
        depth=1,
    )
    factor_to_var = AlgorithmicNode(
        node_id="factor_to_variable",
        name="Factor to Variable",
        description="Factor-to-var messages",
        concept_type=ConceptType.MESSAGE_PASSING,
        status=NodeStatus.ATOMIC,
        inputs=[
            IOSpec(name="var_messages", type_desc="dict[str, ndarray]"),
            IOSpec(name="factor_potentials", type_desc="dict[str, ndarray]"),
            IOSpec(name="memo_state", type_desc="dict[str, ndarray]"),
        ],
        outputs=[IOSpec(name="factor_messages", type_desc="dict[str, ndarray]")],
        depth=1,
    )
    memo = AlgorithmicNode(
        node_id="memoization_state",
        name="Memoization State",
        description="Memo state for convergence",
        concept_type=ConceptType.MESSAGE_PASSING,
        status=NodeStatus.ATOMIC,
        inputs=[
            IOSpec(name="var_messages", type_desc="dict[str, ndarray]"),
            IOSpec(name="factor_messages", type_desc="dict[str, ndarray]"),
        ],
        outputs=[
            IOSpec(name="memo_state", type_desc="dict[str, ndarray]"),
            IOSpec(name="converged", type_desc="bool"),
        ],
        depth=1,
    )
    nodes = [var_to_factor, factor_to_var, memo]
    edges = [
        DependencyEdge(
            source_id="variable_to_factor",
            target_id="factor_to_variable",
            output_name="var_messages",
            input_name="var_messages",
            source_type="dict[str, ndarray]",
            target_type="dict[str, ndarray]",
        ),
        DependencyEdge(
            source_id="factor_to_variable",
            target_id="memoization_state",
            output_name="factor_messages",
            input_name="factor_messages",
            source_type="dict[str, ndarray]",
            target_type="dict[str, ndarray]",
        ),
        DependencyEdge(
            source_id="variable_to_factor",
            target_id="memoization_state",
            output_name="var_messages",
            input_name="var_messages",
            source_type="dict[str, ndarray]",
            target_type="dict[str, ndarray]",
        ),
        # Cycle: memo -> var_to_factor
        DependencyEdge(
            source_id="memoization_state",
            target_id="variable_to_factor",
            output_name="memo_state",
            input_name="memo_state",
            source_type="dict[str, ndarray]",
            target_type="dict[str, ndarray]",
        ),
        # Cycle: memo -> factor_to_var
        DependencyEdge(
            source_id="memoization_state",
            target_id="factor_to_variable",
            output_name="memo_state",
            input_name="memo_state",
            source_type="dict[str, ndarray]",
            target_type="dict[str, ndarray]",
        ),
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# _generate_message_passing_witness tests
# ---------------------------------------------------------------------------


class TestGenerateMessagePassingWitness:
    def test_variable_to_factor_witness(self):
        atom = _bp_atoms()[0]  # Variable to Factor
        lines = _generate_message_passing_witness(
            atom, "variable_to_factor", "witness_variable_to_factor"
        )
        source = "\n".join(lines)
        assert "def witness_variable_to_factor(" in source
        assert "_MEMO_CACHE" in source
        assert "variable_to_factor" in source
        assert "incoming_messages" in source

    def test_factor_to_variable_witness(self):
        atom = _bp_atoms()[1]  # Factor to Variable
        lines = _generate_message_passing_witness(
            atom, "factor_to_variable", "witness_factor_to_variable"
        )
        source = "\n".join(lines)
        assert "def witness_factor_to_variable(" in source
        assert "factor_to_variable" in source
        assert "var_messages" in source
        assert "factor_potentials" in source

    def test_marginal_witness(self):
        atom = _bp_atoms()[2]  # Marginal Computation
        lines = _generate_message_passing_witness(
            atom, "marginal_computation", "witness_marginal_computation"
        )
        source = "\n".join(lines)
        assert "def witness_marginal_computation(" in source
        assert '"marginal"' in source
        assert "factor_messages" in source

    def test_memoization_state_witness(self):
        atom = _bp_atoms()[3]  # Memoization State
        lines = _generate_message_passing_witness(
            atom, "memoization_state", "witness_memoization_state"
        )
        source = "\n".join(lines)
        assert "def witness_memoization_state(" in source
        assert "converged" in source
        assert "tuple[dict[str, AbstractArray], bool]" in source

    def test_generic_fallback_witness(self):
        atom = _mp_atom(
            "Custom Message Node",
            [IOSpec(name="data", type_desc="ndarray")],
            [IOSpec(name="output", type_desc="ndarray")],
        )
        lines = _generate_message_passing_witness(
            atom, "custom_message_node", "witness_custom_message_node"
        )
        source = "\n".join(lines)
        assert "def witness_custom_message_node(" in source
        assert "_MEMO_CACHE" in source


# ---------------------------------------------------------------------------
# generate_ghost_witnesses — MESSAGE_PASSING integration
# ---------------------------------------------------------------------------


class TestGhostWitnessesMessagePassing:
    def test_memo_cache_preamble_present(self):
        atoms = _bp_atoms()
        source, name_map = generate_ghost_witnesses(atoms)
        assert "_MEMO_CACHE: dict = {}" in source
        assert "def _clear_memo_cache()" in source

    def test_all_witness_names_mapped(self):
        atoms = _bp_atoms()
        _, name_map = generate_ghost_witnesses(atoms)
        assert "Variable to Factor" in name_map
        assert "Factor to Variable" in name_map
        assert "Marginal Computation" in name_map
        assert "Memoization State" in name_map

    def test_no_dsp_fallback_for_mp_atoms(self):
        atoms = _bp_atoms()
        source, _ = generate_ghost_witnesses(atoms)
        # MP atoms should NOT get AbstractSignal DSP witnesses
        assert "AbstractSignal(" not in source
        assert "sampling_rate=" not in source

    def test_no_memo_cache_without_mp_atoms(self):
        atoms = [
            MacroAtomSpec(
                name="Plain Node",
                concept_type=ConceptType.CUSTOM,
                inputs=[IOSpec(name="x", type_desc="ndarray")],
                outputs=[IOSpec(name="y", type_desc="ndarray")],
            )
        ]
        source, _ = generate_ghost_witnesses(atoms)
        assert "_MEMO_CACHE" not in source


# ---------------------------------------------------------------------------
# generate_atom_wrappers — memoization for MESSAGE_PASSING
# ---------------------------------------------------------------------------


class TestAtomWrappersMessagePassing:
    def test_memo_preamble_emitted(self):
        atoms = _bp_atoms()
        source = generate_atom_wrappers(atoms, [], {})
        assert "_MEMO: dict = {}" in source
        assert "def _memo_key(" in source

    def test_memoized_wrapper_body(self):
        atoms = [_bp_atoms()[0]]  # Variable to Factor
        source = generate_atom_wrappers(atoms, [], {})
        assert '_key = _memo_key("variable_to_factor"' in source
        assert "if _key in _MEMO:" in source
        assert "return _MEMO[_key]" in source

    def test_no_memo_preamble_for_non_mp(self):
        atoms = [
            MacroAtomSpec(
                name="Plain Node",
                concept_type=ConceptType.CUSTOM,
                inputs=[IOSpec(name="x", type_desc="ndarray")],
                outputs=[IOSpec(name="y", type_desc="ndarray")],
            )
        ]
        source = generate_atom_wrappers(atoms, [], {})
        assert "_MEMO:" not in source
        assert "_memo_key" not in source


# ---------------------------------------------------------------------------
# _detect_message_passing_cycle
# ---------------------------------------------------------------------------


class TestDetectMessagePassingCycle:
    def test_no_cycle_returns_empty(self):
        """Acyclic graph: toposort completes, no cycle detected."""
        n1 = AlgorithmicNode(
            node_id="a",
            name="A",
            description="node A",
            concept_type=ConceptType.MESSAGE_PASSING,
            status=NodeStatus.ATOMIC,
            depth=1,
        )
        n2 = AlgorithmicNode(
            node_id="b",
            name="B",
            description="node B",
            concept_type=ConceptType.MESSAGE_PASSING,
            status=NodeStatus.ATOMIC,
            depth=1,
        )
        edge = DependencyEdge(
            source_id="a",
            target_id="b",
            output_name="out",
            input_name="in",
            source_type="Any",
            target_type="Any",
        )
        cycle_ids, is_mp = _detect_message_passing_cycle([n1, n2], [edge])
        assert cycle_ids == set()
        assert is_mp is False

    def test_message_passing_cycle_detected(self):
        """Cyclic BP graph: all cycle nodes are MESSAGE_PASSING."""
        nodes, edges = _bp_nodes_and_edges()
        cycle_ids, is_mp = _detect_message_passing_cycle(nodes, edges)
        assert len(cycle_ids) > 0
        assert is_mp is True

    def test_mixed_type_cycle_not_message_passing(self):
        """Cycle with a non-MESSAGE_PASSING node should not be flagged."""
        n1 = AlgorithmicNode(
            node_id="a",
            name="A",
            description="node A",
            concept_type=ConceptType.MESSAGE_PASSING,
            status=NodeStatus.ATOMIC,
            depth=1,
        )
        n2 = AlgorithmicNode(
            node_id="b",
            name="B",
            description="node B",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            depth=1,
        )
        edges = [
            DependencyEdge(
                source_id="a",
                target_id="b",
                output_name="out",
                input_name="in",
                source_type="Any",
                target_type="Any",
            ),
            DependencyEdge(
                source_id="b",
                target_id="a",
                output_name="out",
                input_name="in",
                source_type="Any",
                target_type="Any",
            ),
        ]
        cycle_ids, is_mp = _detect_message_passing_cycle([n1, n2], edges)
        assert len(cycle_ids) > 0
        assert is_mp is False


# ---------------------------------------------------------------------------
# GhostSimReport cyclic fields
# ---------------------------------------------------------------------------


class TestGhostSimReportCyclicFields:
    def test_default_values(self):
        report = GhostSimReport()
        assert report.cyclic_deadlock is False
        assert report.deadlock_nodes == []
        assert report.iterations_used == 0

    def test_populated_fields(self):
        report = GhostSimReport(
            ran=True,
            cyclic_deadlock=True,
            deadlock_nodes=["Variable to Factor", "Memoization State"],
            iterations_used=42,
        )
        assert report.cyclic_deadlock is True
        assert len(report.deadlock_nodes) == 2
        assert report.iterations_used == 42


# ---------------------------------------------------------------------------
# LLM router and config integration
# ---------------------------------------------------------------------------


class TestLLMRouterKey:
    def test_key_in_all_prompt_keys(self):
        from sciona.llm_router import ALL_PROMPT_KEYS, INGESTER_FIX_MESSAGE_CYCLE

        assert INGESTER_FIX_MESSAGE_CYCLE in ALL_PROMPT_KEYS
        assert INGESTER_FIX_MESSAGE_CYCLE == "ingester_fix_message_cycle"


class TestConfigFields:
    def test_message_cycle_config_fields_exist(self):
        from sciona.config import AgeomConfig

        config = AgeomConfig()
        assert hasattr(config, "ingester_fix_message_cycle_llm_provider")
        assert hasattr(config, "ingester_fix_message_cycle_llm_model")
        assert config.ingester_fix_message_cycle_llm_provider == "llama_cpp"
        assert config.ingester_fix_message_cycle_llm_model == "qwen3:14b"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_fix_message_cycle_prompts_exist(self):
        from sciona.ingester.prompts import (
            FIX_MESSAGE_CYCLE_SYSTEM,
            FIX_MESSAGE_CYCLE_USER,
        )

        assert "cyclic deadlock" in FIX_MESSAGE_CYCLE_SYSTEM
        assert "{deadlock_nodes}" in FIX_MESSAGE_CYCLE_USER
        assert "{cycle_edges}" in FIX_MESSAGE_CYCLE_USER
        assert "{witness_source}" in FIX_MESSAGE_CYCLE_USER


# ---------------------------------------------------------------------------
# Graph routing — repair_message_cycle
# ---------------------------------------------------------------------------


class TestGraphRouting:
    def test_route_after_ghost_to_message_cycle_repair(self):
        from sciona.ingester.graph import route_after_ghost
        from sciona.ingester.models import IngestionBundle
        from sciona.architect.handoff import CDGExport

        bundle = IngestionBundle(
            cdg=CDGExport(nodes=[], edges=[]),
            ghost_sim_report={
                "cyclic_deadlock": True,
                "deadlock_nodes": ["Variable to Factor"],
            },
        )
        state = {
            "ghost_passed": False,
            "ghost_repair_count": 0,
            "bundle": bundle,
        }
        assert route_after_ghost(state) == "repair_message_cycle"

    def test_route_after_ghost_normal_failure(self):
        from sciona.ingester.graph import route_after_ghost
        from sciona.ingester.models import IngestionBundle
        from sciona.architect.handoff import CDGExport

        bundle = IngestionBundle(
            cdg=CDGExport(nodes=[], edges=[]),
            ghost_sim_report={
                "cyclic_deadlock": False,
            },
        )
        state = {
            "ghost_passed": False,
            "ghost_repair_count": 0,
            "bundle": bundle,
        }
        assert route_after_ghost(state) == "repair_ghost"


# ---------------------------------------------------------------------------
# _MESSAGE_PASSING_CONCEPT_TYPES sanity
# ---------------------------------------------------------------------------


class TestConceptTypeSet:
    def test_message_passing_in_set(self):
        assert ConceptType.MESSAGE_PASSING in _MESSAGE_PASSING_CONCEPT_TYPES

    def test_set_is_frozen(self):
        assert isinstance(_MESSAGE_PASSING_CONCEPT_TYPES, frozenset)
