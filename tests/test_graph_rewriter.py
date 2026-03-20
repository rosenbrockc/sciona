import pytest
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.architect.graph_rewriter import (
    GraphRewriter,
    GraphState,
    Morphism,
    PriorityStrategy,
    RewriteRule,
    _node_matches_pattern,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    nid: str,
    concept: ConceptType = ConceptType.CUSTOM,
    primitive: str | None = None,
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=nid,
        name=nid,
        description=nid,
        concept_type=concept,
        status=NodeStatus.ATOMIC,
        matched_primitive=primitive,
        inputs=[IOSpec(name="in", type_desc="np.ndarray")],
        outputs=[IOSpec(name="out", type_desc="np.ndarray")],
        type_signature=f"{nid} -> result",
    )


def _edge(src: str, tgt: str) -> DependencyEdge:
    return DependencyEdge(
        source_id=src,
        target_id=tgt,
        output_name="out",
        input_name="in",
        source_type="np.ndarray",
        target_type="np.ndarray",
    )


def _cdg(nodes: list[AlgorithmicNode], edges: list[DependencyEdge]) -> CDGExport:
    return CDGExport(nodes=nodes, edges=edges, metadata={})


# ---------------------------------------------------------------------------
# Monad / priority (existing tests, updated)
# ---------------------------------------------------------------------------


def test_graph_state_monad_success():
    initial = _cdg([], [])
    state = GraphState.success(initial)

    def transform(g):
        return GraphState.success(_cdg([_node("n1")], []))

    result = state.bind(transform)
    assert not result.is_failure
    assert len(result.unwrap().nodes) == 1


def test_graph_state_monad_failure_chaining():
    state = GraphState.success(_cdg([], []))

    def fail(g):
        return GraphState.failure("boom")

    def unreachable(g):
        pytest.fail("should not be called")

    result = state.bind(fail).bind(unreachable)
    assert result.is_failure
    with pytest.raises(RuntimeError, match="boom"):
        result.unwrap()


def test_priority_strategy():
    empty = _cdg([], [])
    morph = Morphism(node_map={}, edge_map={})
    r_low = RewriteRule("low", empty, empty, empty, morph, morph, priority=1)
    r_high = RewriteRule("high", empty, empty, empty, morph, morph, priority=10)
    assert PriorityStrategy().sort_rules([r_low, r_high])[0].name == "high"


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


def test_node_matches_pattern_by_primitive():
    pattern = _node("p", primitive="filter_signal_for_detection")
    assert _node_matches_pattern(pattern, _node("x", primitive="filter_signal_for_detection"))
    assert not _node_matches_pattern(pattern, _node("x", primitive="other"))
    assert not _node_matches_pattern(pattern, _node("x"))


def test_node_matches_pattern_by_concept_type():
    pattern = _node("p", concept=ConceptType.SIGNAL_FILTER)
    assert _node_matches_pattern(pattern, _node("x", concept=ConceptType.SIGNAL_FILTER))
    assert not _node_matches_pattern(pattern, _node("x", concept=ConceptType.ANALYSIS))


def test_node_matches_pattern_wildcard():
    pattern = _node("p", concept=ConceptType.CUSTOM)
    assert _node_matches_pattern(pattern, _node("x", concept=ConceptType.SIGNAL_FILTER))
    assert _node_matches_pattern(pattern, _node("x", primitive="anything"))


# ---------------------------------------------------------------------------
# _find_match
# ---------------------------------------------------------------------------


def test_find_match_empty_lhs():
    rw = GraphRewriter()
    morph = Morphism(node_map={}, edge_map={})
    rule = RewriteRule("r", _cdg([], []), _cdg([], []), _cdg([], []), morph, morph)
    graph = _cdg([_node("a")], [])
    m = rw._find_match(rule, graph)
    assert m is not None
    assert m.node_map == {}


def test_find_match_single_node():
    rw = GraphRewriter()
    lhs = _cdg([_node("l", primitive="foo")], [])
    morph = Morphism(node_map={}, edge_map={})
    rule = RewriteRule("r", lhs, lhs, _cdg([], []), morph, morph)
    graph = _cdg([_node("a", primitive="foo"), _node("b", primitive="bar")], [])
    m = rw._find_match(rule, graph)
    assert m is not None
    assert m.node_map["l"] == "a"


def test_find_match_edge_chain():
    rw = GraphRewriter()
    lhs = _cdg(
        [_node("l1", primitive="A"), _node("l2", primitive="B")],
        [_edge("l1", "l2")],
    )
    morph = Morphism(node_map={}, edge_map={})
    rule = RewriteRule("r", lhs, lhs, _cdg([], []), morph, morph)

    # Graph with A -> B edge
    graph = _cdg(
        [_node("g1", primitive="A"), _node("g2", primitive="B")],
        [_edge("g1", "g2")],
    )
    m = rw._find_match(rule, graph)
    assert m is not None
    assert m.node_map == {"l1": "g1", "l2": "g2"}


def test_find_match_no_match():
    rw = GraphRewriter()
    lhs = _cdg([_node("l", primitive="missing")], [])
    morph = Morphism(node_map={}, edge_map={})
    rule = RewriteRule("r", lhs, lhs, _cdg([], []), morph, morph)
    graph = _cdg([_node("a", primitive="foo")], [])
    assert rw._find_match(rule, graph) is None


def test_find_match_wildcard_with_edge():
    """Wildcard LHS node matches any predecessor of an anchored node."""
    rw = GraphRewriter()
    lhs = _cdg(
        [_node("src", concept=ConceptType.CUSTOM), _node("filt", primitive="F")],
        [_edge("src", "filt")],
    )
    morph = Morphism(node_map={}, edge_map={})
    rule = RewriteRule("r", lhs, lhs, _cdg([], []), morph, morph)

    graph = _cdg(
        [_node("input_node", concept=ConceptType.ANALYSIS), _node("filter_node", primitive="F")],
        [_edge("input_node", "filter_node")],
    )
    m = rw._find_match(rule, graph)
    assert m is not None
    assert m.node_map["src"] == "input_node"
    assert m.node_map["filt"] == "filter_node"


# ---------------------------------------------------------------------------
# Gluing condition
# ---------------------------------------------------------------------------


def test_gluing_pure_insertion():
    """Pure insertion (L\\K = empty): gluing always holds."""
    rw = GraphRewriter()
    src = _node("s")
    tgt = _node("t")
    lhs = _cdg([src, tgt], [_edge("s", "t")])
    k = _cdg([src, tgt], [])
    l_morph = Morphism(node_map={"s": "s", "t": "t"}, edge_map={})

    graph = _cdg(
        [_node("gs"), _node("gt"), _node("other")],
        [_edge("gs", "gt"), _edge("other", "gs")],
    )
    match = Morphism(node_map={"s": "gs", "t": "gt"}, edge_map={"s->t": "gs->gt"})

    rule = RewriteRule("r", lhs, lhs, k, l_morph, l_morph)
    assert rw._check_gluing_condition(rule, match, graph) is True


def test_gluing_violation():
    """Deleting a node with edges to context should fail the gluing check."""
    rw = GraphRewriter()
    # LHS has node "del" that will be in L\\K
    lhs = _cdg([_node("del", primitive="X")], [])
    k = _cdg([], [])
    l_morph = Morphism(node_map={}, edge_map={})

    graph = _cdg(
        [_node("gd", primitive="X"), _node("ctx")],
        [_edge("gd", "ctx")],  # edge to context node
    )
    match = Morphism(node_map={"del": "gd"}, edge_map={})
    rule = RewriteRule("r", lhs, lhs, k, l_morph, l_morph)
    assert rw._check_gluing_condition(rule, match, graph) is False


# ---------------------------------------------------------------------------
# Full DPO: interposition rule
# ---------------------------------------------------------------------------


def _make_interposition_rule() -> RewriteRule:
    """Build a rule that interposes node N between A and B."""
    a = _node("a", primitive="A")
    b = _node("b", primitive="B")
    n = _node("n", primitive="N")

    lhs = _cdg([a, b], [_edge("a", "b")])
    k = _cdg([a, b], [])
    rhs = _cdg([a, n, b], [_edge("a", "n"), _edge("n", "b")])

    l_morph = Morphism(node_map={"a": "a", "b": "b"}, edge_map={})
    r_morph = Morphism(node_map={"a": "a", "b": "b"}, edge_map={})

    return RewriteRule("interpose_N", lhs, rhs, k, l_morph, r_morph)


def test_apply_interposition_basic():
    """Interposing N between A and B in a simple A→B graph."""
    rw = GraphRewriter()
    rule = _make_interposition_rule()
    graph = _cdg(
        [_node("ga", primitive="A"), _node("gb", primitive="B")],
        [_edge("ga", "gb")],
    )

    result = rw.apply_rule(rule, graph)
    assert not result.is_failure

    g_prime = result.unwrap()
    assert len(g_prime.nodes) == 3  # ga, gb, new node
    assert len(g_prime.edges) == 2  # ga→new, new→gb

    # The original ga→gb edge should be gone.
    node_ids = {n.node_id for n in g_prime.nodes}
    assert "ga" in node_ids
    assert "gb" in node_ids
    new_ids = node_ids - {"ga", "gb"}
    assert len(new_ids) == 1
    new_id = new_ids.pop()

    edge_pairs = {(e.source_id, e.target_id) for e in g_prime.edges}
    assert ("ga", new_id) in edge_pairs
    assert (new_id, "gb") in edge_pairs
    assert ("ga", "gb") not in edge_pairs


def test_apply_interposition_preserves_context():
    """Context nodes and edges outside the matched region are preserved."""
    rw = GraphRewriter()
    rule = _make_interposition_rule()
    graph = _cdg(
        [
            _node("pre"),
            _node("ga", primitive="A"),
            _node("gb", primitive="B"),
            _node("post"),
        ],
        [
            _edge("pre", "ga"),
            _edge("ga", "gb"),
            _edge("gb", "post"),
        ],
    )

    result = rw.apply_rule(rule, graph)
    assert not result.is_failure

    g_prime = result.unwrap()
    assert len(g_prime.nodes) == 5  # pre, ga, new, gb, post
    edge_pairs = {(e.source_id, e.target_id) for e in g_prime.edges}
    assert ("pre", "ga") in edge_pairs
    assert ("gb", "post") in edge_pairs
    assert ("ga", "gb") not in edge_pairs


def test_apply_rule_no_match():
    """Rule that doesn't match returns a failure, not an exception."""
    rw = GraphRewriter()
    rule = _make_interposition_rule()
    graph = _cdg([_node("x", primitive="X")], [])

    result = rw.apply_rule(rule, graph)
    assert result.is_failure
    assert "no match" in result.error.lower()


def test_apply_interposition_new_node_has_correct_primitive():
    """The interposed node carries the primitive from the RHS template."""
    rw = GraphRewriter()
    rule = _make_interposition_rule()
    graph = _cdg(
        [_node("ga", primitive="A"), _node("gb", primitive="B")],
        [_edge("ga", "gb")],
    )
    g_prime = rw.apply_rule(rule, graph).unwrap()
    new_node = [n for n in g_prime.nodes if n.node_id not in {"ga", "gb"}][0]
    assert new_node.matched_primitive == "N"
