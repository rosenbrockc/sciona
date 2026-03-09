import pytest
from ageom.architect.handoff import CDGExport
from ageom.architect.models import AlgorithmicNode, DependencyEdge, ConceptType, NodeStatus
from ageom.architect.graph_rewriter import GraphState, GraphRewriter, RewriteRule, Morphism, PriorityStrategy

def test_graph_state_monad_success():
    """Verify that GraphState successfully binds transformations."""
    initial_graph = CDGExport(nodes=[], edges=[], metadata={})
    state = GraphState.success(initial_graph)
    
    def transform(g):
        # Return a new graph with one node
        new_node = AlgorithmicNode(
            node_id="n1",
            name="Test Node",
            description="Test",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,
            depth=0,
            inputs=[],
            outputs=[]
        )
        return GraphState.success(CDGExport(nodes=[new_node], edges=[], metadata={}))
    
    result = state.bind(transform)
    assert not result.is_failure
    assert len(result.unwrap().nodes) == 1
    assert result.unwrap().nodes[0].node_id == "n1"

def test_graph_state_monad_failure_chaining():
    """Verify that failures short-circuit the monadic chain."""
    initial_graph = CDGExport(nodes=[], edges=[], metadata={})
    state = GraphState.success(initial_graph)
    
    def fail_transform(g):
        return GraphState.failure("Something went wrong")
    
    def second_transform(g):
        pytest.fail("Second transform should not be called")
        
    result = state.bind(fail_transform).bind(second_transform)
    assert result.is_failure
    with pytest.raises(RuntimeError, match="Something went wrong"):
        result.unwrap()

def test_priority_strategy():
    """Verify that rules are sorted by priority."""
    lhs = CDGExport(nodes=[], edges=[], metadata={})
    rule_low = RewriteRule(
        name="low", lhs=lhs, rhs=lhs, interface=lhs,
        l_morphism=Morphism(node_map={}, edge_map={}),
        r_morphism=Morphism(node_map={}, edge_map={}),
        priority=1
    )
    rule_high = RewriteRule(
        name="high", lhs=lhs, rhs=lhs, interface=lhs,
        l_morphism=Morphism(node_map={}, edge_map={}),
        r_morphism=Morphism(node_map={}, edge_map={}),
        priority=10
    )
    
    strategy = PriorityStrategy()
    sorted_rules = strategy.sort_rules([rule_low, rule_high])
    assert sorted_rules[0].name == "high"
    assert sorted_rules[1].name == "low"

def test_graph_rewriter_no_match():
    """Verify that the rewriter returns a failure when no match is found."""
    lhs = CDGExport(nodes=[], edges=[], metadata={})
    rule = RewriteRule(
        name="no-match-rule", lhs=lhs, rhs=lhs, interface=lhs,
        l_morphism=Morphism(node_map={}, edge_map={}),
        r_morphism=Morphism(node_map={}, edge_map={}),
    )
    
    graph = CDGExport(nodes=[], edges=[], metadata={})
    rewriter = GraphRewriter()
    
    # By default, _find_match returns None in the placeholder implementation
    result = rewriter.apply_rule(rule, graph)
    assert result.is_failure
    assert "found no match" in str(result._error)
