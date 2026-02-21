"""Tests for the unified index (Issue 3)."""

from ageom.architect.embedder import SkillIndex


def test_skill_index_has_semantic_index_methods():
    """SkillIndex should have the methods required by SemanticIndex protocol."""
    idx = SkillIndex()
    assert hasattr(idx, "search_by_embedding")
    assert hasattr(idx, "search_by_type")
    assert hasattr(idx, "get_declaration")


def test_skill_index_search_by_embedding_empty():
    """search_by_embedding on empty index returns empty list."""
    idx = SkillIndex()
    result = idx.search_by_embedding("test query", k=5)
    assert result == []


def test_skill_index_search_by_type_empty():
    """search_by_type on empty index returns empty list."""
    idx = SkillIndex()
    result = idx.search_by_type("nat -> nat", k=5)
    assert result == []


def test_skill_index_get_declaration_empty():
    """get_declaration on empty index returns None."""
    idx = SkillIndex()
    assert idx.get_declaration("foo") is None


def test_unified_index_basic():
    """UnifiedIndex basic operations."""
    from ageom.indexer.unified import UnifiedIndex

    idx = UnifiedIndex()
    assert idx.size == 0
    assert idx.get_declaration("foo") is None
    assert idx.search_by_embedding("test") == []
    assert idx.search_by_type("nat -> nat") == []
