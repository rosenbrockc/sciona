"""Tests for ageom.json_utils.extract_json."""

from __future__ import annotations

import json

import pytest

from ageom.json_utils import extract_json


class TestExtractJsonFastPath:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_plain_array(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_whitespace_padded(self):
        assert extract_json("  \n{\"x\": true}\n  ") == {"x": True}


class TestExtractJsonFences:
    def test_json_fence(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\n'
        assert extract_json(text) == {"key": "value"}

    def test_bare_fence(self):
        text = 'Sure!\n```\n[1, 2]\n```\nDone.'
        assert extract_json(text) == [1, 2]

    def test_fence_with_preamble_and_postamble(self):
        text = (
            "I'll generate the JSON now.\n\n"
            "```json\n"
            '{"macro_atoms": [], "edges": []}\n'
            "```\n\n"
            "Let me know if you need changes."
        )
        result = extract_json(text)
        assert result == {"macro_atoms": [], "edges": []}


class TestExtractJsonBalancedBraces:
    def test_object_in_prose(self):
        text = 'The answer is {"a": 1, "b": [2, 3]} ok?'
        assert extract_json(text) == {"a": 1, "b": [2, 3]}

    def test_array_in_prose(self):
        text = "Results: [1, 2, 3] done"
        assert extract_json(text) == [1, 2, 3]

    def test_nested_braces(self):
        obj = {"outer": {"inner": [1, {"deep": True}]}}
        text = f"Here: {json.dumps(obj)} end"
        assert extract_json(text) == obj

    def test_braces_inside_strings(self):
        obj = {"msg": "use {x} and [y]"}
        text = f"Output: {json.dumps(obj)} bye"
        assert extract_json(text) == obj

    def test_escaped_quotes_in_strings(self):
        obj = {"msg": 'say "hello"'}
        text = f"Result: {json.dumps(obj)}"
        assert extract_json(text) == obj


class TestExtractJsonFailure:
    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json("no json here at all")

    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json("")

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json("{not: valid: json}")
