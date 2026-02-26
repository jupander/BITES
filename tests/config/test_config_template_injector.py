"""
Tests for bites.config.config_template_injector module.

Covers pure functions only (no file I/O):
- flatten_data(): nested dict to OrderedDict with dot-separated keys
- format_yaml_value(): value formatting for YAML output
- merge_key_orders(): template + data key order merging
"""

from collections import OrderedDict

import pytest

from bites.config.config_template_injector import (
    flatten_data,
    format_yaml_value,
    merge_key_orders,
)


# ---------------------------------------------------------------------------
# flatten_data
# ---------------------------------------------------------------------------

class TestFlattenData:
    """Tests for flatten_data()."""

    def test_flat_dict(self):
        """Flat dict returns correctly with leaf values."""
        result = flatten_data({"a": 1, "b": "hello"})
        assert result["a"] == 1
        assert result["b"] == "hello"

    def test_nested_dict_section_headers(self):
        """Nested dict produces None marker for section header."""
        result = flatten_data({"parent": {"child": 42}})
        assert result["parent"] is None  # section header
        assert result["parent.child"] == 42

    def test_deep_nesting(self):
        """Three levels of nesting produce correct dot-separated keys."""
        result = flatten_data({"a": {"b": {"c": "leaf"}}})
        assert result["a"] is None
        assert result["a.b"] is None
        assert result["a.b.c"] == "leaf"

    def test_returns_ordered_dict(self):
        """Returns an OrderedDict."""
        result = flatten_data({"x": 1})
        assert isinstance(result, OrderedDict)

    def test_preserves_insertion_order(self):
        """Keys appear in insertion order."""
        result = flatten_data({"z": 1, "a": 2, "m": 3})
        keys = list(result.keys())
        assert keys == ["z", "a", "m"]

    def test_empty_dict(self):
        """Empty dict returns empty OrderedDict."""
        result = flatten_data({})
        assert len(result) == 0

    def test_mixed_values(self):
        """Mix of nested and flat values."""
        result = flatten_data({"x": 1, "y": {"z": 2}, "w": 3})
        assert result["x"] == 1
        assert result["y"] is None
        assert result["y.z"] == 2
        assert result["w"] == 3

    def test_list_values_are_leaves(self):
        """Lists are treated as leaf values, not recursed into."""
        result = flatten_data({"items": [1, 2, 3]})
        assert result["items"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# format_yaml_value
# ---------------------------------------------------------------------------

class TestFormatYamlValue:
    """Tests for format_yaml_value()."""

    def test_none_returns_empty(self):
        """None produces empty string."""
        assert format_yaml_value(None) == ""

    def test_bool_true(self):
        """True becomes 'true'."""
        assert format_yaml_value(True) == "true"

    def test_bool_false(self):
        """False becomes 'false'."""
        assert format_yaml_value(False) == "false"

    def test_empty_list(self):
        """Empty list produces '[]'."""
        assert format_yaml_value([]) == "[]"

    def test_list_block_style(self):
        """Non-empty list produces block-style YAML with '- ' prefix."""
        result = format_yaml_value(["a", "b"])
        assert "\n" in result
        assert "- a" in result
        assert "- b" in result

    def test_integer(self):
        """Integer returns string representation."""
        assert format_yaml_value(42) == "42"

    def test_float(self):
        """Float returns string representation."""
        assert format_yaml_value(3.14) == "3.14"

    def test_plain_string(self):
        """Simple strings are not quoted."""
        assert format_yaml_value("hello") == "hello"

    def test_string_with_colon_quoted(self):
        """Strings with ':' are quoted."""
        result = format_yaml_value("key: value")
        assert result.startswith('"')
        assert result.endswith('"')

    def test_string_with_hash_quoted(self):
        """Strings with '#' are quoted."""
        result = format_yaml_value("a # comment")
        assert result.startswith('"')

    def test_numeric_string_quoted(self):
        """Strings that look like numbers are quoted."""
        result = format_yaml_value("3.14")
        assert result == '"3.14"'

    def test_bool_string_quoted(self):
        """Strings that look like YAML booleans are quoted."""
        for val in ["true", "false", "yes", "no", "null"]:
            result = format_yaml_value(val)
            assert result.startswith('"'), f"'{val}' should be quoted"

    def test_empty_string_quoted(self):
        """Empty string is quoted."""
        result = format_yaml_value("")
        assert result == '""'

    def test_string_with_brackets_quoted(self):
        """Strings with brackets are quoted."""
        result = format_yaml_value("[1, 2]")
        assert result.startswith('"')


# ---------------------------------------------------------------------------
# merge_key_orders
# ---------------------------------------------------------------------------

class TestMergeKeyOrders:
    """Tests for merge_key_orders()."""

    def test_matching_orders(self):
        """Identical orders produce same result."""
        order = ["a", "b", "c"]
        result = merge_key_orders(order, order)
        assert result == ["a", "b", "c"]

    def test_data_only_keys_appended(self):
        """Keys only in data are added."""
        template = ["a", "b"]
        data = ["a", "b", "c"]
        result = merge_key_orders(template, data)
        assert "c" in result
        assert result.index("a") < result.index("b")

    def test_template_only_keys_excluded(self):
        """Keys only in template are not in result (they have no data)."""
        template = ["a", "b", "c"]
        data = ["a", "b"]
        result = merge_key_orders(template, data)
        assert "c" not in result

    def test_child_keys_follow_parent(self):
        """Child keys in data are inserted after their parent."""
        template = ["parent", "other"]
        data = ["parent", "parent.child", "other"]
        result = merge_key_orders(template, data)
        parent_idx = result.index("parent")
        child_idx = result.index("parent.child")
        other_idx = result.index("other")
        assert parent_idx < child_idx
        assert child_idx < other_idx

    def test_empty_template(self):
        """Empty template preserves all data keys."""
        data = ["a", "b", "c"]
        result = merge_key_orders([], data)
        assert set(result) == {"a", "b", "c"}

    def test_empty_data(self):
        """Empty data produces empty result."""
        result = merge_key_orders(["a", "b"], [])
        assert result == []

    def test_both_empty(self):
        """Both empty produces empty result."""
        assert merge_key_orders([], []) == []

    def test_preserves_template_order(self):
        """Data keys that exist in template follow template order."""
        template = ["c", "a", "b"]
        data = ["a", "b", "c"]
        result = merge_key_orders(template, data)
        assert result.index("c") < result.index("a")
        assert result.index("a") < result.index("b")

    def test_no_duplicates(self):
        """Result contains no duplicate keys."""
        template = ["a", "b", "c"]
        data = ["a", "b", "c", "d"]
        result = merge_key_orders(template, data)
        assert len(result) == len(set(result))
