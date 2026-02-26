"""
Tests for bites.utils.output_utils module.

Covers pure functions that don't require real file I/O or database access:
- flatten_dict(): nested dict flattening
- stop_the_timer(): elapsed time formatting
- config_to_serializable_dict(): config serialization
- _serialize_for_csv(): CSV value serialization
- get_fixed_columns(): column list for CSV export
"""

import datetime
import json
import time
from pathlib import Path

import numpy as np
import pytest

from bites.core.output_manager import flatten_dict, stop_the_timer
from bites.core.trial_export import _serialize_for_csv, get_fixed_columns
from bites.config.config_io import config_to_serializable_dict


# ---------------------------------------------------------------------------
# flatten_dict
# ---------------------------------------------------------------------------

class TestFlattenDict:
    """Tests for flatten_dict()."""

    def test_flat_dict_unchanged(self):
        """Already-flat dict passes through with same keys."""
        d = {"a": 1, "b": 2}
        result = flatten_dict(d)
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        """Nested dict produces dot-separated keys."""
        d = {"a": {"b": 1, "c": 2}}
        result = flatten_dict(d)
        assert result == {"a.b": 1, "a.c": 2}

    def test_deeply_nested(self):
        """Multi-level nesting produces correct keys."""
        d = {"a": {"b": {"c": 42}}}
        result = flatten_dict(d)
        assert result == {"a.b.c": 42}

    def test_mixed_depth(self):
        """Mix of flat and nested keys."""
        d = {"x": 1, "y": {"z": 2}}
        result = flatten_dict(d)
        assert result == {"x": 1, "y.z": 2}

    def test_custom_separator(self):
        """Custom separator is used between levels."""
        d = {"a": {"b": 1}}
        result = flatten_dict(d, sep="/")
        assert result == {"a/b": 1}

    def test_max_depth_protection(self):
        """Recursion depth limit prevents stack overflow."""
        # Build a deeply nested dict
        d = {"val": "leaf"}
        for _ in range(25):
            d = {"nested": d}
        result = flatten_dict(d, max_depth=20)
        # Should not raise, and should have a key
        assert len(result) >= 1

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        assert flatten_dict({}) == {}

    def test_parent_key_prefix(self):
        """Parent key prefix is prepended."""
        d = {"b": 1}
        result = flatten_dict(d, parent_key="a")
        assert result == {"a.b": 1}


# ---------------------------------------------------------------------------
# stop_the_timer
# ---------------------------------------------------------------------------

class TestStopTheTimer:
    """Tests for stop_the_timer()."""

    def test_returns_tuple(self):
        """Returns (message_string, elapsed_seconds) tuple."""
        start = time.time() - 5  # 5 seconds ago
        result = stop_the_timer(start)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_seconds_format(self):
        """Short times show 'seconds' in message."""
        start = time.time() - 30
        msg, elapsed = stop_the_timer(start)
        assert "seconds" in msg
        assert 28 < elapsed < 35

    def test_minutes_format(self):
        """Times > 60s show 'minutes' in message."""
        start = time.time() - 120
        msg, elapsed = stop_the_timer(start)
        assert "minutes" in msg
        assert 118 < elapsed < 125

    def test_hours_format(self):
        """Times > 3600s show 'hours' in message."""
        start = time.time() - 7200
        msg, elapsed = stop_the_timer(start)
        assert "hours" in msg
        assert 7198 < elapsed < 7205

    def test_contains_finish_time(self):
        """Message contains 'Finished at' with a time."""
        start = time.time() - 1
        msg, _ = stop_the_timer(start)
        assert "Finished at" in msg

    def test_elapsed_is_float(self):
        """Second element is a float."""
        start = time.time() - 1
        _, elapsed = stop_the_timer(start)
        assert isinstance(elapsed, float)


# ---------------------------------------------------------------------------
# config_to_serializable_dict
# ---------------------------------------------------------------------------

class TestConfigToSerializableDict:
    """Tests for config_to_serializable_dict()."""

    def test_path_converted_to_string(self):
        """Path objects become strings."""
        result = config_to_serializable_dict({"path": Path("/a/b")})
        assert result["path"] == str(Path("/a/b"))

    def test_datetime_converted(self):
        """datetime objects become ISO format strings."""
        dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        result = config_to_serializable_dict({"when": dt})
        assert result["when"] == "2024-01-15T10:30:00"

    def test_numpy_array_converted(self):
        """numpy arrays become lists."""
        arr = np.array([1.0, 2.0, 3.0])
        result = config_to_serializable_dict({"data": arr})
        assert result["data"] == [1.0, 2.0, 3.0]

    def test_numpy_scalar_converted(self):
        """numpy scalars become Python scalars."""
        val = np.float64(3.14)
        result = config_to_serializable_dict({"val": val})
        assert result["val"] == pytest.approx(3.14)
        assert isinstance(result["val"], float)

    def test_plain_value_passthrough(self):
        """Simple values pass through unchanged."""
        result = config_to_serializable_dict({"x": 42, "y": "hello"})
        assert result == {"x": 42, "y": "hello"}

    def test_nested_dict(self):
        """Nested dicts are recursively processed."""
        data = {"outer": {"inner": Path("/x")}}
        result = config_to_serializable_dict(data)
        assert result["outer"]["inner"] == str(Path("/x"))

    def test_list_of_paths(self):
        """Lists containing Path objects are processed."""
        data = {"paths": [Path("/a"), Path("/b")]}
        result = config_to_serializable_dict(data)
        assert result["paths"] == [str(Path("/a")), str(Path("/b"))]

    def test_timedelta_converted(self):
        """timedelta becomes total_seconds float."""
        td = datetime.timedelta(hours=1, minutes=30)
        result = config_to_serializable_dict({"dur": td})
        assert result["dur"] == 5400.0

    def test_bytes_converted(self):
        """bytes objects become UTF-8 strings."""
        result = config_to_serializable_dict({"data": b"hello"})
        assert result["data"] == "hello"


# ---------------------------------------------------------------------------
# _serialize_for_csv
# ---------------------------------------------------------------------------

class TestSerializeForCsv:
    """Tests for _serialize_for_csv()."""

    def test_dict_to_json(self):
        """Dict is JSON-encoded."""
        result = _serialize_for_csv({"a": 1})
        assert result == json.dumps({"a": 1})

    def test_list_to_json(self):
        """List is JSON-encoded."""
        result = _serialize_for_csv([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_datetime_to_iso(self):
        """Datetime becomes ISO format string."""
        dt = datetime.datetime(2024, 6, 15, 12, 0, 0)
        result = _serialize_for_csv(dt)
        assert result == "2024-06-15T12:00:00"

    def test_numpy_array_to_json(self):
        """numpy array becomes JSON list."""
        arr = np.array([1.0, 2.0])
        result = _serialize_for_csv(arr)
        assert json.loads(result) == [1.0, 2.0]

    def test_plain_value_passthrough(self):
        """Simple values pass through unchanged."""
        assert _serialize_for_csv(42) == 42
        assert _serialize_for_csv("hello") == "hello"
        assert _serialize_for_csv(3.14) == 3.14


# ---------------------------------------------------------------------------
# get_fixed_columns
# ---------------------------------------------------------------------------

class TestGetFixedColumns:
    """Tests for get_fixed_columns()."""

    def test_returns_list(self):
        """Returns a list."""
        result = get_fixed_columns()
        assert isinstance(result, list)

    def test_contains_trial_number(self):
        """Contains 'trial_number' column."""
        assert "trial_number" in get_fixed_columns()

    def test_contains_loss(self):
        """Contains 'loss' column."""
        assert "loss" in get_fixed_columns()

    def test_contains_parameter_names(self):
        """Contains known parameter names."""
        cols = get_fixed_columns()
        for param in ["Egr", "Mgr", "Act", "uMgr"]:
            assert param in cols

    def test_contains_state(self):
        """Contains 'state' column."""
        assert "state" in get_fixed_columns()

    def test_no_duplicates(self):
        """No duplicate column names."""
        cols = get_fixed_columns()
        assert len(cols) == len(set(cols))
