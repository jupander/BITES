"""
Tests for bites.core.run_tooth_model module.

Covers pure functions that don't require subprocess execution:
- get_parameter_names_array(): parameter name list
- construct_model_parameters_string(): parameter serialization
- construct_run_identifier_string(): unique identifier generation
- construct_command_string(): command-line argument construction
- _get_last_iteration_from_output(): file-based iteration scanning
- _get_newest_mesh_file(): newest mesh file detection
"""

import re
from pathlib import Path

import pytest

from bites.core.run_tooth_model import (
    get_parameter_names_array,
    construct_model_parameters_string,
    construct_run_identifier_string,
    construct_command_string,
    _get_last_iteration_from_output,
    _get_newest_mesh_file,
)
from bites.constants import RUN_IDENTIFIER_PREFIX


# ---------------------------------------------------------------------------
# get_parameter_names_array
# ---------------------------------------------------------------------------

class TestGetParameterNamesArray:
    """Tests for get_parameter_names_array()."""

    def test_returns_list(self):
        """Returns a list."""
        result = get_parameter_names_array()
        assert isinstance(result, list)

    def test_length_is_30(self):
        """Returns exactly 30 entries (including '-' placeholders)."""
        result = get_parameter_names_array()
        assert len(result) == 30

    def test_contains_known_params(self):
        """Contains expected named parameters."""
        result = get_parameter_names_array()
        for param in ["Egr", "Mgr", "Act", "Adh", "Inh", "uMgr"]:
            assert param in result

    def test_contains_dash_placeholders(self):
        """Contains '-' placeholder entries."""
        result = get_parameter_names_array()
        assert result.count("-") == 3

    def test_first_param_is_egr(self):
        """First parameter is 'Egr'."""
        result = get_parameter_names_array()
        assert result[0] == "Egr"

    def test_last_param_is_umgr(self):
        """Last parameter is 'uMgr'."""
        result = get_parameter_names_array()
        assert result[-1] == "uMgr"

    def test_returns_new_list_each_call(self):
        """Each call returns a distinct list object (no shared state)."""
        a = get_parameter_names_array()
        b = get_parameter_names_array()
        assert a == b
        assert a is not b


# ---------------------------------------------------------------------------
# construct_model_parameters_string
# ---------------------------------------------------------------------------

class TestConstructModelParametersString:
    """Tests for construct_model_parameters_string(trial_parameters_dict)."""

    def test_returns_semicolon_separated_string(self, sample_trial_params):
        """Returns a string with semicolon separators."""
        result = construct_model_parameters_string(sample_trial_params)
        assert isinstance(result, str)
        assert ";" in result

    def test_has_30_values(self, sample_trial_params):
        """Output has exactly 30 semicolon-separated values."""
        result = construct_model_parameters_string(sample_trial_params)
        values = result.split(";")
        assert len(values) == 30

    def test_dash_auto_added(self):
        """Adds '-': 0.0 to dict if missing."""
        params = {name: 1.0 for name in get_parameter_names_array() if name != "-"}
        # Should not raise -- '-' is auto-added
        result = construct_model_parameters_string(params)
        values = result.split(";")
        assert len(values) == 30

    def test_specific_value_appears(self, sample_trial_params):
        """Known parameter value appears in the output string."""
        result = construct_model_parameters_string(sample_trial_params)
        # Egr=0.025 is the first parameter
        assert result.startswith("0.025;")

    def test_raises_on_incomplete_dict(self):
        """Raises ValueError if fewer than 30 params available."""
        with pytest.raises(ValueError, match="Expected 30 parameters"):
            construct_model_parameters_string({"Egr": 0.1})

    def test_preserves_param_order(self, sample_trial_params):
        """Values appear in the order defined by get_parameter_names_array()."""
        result = construct_model_parameters_string(sample_trial_params)
        values = result.split(";")
        names = get_parameter_names_array()
        for i, name in enumerate(names):
            expected = str(sample_trial_params[name])
            assert values[i] == expected, \
                f"Position {i} ({name}): expected {expected}, got {values[i]}"

    def test_does_not_mutate_dict_with_dash(self, sample_trial_params):
        """Dict that already has '-' is not modified unexpectedly."""
        original_dash_val = sample_trial_params["-"]
        construct_model_parameters_string(sample_trial_params)
        assert sample_trial_params["-"] == original_dash_val


# ---------------------------------------------------------------------------
# construct_run_identifier_string
# ---------------------------------------------------------------------------

class TestConstructRunIdentifierString:
    """Tests for construct_run_identifier_string()."""

    def test_starts_with_prefix(self):
        """Run identifier starts with RUN_IDENTIFIER_PREFIX ('run')."""
        result = construct_run_identifier_string()
        assert result.startswith(RUN_IDENTIFIER_PREFIX)

    def test_contains_timestamp_pattern(self):
        """Contains YYMMDD-HHMMSS pattern after prefix."""
        result = construct_run_identifier_string()
        # After 'run', expect 6 digits, hyphen, 9 digits (HHMMSS + milliseconds)
        after_prefix = result[len(RUN_IDENTIFIER_PREFIX):]
        assert re.match(r'\d{6}-\d{9}', after_prefix), \
            f"Expected timestamp pattern, got: {after_prefix}"

    def test_ends_with_letters(self):
        """Ends with 3 lowercase random letters."""
        result = construct_run_identifier_string()
        assert result[-3:].isalpha()
        assert result[-3:].islower()

    def test_uniqueness(self):
        """Two consecutive calls produce different identifiers."""
        a = construct_run_identifier_string()
        b = construct_run_identifier_string()
        assert a != b

    def test_reasonable_length(self):
        """Identifier has a reasonable length (prefix + timestamp + letters)."""
        result = construct_run_identifier_string()
        # "run" (3) + "YYMMDD-HHMMSSmmm" (16) + "abc" (3) = 22
        assert 20 <= len(result) <= 25


# ---------------------------------------------------------------------------
# construct_command_string
# ---------------------------------------------------------------------------

class TestConstructCommandString:
    """Tests for construct_command_string()."""

    def test_returns_tuple(self, sample_trial_params, tmp_path):
        """Returns (command_list, run_identifier) tuple."""
        exe = tmp_path / "tooth_model.exe"
        result = construct_command_string(exe, sample_trial_params, 1000, tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_command_is_list(self, sample_trial_params, tmp_path):
        """First element of tuple is a list."""
        exe = tmp_path / "tooth_model.exe"
        command, _ = construct_command_string(exe, sample_trial_params, 1000, tmp_path)
        assert isinstance(command, list)

    def test_command_has_7_elements(self, sample_trial_params, tmp_path):
        """Command list has 7 elements."""
        exe = tmp_path / "tooth_model.exe"
        command, _ = construct_command_string(exe, sample_trial_params, 1000, tmp_path)
        assert len(command) == 7

    def test_first_element_is_executable(self, sample_trial_params, tmp_path):
        """First command element is the executable path string."""
        exe = tmp_path / "tooth_model.exe"
        command, _ = construct_command_string(exe, sample_trial_params, 1000, tmp_path)
        assert command[0] == str(exe)

    def test_run_identifier_returned(self, sample_trial_params, tmp_path):
        """Second element is a valid run identifier string."""
        exe = tmp_path / "tooth_model.exe"
        _, run_id = construct_command_string(exe, sample_trial_params, 1000, tmp_path)
        assert run_id.startswith(RUN_IDENTIFIER_PREFIX)

    def test_iter_max_used_for_n_intervals(self, sample_trial_params, tmp_path):
        """iter_max from dict is used for n_intervals calculation."""
        exe = tmp_path / "tooth_model.exe"
        sample_trial_params["iter_max"] = 10000
        command, _ = construct_command_string(exe, sample_trial_params, 1000, tmp_path)
        # n_intervals = 10000 // 1000 = 10
        assert command[4] == "10"

    def test_iter_fallback(self, tmp_path):
        """Falls back to 'iter' key when 'iter_max' not present."""
        params = {name: 1.0 for name in get_parameter_names_array() if name != "-"}
        params["-"] = 0.0
        params["iter"] = 5000.0  # no iter_max
        exe = tmp_path / "tooth_model.exe"
        command, _ = construct_command_string(exe, params, 1000, tmp_path)
        # n_intervals = 5000 // 1000 = 5
        assert command[4] == "5"

    def test_output_path_is_last_element(self, sample_trial_params, tmp_path):
        """Last command element is the output path."""
        exe = tmp_path / "tooth_model.exe"
        command, _ = construct_command_string(exe, sample_trial_params, 1000, tmp_path)
        assert command[6] == str(tmp_path)

    def test_all_elements_are_strings(self, sample_trial_params, tmp_path):
        """All command elements are strings."""
        exe = tmp_path / "tooth_model.exe"
        command, _ = construct_command_string(exe, sample_trial_params, 1000, tmp_path)
        for i, elem in enumerate(command):
            assert isinstance(elem, str), f"Element {i} is {type(elem)}, not str"


# ---------------------------------------------------------------------------
# _get_last_iteration_from_output
# ---------------------------------------------------------------------------

class TestGetLastIterationFromOutput:
    """Tests for _get_last_iteration_from_output()."""

    def test_finds_highest_iteration(self, tmp_path):
        """Returns the highest iteration number from matching files."""
        run_id = "run240101-120000abc"
        # Create mesh files with different iteration numbers
        for iter_num in [1000, 3000, 5000]:
            (tmp_path / f"{iter_num}__{run_id}_.off").touch()

        result = _get_last_iteration_from_output(tmp_path, run_id)
        assert result == 5000

    def test_returns_none_when_no_files(self, tmp_path):
        """Returns None when directory has no matching files."""
        result = _get_last_iteration_from_output(tmp_path, "run999999-000000zzz")
        assert result is None

    def test_ignores_non_matching_files(self, tmp_path):
        """Ignores files that don't match the run identifier pattern."""
        run_id = "run240101-120000abc"
        # Create a file for a different run
        (tmp_path / "5000__run999999-000000zzz_.off").touch()
        # And a non-OFF file
        (tmp_path / f"5000__{run_id}_.txt").touch()

        result = _get_last_iteration_from_output(tmp_path, run_id)
        assert result is None

    def test_single_file(self, tmp_path):
        """Works correctly with a single matching file."""
        run_id = "run240101-120000abc"
        (tmp_path / f"7000__{run_id}_.off").touch()

        result = _get_last_iteration_from_output(tmp_path, run_id)
        assert result == 7000


# ---------------------------------------------------------------------------
# _get_newest_mesh_file
# ---------------------------------------------------------------------------

class TestGetNewestMeshFile:
    """Tests for _get_newest_mesh_file()."""

    def test_returns_none_when_empty(self, tmp_path):
        """Returns None when no matching files exist."""
        result = _get_newest_mesh_file(tmp_path, "run999999-000000zzz")
        assert result is None

    def test_finds_file(self, tmp_path):
        """Returns (path, iteration) tuple for a matching file."""
        run_id = "run240101-120000abc"
        mesh_file = tmp_path / f"3000__{run_id}_.off"
        mesh_file.write_text("dummy mesh")

        result = _get_newest_mesh_file(tmp_path, run_id)
        assert result is not None
        path, iter_num = result
        assert path == mesh_file
        assert iter_num == 3000

    def test_returns_newest_by_mtime(self, tmp_path):
        """Returns the file with the most recent modification time."""
        import time
        run_id = "run240101-120000abc"

        old_file = tmp_path / f"1000__{run_id}_.off"
        old_file.write_text("old")
        time.sleep(0.05)  # ensure different mtime

        new_file = tmp_path / f"5000__{run_id}_.off"
        new_file.write_text("new")

        result = _get_newest_mesh_file(tmp_path, run_id)
        assert result is not None
        path, iter_num = result
        assert path == new_file
        assert iter_num == 5000
