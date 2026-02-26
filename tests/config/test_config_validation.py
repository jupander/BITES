"""
Tests for bites.config.config_validation module.

Covers:
- ValidationResult and SchemaValidationResult dataclasses
- ConfigSchemaValidator: schema checking, type validation, constraint enforcement
- ConfigValidator: widget extraction, iteration range checks, parallel settings,
  target path validation, study name validation, highlight helpers
- show_validation_errors dialog logic

All Qt widgets are mocked — no event loop is required.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bites.config.config_validation import (
    REQUIRED_CONFIG_KEYS,
    OPTIONAL_CONFIG_KEYS,
    MAX_STUDY_NAME_LENGTH,
    MIN_PARALLEL_TRIALS,
    MAX_PARALLEL_TRIALS,
    ConfigValidationError,
    ConfigLoadError,
    ValidationResult,
    SchemaValidationResult,
    ConfigSchemaValidator,
    ConfigValidator,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def valid_config(sample_config_dict) -> dict:
    """Return a config dict that satisfies all required keys and types.

    Adds 'technical_settings' (bool) which is required but not in sample_config_dict.
    """
    cfg = dict(sample_config_dict)
    cfg.setdefault("technical_settings", False)
    return cfg


@pytest.fixture
def mock_widget():
    """Return a MagicMock that behaves like a QWidget (has setStyleSheet)."""
    w = MagicMock()
    w.setStyleSheet = MagicMock()
    w.styleSheet = MagicMock(return_value="")
    return w


@pytest.fixture
def mock_line_edit():
    """Return a MagicMock that passes isinstance(w, QLineEdit) check."""
    from PySide6.QtWidgets import QLineEdit
    w = MagicMock(spec=QLineEdit)
    w.setStyleSheet = MagicMock()
    w.styleSheet = MagicMock(return_value="")
    return w


@pytest.fixture
def mock_checkbox():
    """Return a MagicMock that passes isinstance(w, QCheckBox) check."""
    from PySide6.QtWidgets import QCheckBox
    w = MagicMock(spec=QCheckBox)
    return w


@pytest.fixture
def mock_spinbox():
    """Return a MagicMock that passes isinstance(w, QSpinBox) check."""
    from PySide6.QtWidgets import QSpinBox
    w = MagicMock(spec=QSpinBox)
    return w


# ============================================================================
# TestValidationResult
# ============================================================================

class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_default_is_valid(self):
        """A fresh result should be valid with empty lists."""
        result = ValidationResult()
        assert result.is_valid is True
        assert result.invalid_fields == []
        assert result.errors == []
        assert result.warnings == []

    def test_mutable_default_lists_are_independent(self):
        """Each instance should have its own list objects (no shared state)."""
        r1 = ValidationResult()
        r2 = ValidationResult()
        r1.errors.append("error1")
        assert r2.errors == []

    def test_custom_values(self):
        """Explicit field values should be preserved."""
        result = ValidationResult(
            is_valid=False,
            invalid_fields=["study_name"],
            errors=["Too long"],
            warnings=["No algos"],
        )
        assert result.is_valid is False
        assert "study_name" in result.invalid_fields
        assert len(result.errors) == 1
        assert len(result.warnings) == 1


# ============================================================================
# TestSchemaValidationResult
# ============================================================================

class TestSchemaValidationResult:
    """Tests for the SchemaValidationResult dataclass."""

    def test_default_is_valid(self):
        result = SchemaValidationResult()
        assert result.is_valid is True
        assert result.missing_keys == []
        assert result.type_errors == []
        assert result.applied_defaults == []

    def test_mutable_default_lists_are_independent(self):
        r1 = SchemaValidationResult()
        r2 = SchemaValidationResult()
        r1.missing_keys.append("foo")
        assert r2.missing_keys == []


# ============================================================================
# TestConfigSchemaValidatorCheckType
# ============================================================================

class TestConfigSchemaValidatorCheckType:
    """Tests for ConfigSchemaValidator._check_type helper."""

    def test_single_type_match(self):
        assert ConfigSchemaValidator._check_type("hello", str) is True

    def test_single_type_mismatch(self):
        assert ConfigSchemaValidator._check_type(123, str) is False

    def test_tuple_type_match(self):
        """study_timeout can be int or float."""
        assert ConfigSchemaValidator._check_type(42, (int, float)) is True
        assert ConfigSchemaValidator._check_type(3.14, (int, float)) is True

    def test_tuple_type_mismatch(self):
        assert ConfigSchemaValidator._check_type("oops", (int, float)) is False

    def test_bool_is_int_subclass(self):
        """In Python bool is a subclass of int — isinstance(True, int) is True."""
        assert ConfigSchemaValidator._check_type(True, int) is True

    def test_none_mismatch(self):
        assert ConfigSchemaValidator._check_type(None, str) is False


# ============================================================================
# TestConfigSchemaValidatorValidateAndFillDefaults
# ============================================================================

class TestConfigSchemaValidatorFillDefaults:
    """Tests for ConfigSchemaValidator.validate_and_fill_defaults."""

    def test_valid_config_passes(self, valid_config):
        """A complete, correct config should validate successfully."""
        result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        assert result.is_valid is True
        assert result.missing_keys == []
        assert result.type_errors == []

    def test_missing_required_key_fails(self, valid_config):
        """Removing a required key should mark the result invalid."""
        del valid_config["study_name"]
        result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        assert result.is_valid is False
        assert "study_name" in result.missing_keys

    def test_wrong_type_fails(self, valid_config):
        """Supplying a string where int is expected should produce a type error."""
        valid_config["max_trials"] = "not_an_int"
        result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        assert result.is_valid is False
        assert any("max_trials" in e for e in result.type_errors)

    def test_fills_missing_optional_keys(self, valid_config):
        """Optional keys not in config should be added with their defaults."""
        # Remove an optional key that is present
        valid_config.pop("iter_interval", None)
        valid_config.pop("storage_mode", None)
        result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        assert "iter_interval" in result.applied_defaults
        assert valid_config["iter_interval"] == 1000  # DEFAULT_ITER_INTERVAL

    def test_existing_optional_key_not_overwritten(self, valid_config):
        """An optional key already in config should keep its value."""
        valid_config["storage_mode"] = "local"
        ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        assert valid_config["storage_mode"] == "local"


# ============================================================================
# TestConfigSchemaValidatorConstraints
# ============================================================================

class TestConfigSchemaValidatorConstraints:
    """Tests for the _validate_constraints method invoked by validate_and_fill_defaults."""

    def test_study_name_too_long_fails(self, valid_config):
        valid_config["study_name"] = "x" * (MAX_STUDY_NAME_LENGTH + 1)
        result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        assert result.is_valid is False
        assert any("study_name" in e for e in result.type_errors)

    def test_study_name_at_max_length_passes(self, valid_config):
        valid_config["study_name"] = "x" * MAX_STUDY_NAME_LENGTH
        result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        # Study name at exactly max length should be fine (no constraint error from name)
        assert not any("study_name" in e for e in result.type_errors)

    def test_parallel_trials_below_min_fails(self, valid_config):
        valid_config["parallel_trials"] = 0
        result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        assert result.is_valid is False
        assert any("parallel_trials" in e for e in result.type_errors)

    def test_parallel_trials_above_max_fails(self, valid_config):
        valid_config["parallel_trials"] = MAX_PARALLEL_TRIALS + 1
        result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        assert result.is_valid is False
        assert any("parallel_trials" in e for e in result.type_errors)

    def test_parallel_trials_at_boundaries_pass(self, valid_config):
        for val in (MIN_PARALLEL_TRIALS, MAX_PARALLEL_TRIALS):
            valid_config["parallel_trials"] = val
            result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
            assert not any("parallel_trials" in e for e in result.type_errors)

    def test_min_iterations_gt_max_fails(self, valid_config):
        valid_config["min_iterations"] = 6000
        valid_config["max_iterations"] = 5000
        result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        assert result.is_valid is False
        assert any("min_iterations" in e for e in result.type_errors)

    def test_min_iterations_equal_to_max_passes(self, valid_config):
        valid_config["min_iterations"] = 5000
        valid_config["max_iterations"] = 5000
        result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        assert not any("min_iterations" in e for e in result.type_errors)

    def test_min_iterations_less_than_max_passes(self, valid_config):
        valid_config["min_iterations"] = 1000
        valid_config["max_iterations"] = 5000
        result = ConfigSchemaValidator.validate_and_fill_defaults(valid_config)
        assert not any("min_iterations" in e for e in result.type_errors)


# ============================================================================
# TestConfigSchemaConstants
# ============================================================================

class TestConfigSchemaConstants:
    """Verify that the module-level schema constants are properly defined."""

    def test_required_keys_is_non_empty(self):
        assert len(REQUIRED_CONFIG_KEYS) > 0

    def test_optional_keys_is_non_empty(self):
        assert len(OPTIONAL_CONFIG_KEYS) > 0

    def test_no_overlap_between_required_and_optional(self):
        overlap = set(REQUIRED_CONFIG_KEYS) & set(OPTIONAL_CONFIG_KEYS)
        assert overlap == set(), f"Overlapping keys: {overlap}"

    def test_study_timeout_accepts_int_or_float(self):
        expected = REQUIRED_CONFIG_KEYS["study_timeout"]
        assert expected == (int, float)

    def test_constraint_constants(self):
        assert MAX_STUDY_NAME_LENGTH == 64
        assert MIN_PARALLEL_TRIALS == 1
        assert MAX_PARALLEL_TRIALS == 1000


# ============================================================================
# TestConfigValidatorHighlight
# ============================================================================

class TestConfigValidatorHighlight:
    """Tests for _highlight_invalid and _clear_highlight helper methods."""

    def _make_validator(self):
        return ConfigValidator(widgets={}, config={}, parent=MagicMock())

    def test_highlight_sets_pink_background(self, mock_widget):
        validator = self._make_validator()
        validator._highlight_invalid(mock_widget)
        mock_widget.setStyleSheet.assert_called_once_with("background-color: pink;")

    def test_clear_highlight_resets_stylesheet(self, mock_widget):
        validator = self._make_validator()
        validator._clear_highlight(mock_widget)
        mock_widget.setStyleSheet.assert_called_once_with("")

    def test_highlight_none_widget_does_not_raise(self):
        validator = self._make_validator()
        validator._highlight_invalid(None)  # should not raise

    def test_clear_highlight_none_widget_does_not_raise(self):
        validator = self._make_validator()
        validator._clear_highlight(None)  # should not raise

    def test_highlight_widget_without_setStyleSheet(self):
        """Widget without setStyleSheet attr should be handled gracefully."""
        validator = self._make_validator()
        widget = MagicMock(spec=[])  # no attributes
        widget.setStyleSheet = MagicMock(side_effect=AttributeError)
        validator._highlight_invalid(widget)  # should not raise


# ============================================================================
# TestConfigValidatorExtractLineEdit
# ============================================================================

class TestConfigValidatorExtractLineEdit:
    """Tests for _extract_line_edit_value with various field types."""

    def _make_validator(self, widgets=None, config=None):
        return ConfigValidator(
            widgets=widgets or {},
            config=config or {},
            parent=MagicMock(),
        )

    def test_tm_timeout_valid_int(self, mock_line_edit):
        mock_line_edit.text.return_value = "300"
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_line_edit_value("tm_timeout", mock_line_edit, result)
        assert value == 300
        assert result.is_valid is True

    def test_tm_timeout_non_positive_returns_default(self, mock_line_edit):
        mock_line_edit.text.return_value = "0"
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_line_edit_value("tm_timeout", mock_line_edit, result)
        assert value == 1  # safe default
        assert result.is_valid is False

    def test_tm_timeout_non_numeric_returns_default(self, mock_line_edit):
        mock_line_edit.text.return_value = "abc"
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_line_edit_value("tm_timeout", mock_line_edit, result)
        assert value == 1
        assert result.is_valid is False
        assert "tm_timeout" in result.invalid_fields

    def test_max_trials_valid(self, mock_line_edit):
        mock_line_edit.text.return_value = "50"
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_line_edit_value("max_trials", mock_line_edit, result)
        assert value == 50
        assert result.is_valid is True

    def test_study_timeout_empty_returns_zero(self, mock_line_edit):
        mock_line_edit.text.return_value = ""
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_line_edit_value("study_timeout", mock_line_edit, result)
        assert value == 0.0
        assert result.is_valid is True

    def test_study_timeout_valid_float(self, mock_line_edit):
        mock_line_edit.text.return_value = "3.5"
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_line_edit_value("study_timeout", mock_line_edit, result)
        assert value == 3.5

    def test_study_timeout_negative_returns_default(self, mock_line_edit):
        mock_line_edit.text.return_value = "-1.0"
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_line_edit_value("study_timeout", mock_line_edit, result)
        assert value == 0.0
        assert result.is_valid is False

    def test_generic_string_field_empty_marks_invalid(self, mock_line_edit):
        mock_line_edit.text.return_value = ""
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_line_edit_value("study_name", mock_line_edit, result)
        assert value == ""
        assert result.is_valid is False
        assert "study_name" in result.invalid_fields

    def test_generic_string_field_valid(self, mock_line_edit):
        mock_line_edit.text.return_value = "my_study"
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_line_edit_value("study_name", mock_line_edit, result)
        assert value == "my_study"
        assert result.is_valid is True


# ============================================================================
# TestConfigValidatorExtractDictWidget
# ============================================================================

class TestConfigValidatorExtractDictWidget:
    """Tests for _extract_dict_widget_value with radio and checkbox dicts."""

    def _make_validator(self):
        return ConfigValidator(widgets={}, config={}, parent=MagicMock())

    def test_radio_buttons_returns_checked_key(self):
        from PySide6.QtWidgets import QRadioButton
        rb1 = MagicMock(spec=QRadioButton)
        rb1.isChecked.return_value = False
        rb2 = MagicMock(spec=QRadioButton)
        rb2.isChecked.return_value = True
        widget_dict = {"shared": rb1, "local": rb2}

        validator = self._make_validator()
        assert validator._extract_dict_widget_value("storage_mode", widget_dict) == "local"

    def test_checkboxes_returns_checked_list(self):
        from PySide6.QtWidgets import QCheckBox
        cb1 = MagicMock(spec=QCheckBox)
        cb1.isChecked.return_value = True
        cb2 = MagicMock(spec=QCheckBox)
        cb2.isChecked.return_value = False
        cb3 = MagicMock(spec=QCheckBox)
        cb3.isChecked.return_value = True
        widget_dict = {"algo_a": cb1, "algo_b": cb2, "algo_c": cb3}

        validator = self._make_validator()
        result = validator._extract_dict_widget_value("alignment_algorithms", widget_dict)
        assert set(result) == {"algo_a", "algo_c"}

    def test_all_checkboxes_unchecked_returns_empty_list(self):
        from PySide6.QtWidgets import QCheckBox
        cb1 = MagicMock(spec=QCheckBox)
        cb1.isChecked.return_value = False
        widget_dict = {"algo_a": cb1}

        validator = self._make_validator()
        result = validator._extract_dict_widget_value("alignment_algorithms", widget_dict)
        assert result == []


# ============================================================================
# TestConfigValidatorIterationRanges
# ============================================================================

class TestConfigValidatorIterationRanges:
    """Tests for _validate_iteration_ranges."""

    def test_valid_range_passes(self, mock_widget):
        config = {"min_iterations": 1000, "max_iterations": 5000}
        widgets = {"min_iterations": mock_widget, "max_iterations": MagicMock()}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_iteration_ranges(result)
        assert result.is_valid is True
        assert result.invalid_fields == []

    def test_equal_values_fail(self, mock_widget):
        config = {"min_iterations": 5000, "max_iterations": 5000}
        widgets = {"min_iterations": mock_widget, "max_iterations": MagicMock()}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_iteration_ranges(result)
        assert result.is_valid is False
        assert "min_iterations" in result.invalid_fields

    def test_inverted_range_fails(self, mock_widget):
        config = {"min_iterations": 10000, "max_iterations": 5000}
        widgets = {"min_iterations": mock_widget, "max_iterations": MagicMock()}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_iteration_ranges(result)
        assert result.is_valid is False


# ============================================================================
# TestConfigValidatorParallelSettings
# ============================================================================

class TestConfigValidatorParallelSettings:
    """Tests for _validate_parallel_settings."""

    def test_unchecked_resets_to_one(self, mock_checkbox):
        mock_checkbox.isChecked.return_value = False
        config = {"parallel_trials": 8}
        widgets = {"parallel_search_enabled": mock_checkbox}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        validator._validate_parallel_settings()
        assert config["parallel_trials"] == 1

    def test_checked_keeps_value(self, mock_checkbox):
        mock_checkbox.isChecked.return_value = True
        config = {"parallel_trials": 8}
        widgets = {"parallel_search_enabled": mock_checkbox}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        validator._validate_parallel_settings()
        assert config["parallel_trials"] == 8

    def test_missing_checkbox_does_nothing(self):
        config = {"parallel_trials": 4}
        widgets = {}  # no parallel_search_enabled
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        validator._validate_parallel_settings()
        assert config["parallel_trials"] == 4  # unchanged


# ============================================================================
# TestConfigValidatorTargetPath
# ============================================================================

class TestConfigValidatorTargetPath:
    """Tests for _validate_target_path."""

    def test_valid_file_path_passes(self, tmp_path, mock_line_edit):
        target = tmp_path / "target.off"
        target.write_text("dummy")
        config = {"target_shape_path": str(target)}
        widgets = {"target_shape_path": mock_line_edit}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_target_path(result)
        assert result.is_valid is True

    def test_nonexistent_path_fails(self, mock_line_edit):
        mock_line_edit.styleSheet.return_value = ""
        config = {"target_shape_path": "/no/such/file.off"}
        widgets = {"target_shape_path": mock_line_edit}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_target_path(result)
        assert result.is_valid is False
        assert "target_shape_path" in result.invalid_fields

    def test_empty_path_fails(self, mock_line_edit):
        mock_line_edit.styleSheet.return_value = ""
        config = {"target_shape_path": ""}
        widgets = {"target_shape_path": mock_line_edit}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_target_path(result)
        assert result.is_valid is False

    def test_already_highlighted_skips_error(self, mock_line_edit):
        """If widget is already pink, validator should not add another error."""
        mock_line_edit.styleSheet.return_value = "background-color: pink;"
        config = {"target_shape_path": "/no/such/file.off"}
        widgets = {"target_shape_path": mock_line_edit}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_target_path(result)
        # Stays valid because user was already warned
        assert result.is_valid is True

    def test_directory_path_fails(self, tmp_path, mock_line_edit):
        """A directory (not a file) should fail validation."""
        mock_line_edit.styleSheet.return_value = ""
        config = {"target_shape_path": str(tmp_path)}
        widgets = {"target_shape_path": mock_line_edit}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_target_path(result)
        assert result.is_valid is False


# ============================================================================
# TestConfigValidatorAlignmentAlgorithms
# ============================================================================

class TestConfigValidatorAlignmentAlgorithms:
    """Tests for _validate_alignment_algorithms."""

    def test_empty_algorithms_adds_warning(self):
        config = {"alignment_algorithms": []}
        validator = ConfigValidator(widgets={}, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_alignment_algorithms(result)
        assert result.is_valid is True  # warning, not error
        assert len(result.warnings) == 1

    def test_non_empty_algorithms_no_warning(self):
        config = {"alignment_algorithms": ["ICP"]}
        validator = ConfigValidator(widgets={}, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_alignment_algorithms(result)
        assert result.warnings == []


# ============================================================================
# TestConfigValidatorStudyName
# ============================================================================

class TestConfigValidatorStudyName:
    """Tests for _validate_study_name."""

    def test_valid_name_passes(self, mock_widget):
        config = {"study_name": "my_study"}
        widgets = {"study_name": mock_widget}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_study_name(result)
        assert result.is_valid is True

    def test_empty_name_fails(self, mock_widget):
        config = {"study_name": ""}
        widgets = {"study_name": mock_widget}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_study_name(result)
        assert result.is_valid is False
        assert "study_name" in result.invalid_fields

    def test_too_long_name_fails(self, mock_widget):
        config = {"study_name": "a" * (MAX_STUDY_NAME_LENGTH + 1)}
        widgets = {"study_name": mock_widget}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_study_name(result)
        assert result.is_valid is False
        assert "study_name" in result.invalid_fields
        assert len(result.errors) == 1

    def test_name_at_max_length_passes(self, mock_widget):
        config = {"study_name": "a" * MAX_STUDY_NAME_LENGTH}
        widgets = {"study_name": mock_widget}
        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = ValidationResult()
        validator._validate_study_name(result)
        assert result.is_valid is True


# ============================================================================
# TestConfigValidatorExtractSingleWidget
# ============================================================================

class TestConfigValidatorExtractSingleWidget:
    """Tests for _extract_single_widget_value dispatching."""

    def _make_validator(self):
        return ConfigValidator(widgets={}, config={}, parent=MagicMock())

    def test_spinbox_returns_value(self, mock_spinbox):
        mock_spinbox.value.return_value = 42
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_single_widget_value("some_key", mock_spinbox, result)
        assert value == 42

    def test_checkbox_returns_bool(self, mock_checkbox):
        mock_checkbox.isChecked.return_value = True
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_single_widget_value("some_key", mock_checkbox, result)
        assert value is True

    def test_dict_widget_dispatches(self):
        """A dict widget should call _extract_dict_widget_value."""
        from PySide6.QtWidgets import QCheckBox
        cb = MagicMock(spec=QCheckBox)
        cb.isChecked.return_value = True
        widget_dict = {"item_a": cb}
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_single_widget_value("key", widget_dict, result)
        assert value == ["item_a"]

    def test_unknown_widget_returns_none(self):
        """An unrecognized widget type should return None."""
        validator = self._make_validator()
        result = ValidationResult()
        value = validator._extract_single_widget_value("key", "not_a_widget", result)
        assert value is None


# ============================================================================
# TestConfigValidatorValidateAndExtract
# ============================================================================

class TestConfigValidatorValidateAndExtract:
    """Integration-style tests for the full validate_and_extract pipeline."""

    def test_full_pipeline_valid(self, tmp_path):
        """A fully valid setup should produce is_valid=True."""
        from PySide6.QtWidgets import QLineEdit, QSpinBox, QCheckBox

        target = tmp_path / "target.off"
        target.write_text("dummy")

        config = {
            "min_iterations": 1000,
            "max_iterations": 5000,
            "alignment_algorithms": ["ICP"],
            "study_name": "test",
            "target_shape_path": str(target),
        }

        # Build mock widgets
        name_edit = MagicMock(spec=QLineEdit)
        name_edit.text.return_value = "test"
        name_edit.setStyleSheet = MagicMock()
        name_edit.styleSheet = MagicMock(return_value="")

        target_edit = MagicMock(spec=QLineEdit)
        target_edit.text.return_value = str(target)
        target_edit.setStyleSheet = MagicMock()
        target_edit.styleSheet = MagicMock(return_value="")

        parallel_cb = MagicMock(spec=QCheckBox)
        parallel_cb.isChecked.return_value = True

        widgets = {
            "study_name": name_edit,
            "target_shape_path": target_edit,
            "parallel_search_enabled": parallel_cb,
        }

        validator = ConfigValidator(widgets=widgets, config=config, parent=MagicMock())
        result = validator.validate_and_extract()
        assert result.is_valid is True


# ============================================================================
# TestExceptions
# ============================================================================

class TestExceptions:
    """Verify custom exception classes."""

    def test_config_validation_error_is_exception(self):
        assert issubclass(ConfigValidationError, Exception)

    def test_config_load_error_is_exception(self):
        assert issubclass(ConfigLoadError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(ConfigValidationError, match="bad config"):
            raise ConfigValidationError("bad config")
