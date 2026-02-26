"""
Config validation utilities for the configuration menu.

Extracts validation logic from the main ConfigWindow class to improve maintainability.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from PySide6.QtWidgets import (
    QLineEdit, QSpinBox, QCheckBox, QRadioButton, QSlider, QMessageBox, QWidget
)

from bites.gui.windows.config_editor_utils import IterSpinBox
from bites.constants import DEFAULT_ITER_INTERVAL

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config Schema Definition
# -----------------------------------------------------------------------------

# Required keys and their expected types
REQUIRED_CONFIG_KEYS: Dict[str, type] = {
    'target_shape_path': str,
    'study_name': str,
    'max_trials': int,
    'study_timeout': (int, float),  # Can be either
    'min_iterations': int,
    'max_iterations': int,
    'tm_timeout': int,
    'parallel_trials': int,
    'alignment_algorithms': list,
    'params_to_optimize': list,
    'params_optimization_ranges': dict,
    'params_fixed_values': dict,
    'param_names_all': list,
    'bias_param_config': dict,
    'save_original_tooth_model_meshes': bool,
    'open_results_folder_after_run': bool,
    'technical_settings': bool,
}

# Optional keys with defaults
#
# DESIGN PRINCIPLE: All path/folder fields in Technical Settings are OPTIONAL OVERRIDES.
# - Empty string ('') means "use the default" (resolved at runtime in config_loader.py)
# - Non-empty string means "use this override value instead"
#
# This allows configs to be portable across systems - paths are resolved relative to
# the project root unless explicitly overridden.
#
OPTIONAL_CONFIG_KEYS: Dict[str, Any] = {
    # Non-override settings (these have actual default values)
    'iter_interval': DEFAULT_ITER_INTERVAL,
    'min_cut_height': None,
    'storage_mode': 'shared',
    'postgresql_config': {
        'user': 'postgres',
        'password': 'PGPASS_ENABLED',
        'host': 'localhost',
        'port': 5432,
        'db_name': 'postgres'
    },
    # Path/folder OVERRIDES (empty string = use default, resolved in config_loader.py)
    'model_executable_path_override': '',
    'tm_subfolder_name_override': '',
    'results_folder_override': '',  # folder where results subfolders are created
    'results_backup_folder_override': '',
    'configs_folder_override': '',
    'assets_folder_override': '',
}

# Constraints
MAX_STUDY_NAME_LENGTH = 64
MIN_PARALLEL_TRIALS = 1
MAX_PARALLEL_TRIALS = 1000


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


class ConfigLoadError(Exception):
    """Raised when config loading fails."""
    pass


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of config validation."""
    is_valid: bool = True
    invalid_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SchemaValidationResult:
    """Result of schema validation."""
    is_valid: bool = True
    missing_keys: List[str] = field(default_factory=list)
    type_errors: List[str] = field(default_factory=list)
    applied_defaults: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Schema Validator
# -----------------------------------------------------------------------------

class ConfigSchemaValidator:
    """Validates config dictionary against expected schema."""

    @classmethod
    def validate_and_fill_defaults(cls, config: dict) -> SchemaValidationResult:
        """
        Validate config schema and fill in missing optional keys with defaults.

        Args:
            config: The config dictionary to validate (modified in place)

        Returns:
            SchemaValidationResult with validation status and details
        """
        result = SchemaValidationResult()

        # Check required keys
        for key, expected_type in REQUIRED_CONFIG_KEYS.items():
            if key not in config:
                result.missing_keys.append(key)
                result.is_valid = False
            elif not cls._check_type(config[key], expected_type):
                result.type_errors.append(
                    f"'{key}' expected {expected_type.__name__ if isinstance(expected_type, type) else expected_type}, "
                    f"got {type(config[key]).__name__}"
                )
                result.is_valid = False

        # Fill in optional keys with defaults
        for key, default_value in OPTIONAL_CONFIG_KEYS.items():
            if key not in config:
                config[key] = default_value
                result.applied_defaults.append(key)

        # Additional constraint validations
        cls._validate_constraints(config, result)

        return result

    @classmethod
    def _check_type(cls, value: Any, expected_type) -> bool:
        """Check if value matches expected type (handles tuple of types)."""
        if isinstance(expected_type, tuple):
            return isinstance(value, expected_type)
        return isinstance(value, expected_type)

    @classmethod
    def _validate_constraints(cls, config: dict, result: SchemaValidationResult) -> None:
        """Validate additional constraints beyond type checking."""

        # Check study_name length
        study_name = config.get('study_name', '')
        if len(study_name) > MAX_STUDY_NAME_LENGTH:
            result.type_errors.append(
                f"'study_name' exceeds maximum length of {MAX_STUDY_NAME_LENGTH} characters"
            )
            result.is_valid = False

        # Check parallel_trials bounds
        parallel_trials = config.get('parallel_trials', 1)
        if not (MIN_PARALLEL_TRIALS <= parallel_trials <= MAX_PARALLEL_TRIALS):
            result.type_errors.append(
                f"'parallel_trials' must be between {MIN_PARALLEL_TRIALS} and {MAX_PARALLEL_TRIALS}"
            )
            result.is_valid = False

        # Check that alignment_algorithms is not empty (warning, not error)
        alignment_algos = config.get('alignment_algorithms', [])
        if not alignment_algos:
            # This is a warning, not an error - user might want to configure later
            pass

        # Check iteration range
        min_iter = config.get('min_iterations', 0)
        max_iter = config.get('max_iterations', 0)
        if min_iter > max_iter:
            result.type_errors.append(
                f"'min_iterations' ({min_iter}) must be less than or equal to 'max_iterations' ({max_iter})"
            )
            result.is_valid = False


# -----------------------------------------------------------------------------
# Widget Validator
# -----------------------------------------------------------------------------

class ConfigValidator:
    """Validates and extracts values from Config Editor widgets."""

    def __init__(self, widgets: dict, config: dict, parent: QWidget):
        self.widgets = widgets
        self.config = config
        self.parent = parent

    def validate_and_extract(self) -> ValidationResult:
        """
        Validate all widgets and extract their values to config.

        Returns:
            ValidationResult with status and any errors
        """
        result = ValidationResult()

        self._extract_widget_values(result)
        self._validate_iteration_ranges(result)
        self._validate_parallel_settings()
        self._validate_target_path(result)
        self._validate_alignment_algorithms(result)
        self._validate_study_name(result)

        return result

    def _extract_widget_values(self, result: ValidationResult) -> None:
        """Extract values from all widgets into the config dictionary."""
        for key, widget in self.widgets.items():
            try:
                value = self._extract_single_widget_value(key, widget, result)
                if value is not None:
                    self.config[key] = value
            except Exception as e:
                result.errors.append(f"Error extracting {key}: {e}")
                result.is_valid = False

    def _extract_single_widget_value(
        self, key: str, widget: Any, result: ValidationResult
    ) -> Any:
        """Extract value from a single widget."""
        if isinstance(widget, QLineEdit):
            return self._extract_line_edit_value(key, widget, result)
        elif isinstance(widget, QSpinBox):
            return widget.value()
        elif isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QSlider):
            return widget.value()
        elif isinstance(widget, IterSpinBox):
            return widget.value()
        elif isinstance(widget, dict):
            return self._extract_dict_widget_value(key, widget)
        else:
            logger.warning(f"Widget type for key '{key}' not recognized: {type(widget)}")
            return None

    def _extract_line_edit_value(
        self, key: str, widget: QLineEdit, result: ValidationResult
    ) -> Any:
        """Extract and convert value from QLineEdit."""
        text = widget.text()

        # Handle integer fields
        if key in ('tm_timeout', 'max_trials'):
            try:
                value = int(text) if text else 0
                if value <= 0:
                    raise ValueError("Must be positive")
                return value
            except ValueError:
                logger.warning(f"Invalid value for key '{key}': {text}")
                self._highlight_invalid(widget)
                result.invalid_fields.append(key)
                result.is_valid = False
                return 1  # Return safe default instead of invalid text

        # Handle float fields (study_timeout)
        elif key == 'study_timeout':
            if text == '':
                return 0.0
            try:
                value = float(text)
                if value < 0:
                    raise ValueError("Must be non-negative")
                return value
            except ValueError:
                logger.warning(f"Invalid value for key '{key}': {text}")
                self._highlight_invalid(widget)
                result.invalid_fields.append(key)
                result.is_valid = False
                return 0.0

        # Check for empty values (for other string fields)
        if text == '' or text is None:
            logger.warning(f"Invalid empty value for key '{key}'")
            self._highlight_invalid(widget)
            result.invalid_fields.append(key)
            result.is_valid = False
        else:
            self._clear_highlight(widget)

        return text

    def _extract_dict_widget_value(self, key: str, widget_dict: dict) -> Any:
        """Extract value from a dict of radio buttons or checkboxes."""
        # Check for radio buttons (returns single value)
        for sub_key, sub_widget in widget_dict.items():
            if isinstance(sub_widget, QRadioButton):
                if sub_widget.isChecked():
                    return sub_key

        # Must be checkboxes (returns list of checked items)
        checked_items = []
        for sub_key, sub_widget in widget_dict.items():
            if isinstance(sub_widget, QCheckBox) and sub_widget.isChecked():
                checked_items.append(sub_key)

        return checked_items

    def _validate_iteration_ranges(self, result: ValidationResult) -> None:
        """Validate that min values are less than max values."""
        range_pairs = [('min_iterations', 'max_iterations')]

        for min_key, max_key in range_pairs:
            min_val = self.config.get(min_key, 0)
            max_val = self.config.get(max_key, 0)

            if min_val >= max_val:
                logger.warning(f"Invalid values: {min_val} >= {max_val} "
                               f"({min_key} should be less than {max_key})")

                self._highlight_invalid(self.widgets.get(min_key))
                self._highlight_invalid(self.widgets.get(max_key))

                result.invalid_fields.extend([min_key, max_key])
                result.is_valid = False
            else:
                self._clear_highlight(self.widgets.get(min_key))
                self._clear_highlight(self.widgets.get(max_key))

    def _validate_parallel_settings(self) -> None:
        """Ensure parallel_trials is set correctly based on checkbox."""
        # no need to check that the number of CPUs is valid - the spinbox already limits the values appropriately
        # but set parallel_trials to 1 if parallel search is not enabled
        parallel_checkbox = self.widgets.get('parallel_search_enabled')
        if parallel_checkbox and not parallel_checkbox.isChecked():
            self.config['parallel_trials'] = 1  # set to 1 if parallel search is not enabled

    def _validate_target_path(self, result: ValidationResult) -> None:
        """Validate the target shape path exists."""
        target_path_str = self.config.get('target_shape_path', '')
        target_path = Path(target_path_str) if target_path_str else None
        widget = self.widgets.get('target_shape_path')

        if target_path is None or not target_path.exists() or not target_path.is_file():
            logger.warning(f"Invalid path for target_shape_path: {target_path_str}")

            # if the field is already highlighted in pink, the user has already been warned
            if widget and widget.styleSheet() == "background-color: pink;":
                pass  # do nothing (the settings will be saved if nothing else is wrong)
            else:
                self._highlight_invalid(widget)
                result.invalid_fields.append('target_shape_path')
                result.is_valid = False
        else:
            self._clear_highlight(widget)

    def _validate_alignment_algorithms(self, result: ValidationResult) -> None:
        """Validate that at least one alignment algorithm is selected."""
        algos = self.config.get('alignment_algorithms', [])
        if not algos:
            result.warnings.append("No alignment algorithms selected")
            # Not marking as invalid - just a warning

    def _validate_study_name(self, result: ValidationResult) -> None:
        """Validate study name length and format."""
        study_name = self.config.get('study_name', '')
        widget = self.widgets.get('study_name')

        if len(study_name) > MAX_STUDY_NAME_LENGTH:
            self._highlight_invalid(widget)
            result.invalid_fields.append('study_name')
            result.errors.append(
                f"Study name exceeds maximum length of {MAX_STUDY_NAME_LENGTH} characters"
            )
            result.is_valid = False
        elif not study_name:
            self._highlight_invalid(widget)
            result.invalid_fields.append('study_name')
            result.is_valid = False

    def _highlight_invalid(self, widget) -> None:
        """Highlight a widget as invalid."""
        if widget:
            try:
                widget.setStyleSheet("background-color: pink;")
            except AttributeError:
                pass

    def _clear_highlight(self, widget) -> None:
        """Clear invalid highlighting from a widget."""
        if widget:
            try:
                widget.setStyleSheet("")
            except AttributeError:
                pass

    def show_validation_errors(self, result: ValidationResult) -> bool:
        """
        Show a message box if validation failed.

        Returns:
            True if validation passed or user wants to continue anyway,
            False if user wants to fix errors first.
        """
        if not result.is_valid:
            reply = QMessageBox.warning(
                self.parent,
                "Validation Errors",
                "Please fix the following issues:\n\n"
                f"Invalid fields: {', '.join(result.invalid_fields)}\n\n"
                + ('\n'.join(result.errors) + '\n\n' if result.errors else '')
                + "Invalid fields are highlighted in pink.\n\n"
                "Do you want to save anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            return reply == QMessageBox.StandardButton.Yes

        # Show warnings if any (but still valid)
        if result.warnings:
            QMessageBox.information(
                self.parent,
                "Warnings",
                "Note:\n" + '\n'.join(result.warnings)
            )

        return True
