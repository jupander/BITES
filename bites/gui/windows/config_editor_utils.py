"""
Utility functions and widgets for the configuration menu.

Provides:
- IterSpinBox: QSpinBox subclass for iteration-count inputs
- Widget factory functions for common Config Editor patterns
- Pure-logic utility functions extracted from ConfigWindow
"""

import logging

from PySide6.QtWidgets import (
    QLabel, QHBoxLayout, QSpinBox, QCheckBox, QWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QValidator

from bites.constants import MAX_POSSIBLE_ITERATIONS

logger = logging.getLogger(__name__)

# On Windows 11, the default Qt style renders QSpinBox up/down buttons
# side-by-side horizontally, which makes the number text invisible.
# This stylesheet constrains the button width to restore usability.
_SPINBOX_BUTTON_STYLE = """
    QSpinBox::up-button, QSpinBox::down-button {
        width: 16px;
    }
"""


# -----------------------------------------------------------------------------
# Iteration SpinBox
# -----------------------------------------------------------------------------

class IterSpinBox(QSpinBox):
    """QSpinBox that only accepts values that are multiples of its step size."""

    def __init__(self, step=1000, min_val=0, max_val=MAX_POSSIBLE_ITERATIONS, parent=None):
        super().__init__(parent)
        self._step = step
        self.setMinimum(min_val)
        self.setMaximum(max_val)
        self.setSingleStep(step)
        self.setStyleSheet(_SPINBOX_BUTTON_STYLE)

    def validate(self, text, pos) -> tuple:
        try:
            value = int(text)
        except ValueError:
            return QValidator.Invalid, text, pos

        if value % self._step == 0:
            return QValidator.Acceptable, text, pos
        else:
            return QValidator.Invalid, text, pos


# -----------------------------------------------------------------------------
# Widget factory functions
# -----------------------------------------------------------------------------

def create_iter_spin_box(value, label, step, min_val=0, max_val=MAX_POSSIBLE_ITERATIONS, disabled=False) -> tuple:
    """Create a labeled IterSpinBox in a horizontal layout."""
    layout = QHBoxLayout()
    label_widget = QLabel(label)
    label_widget.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
    label_widget.setDisabled(disabled)
    spin_box = IterSpinBox(step, min_val, max_val)
    spin_box.setValue(value)
    spin_box.setDisabled(disabled)
    layout.addWidget(label_widget)
    layout.addWidget(spin_box)
    return layout, spin_box


def create_spin_box(value, label, disabled=False) -> tuple:
    """Create a labeled QSpinBox in a horizontal layout."""
    layout = QHBoxLayout()
    label_widget = QLabel(label)
    label_widget.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
    spin_box = QSpinBox()
    spin_box.setMaximum(MAX_POSSIBLE_ITERATIONS)
    spin_box.setValue(value)
    spin_box.setDisabled(disabled)
    spin_box.setStyleSheet(_SPINBOX_BUTTON_STYLE)
    layout.addWidget(label_widget)
    layout.addWidget(spin_box)
    return layout, spin_box


def create_checkbox(label, checked, tooltip=None) -> QCheckBox:
    """Create a QCheckBox with optional tooltip."""
    checkbox = QCheckBox(label)
    checkbox.setChecked(checked)
    if tooltip:
        checkbox.setToolTip(tooltip)
    return checkbox


# -----------------------------------------------------------------------------
# Pure-logic utility functions (extracted from ConfigWindow)
# -----------------------------------------------------------------------------

def format_elapsed_time(elapsed_time_in_seconds) -> str:
    """Format elapsed time in a human-readable format."""
    if elapsed_time_in_seconds > 3600:
        elapsed_time = elapsed_time_in_seconds / 3600
        return f"{elapsed_time:.1f} hours"
    elif elapsed_time_in_seconds > 60:
        elapsed_time = elapsed_time_in_seconds / 60
        return f"{elapsed_time:.1f} minutes"
    else:
        return f"{elapsed_time_in_seconds:.2f} seconds"


# Display constants for parameter list
PARAMS_TO_SHOW_THRESHOLD = 13
PARAMS_TO_SHOW_COUNT = 9


def safe_parse_int(text: str, default: int = 0) -> int:
    """Safely parse an integer from text, returning default on failure."""
    if not text:
        return default
    try:
        value = int(text)
        return value if value >= 0 else default
    except ValueError:
        return default


def calculate_eta(
    tm_timeout: int,
    max_trials: int,
    parallel_enabled: bool,
    parallel_count: int,
    study_timeout_text: str,
) -> tuple[float, bool]:
    """
    Calculate estimated maximum duration of optimization in seconds.

    Args:
        tm_timeout: Tooth model timeout in seconds per trial.
        max_trials: Total number of trials to run.
        parallel_enabled: Whether parallel search is enabled.
        parallel_count: Number of parallel CPUs (ignored if not enabled).
        study_timeout_text: Study timeout string in hours (empty or "0" = no limit).

    Returns:
        result: Estimated max duration in seconds.
        is_run_timeout: Whether the study timeout is expected to be the limiting factor.
    """
    if parallel_enabled:
        effective_parallel = max(1, parallel_count)
        result = tm_timeout * max_trials / effective_parallel
    else:
        result = tm_timeout * max_trials

    is_run_timeout = False 

    # Cap by study timeout if set
    if study_timeout_text:
        try:
            study_timeout = float(study_timeout_text)
            if study_timeout > 0:
                if study_timeout * 3600 < result:
                    #logger.info(f"Study timeout of {study_timeout} hours is set, capping ETA at {format_elapsed_time(study_timeout * 3600)}")
                    is_run_timeout = True
                result = min(result, study_timeout * 3600)
        except ValueError:
            pass  # Ignore invalid input, just use TM-based estimate

    return result, is_run_timeout


def format_params_to_optimize(
    param_names_all: list,
    params_to_optimize: list,
    bias_param_config: dict,
) -> str:
    """
    Generate display string for parameters being optimized.

    Args:
        param_names_all: All parameter names in display order.
        params_to_optimize: List of parameter names selected for optimization.
        bias_param_config: Bias parameter configuration dict.

    Returns:
        Formatted string like "15 selected for optimization: Egr, Mgr, ..."
    """
    params_in_order = [
        param for param in param_names_all
        if param in params_to_optimize
    ]
    n_params = len(params_in_order)

    n_other_params = n_params - PARAMS_TO_SHOW_COUNT

    if n_params < PARAMS_TO_SHOW_THRESHOLD:
        first_params = ', '.join(params_in_order)
    else:
        first_params = ', '.join(params_in_order[:PARAMS_TO_SHOW_COUNT])

    # special cases for Abi, Pbi, Bbi, Lbi
    if bias_param_config.get('Abi_value_for_Pbi_too', False):
        if 'Abi, Pbi' in first_params:
            first_params = first_params.replace("Abi, Pbi", "Abi-Pbi")
        elif 'Abi' in first_params:
            first_params = first_params.replace("Abi", "Abi-Pbi")
            n_other_params -= 1
        else:
            n_other_params -= 1

    if bias_param_config.get('Bbi_value_for_Lbi_too', False):
        if 'Lbi, Bbi' in first_params:
            first_params = first_params.replace("Lbi, Bbi", "Lbi-Bbi")
        elif 'Lbi' in first_params:
            first_params = first_params.replace("Lbi", "Lbi-Bbi")
            n_other_params -= 1
        else:
            n_other_params -= 1

    if n_params < PARAMS_TO_SHOW_THRESHOLD:
        other_params = ''
    else:
        other_params = f', and {n_other_params} others'

    effective_n_params = (
        n_params
        - int(bias_param_config.get('Abi_value_for_Pbi_too', False))
        - int(bias_param_config.get('Bbi_value_for_Lbi_too', False))
    )

    return f"{effective_n_params} selected for optimization: {first_params}{other_params}"
