"""
Configuration menu window for the optimization settings.

This module provides the main configuration dialog for managing optimization parameters,
target shapes, parallelization settings, and technical options.
"""

import sys
import copy
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import psutil
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QCheckBox, QSpinBox, QDialog, QFileDialog, QGroupBox,
    QMessageBox, QScrollArea, QToolButton
)
from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator

import bites.gui.windows.param_ranges_window as param_ranges
import bites.gui.windows.default_values_window as params_fixed
import bites.gui.windows.technical_settings as technical_settings
import bites.gui.windows.postgres_config as postgres_config
import bites.gui.windows.config_editor_utils as utils
from bites.gui.windows.config_editor_utils import (
    format_params_to_optimize, safe_parse_int, calculate_eta,
)
from bites.gui.mixins import GeometryPersistenceMixin, DirtyTrackingMixin
from bites.gui.widgets import PathInputWidget, PathType
import bites.gui.windows.config_editor_strings as strings
import bites.gui.gui_utils as gui_utils
import bites.config.config_io as config_io

from bites.config.config_defaults import create_config_file_with_default_values
from bites.config.config_loader import Config
from bites.gui.windows.schematic_viewer import SchematicViewerDialog
from bites.gui.visualization.viewers.vispy_process_manager import VispyProcessManager, CutHeightIPCReader, SimplifierIPCReader
from bites.config.config_validation import (
    ConfigValidator,
    ConfigSchemaValidator,
    ConfigLoadError,
    MAX_STUDY_NAME_LENGTH,
)
from bites import paths
from bites.paths import get_asset, get_config

from bites.constants import (
    DEFAULT_CONFIG_FILE_NAME,
    DEFAULT_ITER_INTERVAL,
    IMPLEMENTED_ALIGNMENT_ALGORITHMS,
)


logger = logging.getLogger(__name__)


# UI Constants
FIXED_WIDTH_SMALL = 50
SCREEN_MAX_HEIGHT_RATIO = 0.90


# -----------------------------------------------------------------------------
# Main Window Class
# -----------------------------------------------------------------------------

class ConfigWindow(GeometryPersistenceMixin, DirtyTrackingMixin, QDialog):
    """Main configuration dialog for optimization settings."""

    GEOMETRY_KEY = "config_editor"
    DEFAULT_GEOMETRY = None  # Use Qt's auto-sizing

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, config_file_path: str, settings, parent=None):
        """
        Initialize the configuration window.

        Args:
            config_file_path: Path to the config YAML file
            settings: QSettings object for persisting window geometry
            parent: Parent widget

        Raises:
            ConfigLoadError: If config cannot be loaded and user declines to create default
        """
        super().__init__(parent)

        self.settings = settings  # store the settings object for later use (legacy)

        # --- Initialize config ---
        self.config: Optional[Dict[str, Any]] = None
        self.config_file_path = config_file_path
        self._load_or_create_config(config_file_path)

        # --- Initialize instance variables ---
        self.widgets: Dict[str, Any] = {}
        self.layouts: Dict[str, Any] = {}
        self.iter_interval = self._safe_get_config('iter_interval', DEFAULT_ITER_INTERVAL)

        # Initialize process manager for VisPy-based tools (Simplifier, Cut Tester)
        # Uses multiprocessing.Process for faster startup than subprocess.Popen
        self.process_manager = VispyProcessManager(self)

        # --- Build UI ---
        self._build_ui()

        # Connect dirty tracking after UI is built (uses DirtyTrackingMixin)
        self._init_dirty_tracking()
        self._connect_dirty_signals(self.widgets)

        # Apply screen-aware sizing based on scroll content
        self._apply_screen_aware_sizing()

        # Initialize geometry persistence (restores saved position/size)
        self._init_geometry_persistence()

        self.setWindowTitle(
            f'Config Editor - editing: '
            f'{Path(self.config_file_path).name}'
        )

    def _safe_get_config(self, key: str, default: Any = None) -> Any:
        """Safely get a config value with a default fallback."""
        if self.config is None:
            return default
        return self.config.get(key, default)

    def _load_or_create_config(self, config_file_path: str) -> None:
        """
        Load config from file or create new one with defaults.

        Raises:
            ConfigLoadError: If loading fails and user declines to create default
        """
        config_path = Path(config_file_path) if config_file_path else None
        if config_path and config_path.exists() and config_path.is_file():
            self.config = self._try_load_config(config_file_path)

        if not self.config:
            self._prompt_create_default_config(config_file_path)

        # Apply path overrides from config (if any)
        self._apply_path_overrides()

    def _apply_path_overrides(self) -> None:
        """Apply path folder overrides from config to the paths module."""
        if self.config is None:
            return

        overrides = {
            'results_folder_override': paths.set_results_folder,
            'configs_folder_override': paths.set_configs_folder,
            'assets_folder_override': paths.set_assets_folder,
        }
        for key, setter in overrides.items():
            value = self.config.get(key, '')
            if value:
                setter(value)
                logger.info(f"Applied {key}: {value}")

    def _try_load_config(self, config_file_path: str) -> Optional[Dict[str, Any]]:
        """Attempt to load and validate config from file."""
        try:
            config = gui_utils.load_yaml_config(config_file_path)
            logger.info(f"Config file loaded successfully from {config_file_path}")
        except Exception as e:
            logger.error(f"Error loading config dict from {config_file_path}: {e}")
            return None

        # Validate schema and fill defaults
        schema_result = ConfigSchemaValidator.validate_and_fill_defaults(config)
        if not schema_result.is_valid:
            logger.error(f"Config schema validation failed:")
            if schema_result.missing_keys:
                logger.error(f"  Missing keys: {schema_result.missing_keys}")
            if schema_result.type_errors:
                logger.error(f"  Type errors: {schema_result.type_errors}")
            return None

        if schema_result.applied_defaults:
            logger.info(f"Applied defaults for missing keys: {schema_result.applied_defaults}")

        # Also test if the config file can be loaded as a Config object
        try:
            Config.from_yaml(config_file_path)
        except Exception as e:
            logger.error(f"Error loading Config object from {config_file_path}: {e}")
            logger.error("The config file might be corrupted or not in the expected format.")
            return None

        return config

    def _prompt_create_default_config(self, config_file_path: str) -> None:
        """
        Ask user to create default config or raise exception.

        Raises:
            ConfigLoadError: If user declines to create default config
        """
        if config_file_path:
            warning_text = (
                f"No valid config file found at given path:\n'{config_file_path}'\n\n"
                f"Would you like to create a new config file '{DEFAULT_CONFIG_FILE_NAME}' "
                f"with default values?"
            )
        else:
            warning_text = (
                "No configuration file is currently loaded.\n\n"
                f"Would you like to create a new config file '{DEFAULT_CONFIG_FILE_NAME}' "
                f"with default values?"
            )

        reply = QMessageBox.warning(
            self, "Missing or invalid Config File",
            warning_text,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        if reply != QMessageBox.StandardButton.Ok:
            logger.info("User chose not to create a new config file.")
            raise ConfigLoadError("User declined to create default config file")

        logger.info("Creating a new config file with default values...")
        new_path = get_config(DEFAULT_CONFIG_FILE_NAME)
        logger.info(f"Using default config file path: '{new_path}' (existing files will be overwritten)")

        self.config = create_config_file_with_default_values(new_path)
        if not self.config:
            logger.error("Failed to create a new config file with default values.")
            QMessageBox.critical(self, "Error", "Failed to create a new config file with default values.")
            raise ConfigLoadError("Failed to create default config file")

        logger.info(f"Config file with default values created successfully at {new_path}")
        # Store as relative string path (consistent with PathInputWidget browse behavior)
        try:
            self.config_file_path = str(new_path.relative_to(Path.cwd()))
        except ValueError:
            self.config_file_path = str(new_path)

    # -------------------------------------------------------------------------
    # UI Building
    # -------------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the main UI layout."""
        self.main_layout = QVBoxLayout()

        # Top row with tooltip hint and help button
        self._create_top_row()

        # Create sections for the different option groups
        self._create_study_group()
        self._create_target_group()
        self._create_parameter_group()
        self._create_iter_value_group()
        self._create_duration_group()
        self._create_parallelization_group()
        self._create_alignment_algorithms_group()
        self._create_output_folder_group()
        self._create_technical_settings_group()
        self._update_eta()

        # Import/export and save buttons
        self._create_import_export_buttons()

        self.save_button = QPushButton(f'Save changes (to {Path(self.config_file_path).name})')
        self.save_button.setToolTip(strings.SAVE_BUTTON_TOOLTIP)
        self.save_button.clicked.connect(self._save_settings)
        self.main_layout.addWidget(self.save_button)

        # Wrap in scroll area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_content = QWidget()
        self._scroll_content.setLayout(self.main_layout)
        self._scroll_area.setWidget(self._scroll_content)

        meta_layout = QVBoxLayout(self)
        meta_layout.addWidget(self._scroll_area)
        self.setLayout(meta_layout)

    def _create_top_row(self) -> None:
        """Create the top row with tooltip hint, diagram, and guide buttons."""
        top_row_layout = QHBoxLayout()

        tooltip_tip_label = QLabel("Hover cursor over labels, buttons and fields for info")
        tooltip_tip_label.setToolTip("Yes, just like that!")
        top_row_layout.addWidget(tooltip_tip_label)
        top_row_layout.addStretch()

        help_button = QToolButton()
        help_button.setText("Diagram")
        help_button.setToolTip(strings.HELP_BUTTON_TOOLTIP)
        help_button.clicked.connect(self._show_schematic)
        top_row_layout.addWidget(help_button)

        guide_button = QToolButton()
        guide_button.setText("Guide")
        guide_button.setToolTip(strings.GUIDE_BUTTON_TOOLTIP)
        guide_button.clicked.connect(lambda: gui_utils.open_users_guide(self, page=24))
        top_row_layout.addWidget(guide_button)

        self.main_layout.addLayout(top_row_layout)

    def _apply_screen_aware_sizing(self) -> None:
        """Set window size constraints based on available screen space."""
        screen = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(400, 500)

        max_height = int(screen.height() * SCREEN_MAX_HEIGHT_RATIO)
        self.setMaximumHeight(max_height)

        content_height = self._scroll_content.sizeHint().height()
        margins_height = 30
        ideal_height = content_height + margins_height
        target_height = min(ideal_height, max_height)
        self.resize(self.sizeHint().width(), target_height)

    def _create_import_export_buttons(self) -> None:
        """Create the import/export buttons row."""
        horizontal_layout = QHBoxLayout()

        export_button = QPushButton('Save as...')
        export_button.setToolTip(strings.EXPORT_BUTTON_TOOLTIP)
        export_button.clicked.connect(self._export_settings)

        import_button = QPushButton('Import settings from...')
        import_button.setToolTip(strings.IMPORT_BUTTON_TOOLTIP)
        import_button.clicked.connect(self._import_settings)

        horizontal_layout.addWidget(export_button)
        horizontal_layout.addWidget(import_button)
        self.main_layout.addLayout(horizontal_layout)

    # -------------------------------------------------------------------------
    # Group Creation Methods
    # -------------------------------------------------------------------------

    def _create_study_group(self) -> None:
        """Create the study settings group."""
        group_box = QGroupBox('Study')
        group_box.setToolTip(strings.STUDY_GROUP_TOOLTIP)
        layout = QHBoxLayout()

        # Option: study_name
        label = QLabel("Study/database name:")
        label.setToolTip(strings.STUDY_NAME_LABEL_TOOLTIP)
        layout.addWidget(label)

        study_name = self._safe_get_config('study_name', 'default_study')
        self.widgets['study_name'] = QLineEdit(study_name)
        self.widgets['study_name'].setValidator(
            QRegularExpressionValidator(QRegularExpression(f"^[a-zA-Z0-9_]{{1,{MAX_STUDY_NAME_LENGTH}}}$"))
        )
        self.widgets['study_name'].setMaxLength(MAX_STUDY_NAME_LENGTH)
        self.widgets['study_name'].setToolTip(strings.STUDY_NAME_WIDGET_TOOLTIP)
        layout.addWidget(self.widgets['study_name'])

        layout.addStretch()

        # Option: optimization_run_timeout
        label = QLabel("Optimization run timeout (hours):")
        label.setToolTip(strings.STUDY_TIMEOUT_LABEL_TOOLTIP)
        layout.addWidget(label)

        study_timeout = self._safe_get_config('study_timeout', 0)
        self.widgets['study_timeout'] = QLineEdit(str(study_timeout))
        self.widgets['study_timeout'].setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^[0-9]+(\.[0-9]+)?$"))
        )
        self.widgets['study_timeout'].textChanged.connect(self._update_eta)
        self.widgets['study_timeout'].setToolTip(strings.STUDY_TIMEOUT_WIDGET_TOOLTIP)
        self.widgets['study_timeout'].setFixedWidth(FIXED_WIDTH_SMALL)
        layout.addWidget(self.widgets['study_timeout'])

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def _create_target_group(self) -> None:
        """Create the target shape settings group."""
        group_box = QGroupBox('Target shape')
        group_box.setToolTip(strings.TARGET_GROUP_TOOLTIP)
        layout = QVBoxLayout()

        # Option: target_shape_path
        target_path = self._safe_get_config('target_shape_path', '')
        self.widgets['target_shape_path'] = PathInputWidget(
            parent=self, path_type=PathType.MESH, default_value=target_path,
        )
        layout.addWidget(self.widgets['target_shape_path'])
        self.widgets['target_shape_path'].setToolTip(strings.TARGET_PATH_WIDGET_TOOLTIP)
        self.widgets['target_shape_path'].line_edit.textChanged.connect(self._on_target_mesh_changed)

        # Simplifier and cut height buttons
        self.simplifier_button = QPushButton('Simplify target mesh')
        self.simplifier_button.setToolTip(strings.SIMPLIFIER_BUTTON_TOOLTIP)
        self.simplifier_button.setDisabled(not target_path)
        self.simplifier_button.clicked.connect(self._open_simplifier_window)

        self.cut_height_button = QPushButton('Set cut height limit')
        self.cut_height_button.setToolTip(strings.CUT_HEIGHT_BUTTON_TOOLTIP)
        self.cut_height_button.setDisabled(not target_path)
        self.cut_height_button.clicked.connect(self._open_cut_height_window)

        inner_layout = QHBoxLayout()
        inner_layout.addWidget(self.simplifier_button)
        inner_layout.addWidget(self.cut_height_button)
        layout.addLayout(inner_layout)

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def _create_parameter_group(self) -> None:
        """Create the parameters settings group."""
        group_box = QGroupBox('Search space: Model parameters')
        group_box.setToolTip(strings.PARAMETER_GROUP_TOOLTIP)
        layout_v = QVBoxLayout()
        layout_h = QHBoxLayout()

        self.params_to_opt_label = QLabel(self._get_params_to_opt_string())
        self.params_to_opt_label.setToolTip(strings.PARAMS_TO_OPT_LABEL_TOOLTIP)
        layout_v.addWidget(self.params_to_opt_label)

        self.opt_button = QPushButton('Select parameters to optimize')
        self.opt_button.setToolTip(strings.OPT_BUTTON_TOOLTIP)
        self.opt_button.clicked.connect(self._open_opt_ranges_window)
        layout_h.addWidget(self.opt_button)

        self.default_button = QPushButton('Set values for fixed parameters')
        self.default_button.setToolTip(strings.DEFAULT_VALUES_BUTTON_TOOLTIP)
        self.default_button.clicked.connect(self._open_default_values_window)
        layout_h.addWidget(self.default_button)

        layout_v.addLayout(layout_h)
        group_box.setLayout(layout_v)
        self.main_layout.addWidget(group_box)

    def _create_iter_value_group(self) -> None:
        """Create the iteration settings group."""
        group_box = QGroupBox('Search space: Simulation length (iter)')
        group_box.setToolTip(strings.ITER_GROUP_TOOLTIP)

        # Option: min_iterations
        min_iter = self._safe_get_config('min_iterations', 1000)
        self.layouts['min_iterations'], self.widgets['min_iterations'] = utils.create_iter_spin_box(
            min_iter, 'Min iterations:', step=self.iter_interval
        )
        self.widgets['min_iterations'].setToolTip(strings.MIN_ITERATIONS_TOOLTIP)

        # Option: max_iterations
        max_iter = self._safe_get_config('max_iterations', 5000)
        self.layouts['max_iterations'], self.widgets['max_iterations'] = utils.create_iter_spin_box(
            max_iter, 'Max iterations:', step=self.iter_interval
        )
        self.widgets['max_iterations'].setToolTip(strings.MAX_ITERATIONS_TOOLTIP)

        layout = QHBoxLayout()
        layout.addLayout(self.layouts['min_iterations'])
        layout.addLayout(self.layouts['max_iterations'])
        layout.addStretch()

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def _create_duration_group(self) -> None:
        """Create the persistence/duration settings group."""
        group_box = QGroupBox('Persistence')
        group_box.setToolTip(strings.DURATION_GROUP_TOOLTIP)
        layout = QVBoxLayout()

        # --- horizontal level 1 - tm_timeout and max_trials ---
        inner_layout = QHBoxLayout()

        # Option: tm_timeout
        label = QLabel("Tooth model timeout (seconds):")
        label.setToolTip(strings.TM_TIMEOUT_LABEL_TOOLTIP)
        inner_layout.addWidget(label)

        tm_timeout = self._safe_get_config('tm_timeout', 60)
        self.widgets['tm_timeout'] = QLineEdit(str(tm_timeout))
        self.widgets['tm_timeout'].setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^[1-9]\d{0,3}$"))
        )
        self.widgets['tm_timeout'].textChanged.connect(self._update_eta)
        self.widgets['tm_timeout'].setToolTip(strings.TM_TIMEOUT_WIDGET_TOOLTIP)
        self.widgets['tm_timeout'].setFixedWidth(FIXED_WIDTH_SMALL)
        inner_layout.addWidget(self.widgets['tm_timeout'])

        inner_layout.addStretch()

        # Option: max_trials
        label = QLabel("Max trials to run:")
        label.setToolTip(strings.MAX_TRIALS_LABEL_TOOLTIP)
        inner_layout.addWidget(label)

        max_trials = self._safe_get_config('max_trials', 100)
        self.widgets['max_trials'] = QLineEdit(str(max_trials))
        self.widgets['max_trials'].setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^[1-9]\d{0,4}$"))
        )
        self.widgets['max_trials'].textChanged.connect(self._update_eta)
        self.widgets['max_trials'].setToolTip(strings.MAX_TRIALS_WIDGET_TOOLTIP)
        self.widgets['max_trials'].setFixedWidth(FIXED_WIDTH_SMALL)
        inner_layout.addWidget(self.widgets['max_trials'])

        layout.addLayout(inner_layout)

        # --- horizontal level 2 - ETA ---
        eta_layout = QHBoxLayout()
        label = QLabel("-> Approximated max duration of optimization run:")
        label.setToolTip(strings.ETA_LABEL_TOOLTIP)
        eta_layout.addWidget(label)

        self.eta_label = QLabel("Calculating...")
        self.eta_label.setToolTip(strings.ETA_VALUE_TOOLTIP)
        eta_layout.addWidget(self.eta_label)

        layout.addLayout(eta_layout)

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def _create_parallelization_group(self) -> None:
        """Create the parallelization settings group."""
        group_box = QGroupBox('Parallelization')
        group_box.setToolTip(strings.PARALLEL_GROUP_TOOLTIP)
        layout = QVBoxLayout()

        # Get the number of CPUs on the system
        self.num_cpus = psutil.cpu_count(logical=True) or 1

        cpu_info_label = QLabel(f'Total available CPUs (on this system): {self.num_cpus}')
        cpu_info_label.setToolTip(strings.cpu_info_label_tooltip(self.num_cpus))
        layout.addWidget(cpu_info_label)

        inner_layout = QHBoxLayout()

        # Option: parallel_search_enabled
        parallel_trials = self._safe_get_config('parallel_trials', 1)
        self.widgets['parallel_search_enabled'] = utils.create_checkbox(
            label='Parallel search',
            checked=parallel_trials > 1,
            tooltip=strings.PARALLEL_SEARCH_TOOLTIP,
        )
        self.widgets['parallel_search_enabled'].stateChanged.connect(self._toggle_parallel_search_enabled)
        self.widgets['parallel_search_enabled'].stateChanged.connect(self._update_eta)
        inner_layout.addWidget(self.widgets['parallel_search_enabled'])

        # Option: parallel_trials
        spinbox_value = max(2, parallel_trials) if parallel_trials > 1 else 2
        self.layouts['parallel_trials'], self.widgets['parallel_trials'] = utils.create_spin_box(
            spinbox_value, 'Number of CPUs to utilize:',
            disabled=not self.widgets['parallel_search_enabled'].isChecked(),
        )
        self.widgets['parallel_trials'].setRange(2, 1000)
        self.widgets['parallel_trials'].textChanged.connect(self._update_eta)
        self.widgets['parallel_trials'].setToolTip(strings.parallel_trials_tooltip(self.num_cpus))
        self.widgets['parallel_trials'].setFixedWidth(FIXED_WIDTH_SMALL)

        inner_layout.addLayout(self.layouts['parallel_trials'])
        layout.addLayout(inner_layout)

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def _create_alignment_algorithms_group(self) -> None:
        """Create the alignment algorithms settings group."""
        group_box = QGroupBox('Alignment algorithms')
        group_box.setToolTip(strings.ALIGNMENT_GROUP_TOOLTIP)
        layout = QHBoxLayout()

        alignment_algorithm_checkboxes_dict = {}
        selected_algos = self._safe_get_config('alignment_algorithms', [])

        for algo in IMPLEMENTED_ALIGNMENT_ALGORITHMS:
            tooltip = strings.ALGO_TOOLTIPS.get(algo, f'{algo} alignment algorithm')
            box = utils.create_checkbox(algo, algo in selected_algos, tooltip=tooltip)
            layout.addWidget(box)
            alignment_algorithm_checkboxes_dict[algo] = box

        self.widgets['alignment_algorithms'] = alignment_algorithm_checkboxes_dict

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def _create_output_folder_group(self) -> None:
        """Create the output folder settings group."""
        group_box = QGroupBox('Output folder')
        group_box.setToolTip(strings.OUTPUT_GROUP_TOOLTIP)
        layout = QVBoxLayout()
        inner_layout = QHBoxLayout()

        save_original = self._safe_get_config('save_original_tooth_model_meshes', False)
        self.widgets['save_original_tooth_model_meshes'] = utils.create_checkbox(
            'Save original unaligned meshes', save_original, strings.SAVE_ORIGINAL_MESHES_TOOLTIP
        )
        inner_layout.addWidget(self.widgets['save_original_tooth_model_meshes'])

        inner_layout.addStretch()

        open_folder = self._safe_get_config('open_results_folder_after_run', False)
        self.widgets['open_results_folder_after_run'] = utils.create_checkbox(
            'Open output folder after run', open_folder, strings.OPEN_RESULTS_FOLDER_TOOLTIP
        )
        inner_layout.addWidget(self.widgets['open_results_folder_after_run'])

        layout.addLayout(inner_layout)
        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    def _create_technical_settings_group(self) -> None:
        """Create the technical settings group."""
        group_box = QGroupBox('Advanced settings')
        group_box.setToolTip(strings.TECHNICAL_GROUP_TOOLTIP)
        layout = QHBoxLayout()

        tech_enabled = self._safe_get_config('technical_settings', False)
        self.widgets['technical_settings'] = utils.create_checkbox(
            'Enable', tech_enabled, tooltip=strings.TECHNICAL_ENABLE_TOOLTIP
        )
        self.widgets['technical_settings'].stateChanged.connect(self._toggle_technical_settings)
        layout.addWidget(self.widgets['technical_settings'])

        self.tech_button = QPushButton('Technical settings')
        self.tech_button.setDisabled(not tech_enabled)
        self.tech_button.setToolTip(strings.TECHNICAL_BUTTON_TOOLTIP)
        self.tech_button.clicked.connect(self._open_technical_settings_window)
        layout.addWidget(self.tech_button)

        self.postgres_button = QPushButton('PostgreSQL settings')
        self.postgres_button.setDisabled(not tech_enabled)
        self.postgres_button.setToolTip(strings.POSTGRES_BUTTON_TOOLTIP)
        self.postgres_button.clicked.connect(self._open_postgres_settings_window)
        layout.addWidget(self.postgres_button)

        group_box.setLayout(layout)
        self.main_layout.addWidget(group_box)

    # -------------------------------------------------------------------------
    # Dialog Window Methods
    # -------------------------------------------------------------------------

    def _show_schematic(self) -> None:
        """Show the optimization process schematic diagram."""
        try:
            dialog = SchematicViewerDialog(self)
            dialog.exec()
        except Exception as e:
            logger.error(f"Error showing schematic: {e}")
            QMessageBox.warning(self, "Error", f"Failed to show schematic:\n{e}")

    def _open_simplifier_window(self) -> None:
        """Open mesh simplifier in a separate process."""
        mesh_path = self._safe_get_config('target_shape_path', '')

        if not mesh_path or not Path(mesh_path).exists():
            QMessageBox.warning(self, "File Not Found", f"Mesh file not found: {mesh_path}")
            return

        self.process_manager.launch_simplifier(
            mesh_path=mesh_path, on_finished=self._on_simplifier_finished
        )

    def _on_simplifier_finished(self, return_code: int) -> None:
        """Handle simplifier subprocess completion."""
        if return_code == 0:
            logger.info("Simplifier subprocess completed successfully")
            simplified_path = SimplifierIPCReader.read_result()
            if simplified_path:
                self.widgets['target_shape_path'].line_edit.setText(simplified_path)
                self.config['target_shape_path'] = simplified_path
                self.config['min_cut_height'] = None
                self._mark_dirty()
                logger.info(f"Target shape path updated to simplified mesh: {simplified_path}")
                QMessageBox.information(
                    self, "Target Shape Updated",
                    f"The target shape has been updated to the simplified mesh:\n\n{simplified_path}"
                )
        else:
            logger.warning(f"Simplifier subprocess exited with code {return_code}")

    def _open_cut_height_window(self) -> None:
        """Open cut height tester in a separate process."""
        mesh_path = self._safe_get_config('target_shape_path', '')
        cut_height = self._safe_get_config('min_cut_height', None)

        if not mesh_path or not Path(mesh_path).exists():
            QMessageBox.warning(self, "File Not Found", f"Mesh file not found: {mesh_path}")
            return

        self.process_manager.launch_cut_tester(
            mesh_path=mesh_path,
            cut_height=str(cut_height) if cut_height is not None else 'None',
            on_finished=self._on_cut_tester_finished
        )

    def _on_cut_tester_finished(self, return_code: int) -> None:
        """Handle cut tester subprocess completion."""
        if return_code == 0:
            cut_height = CutHeightIPCReader.read_result()
            if cut_height is not None:
                self.config['min_cut_height'] = cut_height
                self._mark_dirty()
                logger.info(f"Cut height saved from subprocess: {cut_height}")

    def _open_opt_ranges_window(self) -> None:
        """Open the optimization ranges dialog."""
        try:
            dialog = param_ranges.OptimizationRangesWindow(self.config, parent=self)
            if dialog.exec():
                self._save_optimization_ranges_to_config(dialog)
                self._update_params_to_opt_label()
                self._mark_dirty()
        except Exception as e:
            logger.error(f"Error opening optimization ranges window: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open optimization ranges window:\n{e}")

    def _save_optimization_ranges_to_config(self, dialog) -> None:
        """Save optimization ranges from dialog to config."""
        self.config['params_optimization_ranges'] = dialog.optimization_ranges_dict.copy()
        self.config['params_to_optimize'] = dialog.params_to_optimize.copy()
        self.config['bias_param_config'] = {
            'Abi_value_for_Pbi_too': dialog.Abi_value_for_Pbi_too,
            'Abi_Pbi_factor': dialog.Abi_Pbi_factor,
            'Bbi_value_for_Lbi_too': dialog.Bbi_value_for_Lbi_too,
            'Bbi_Lbi_factor': dialog.Bbi_Lbi_factor
        }

    def _open_default_values_window(self) -> None:
        """Open the default values dialog."""
        try:
            dialog = params_fixed.DefaultValuesWindow(self.config, parent=self)
            if dialog.exec():
                self.config['params_fixed_values'] = dialog.default_values_dict.copy()
                self._mark_dirty()
        except Exception as e:
            logger.error(f"Error opening default values window: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open default values window:\n{e}")

    def _open_technical_settings_window(self) -> None:
        """Open the technical settings dialog."""
        try:
            dialog = technical_settings.TechnicalSettingsWindow(self.config, parent=self)
            if dialog.exec():
                self._update_iter_intervals(dialog.iter_interval)
                self._save_technical_settings_to_config(dialog)
                self._mark_dirty()
        except Exception as e:
            logger.error(f"Error opening technical settings window: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open technical settings window:\n{e}")

    def _save_technical_settings_to_config(self, dialog) -> None:
        """Save technical settings from dialog to config."""
        try:
            self.config['model_executable_path_override'] = dialog.model_executable_path_override
            self.config['tm_subfolder_name_override'] = dialog.tm_subfolder_name_override
            self.config['iter_interval'] = dialog.iter_interval
            self.config['results_folder_override'] = dialog.results_folder_override
            self.config['results_backup_folder_override'] = dialog.results_backup_folder_override
            self.config['configs_folder_override'] = dialog.configs_folder_override
            self.config['assets_folder_override'] = dialog.assets_folder_override
        except Exception as e:
            logger.error(f"Error saving technical settings: {e}")
            QMessageBox.warning(self, "Error", f"An error occurred while saving the technical settings:\n{e}")

    def _open_postgres_settings_window(self) -> None:
        """Open the PostgreSQL settings dialog."""
        try:
            dialog = postgres_config.ConfigDialog(self.config, self)
            if dialog.exec():
                self.config['postgresql_config'] = copy.deepcopy(dialog.config['postgresql_config'])
                self._mark_dirty()
        except Exception as e:
            logger.error(f"Error opening PostgreSQL settings window: {e}")
            QMessageBox.warning(self, "Error", f"Failed to open PostgreSQL settings window:\n{e}")

    # -------------------------------------------------------------------------
    # Save/Export/Import Methods
    # -------------------------------------------------------------------------

    def _update_config_file_display(self) -> None:
        """Update window title and save button to reflect the current config file."""
        name = Path(self.config_file_path).name
        self.setWindowTitle(f'Config Editor - editing: {name}')
        self.save_button.setText(f'Save changes (to {name})')

    def _save_settings(self) -> None:
        """Save settings to the config file."""
        try:
            validator = ConfigValidator(self.widgets, self.config, self)
            result = validator.validate_and_extract()
            if not validator.show_validation_errors(result):
                return
        except Exception as e:
            logger.error(f"Error validating settings: {e}")
            QMessageBox.warning(self, "Error", f"An error occurred while validating the settings:\n{e}")
            return

        try:
            config_io.save_yaml_config(self.config_file_path, self.config, template=False)
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save config file:\n{e}")
            return

        QMessageBox.information(self, "Saved", "Settings saved successfully!")
        self._reset_dirty()
        self.accept()

    def _export_settings(self) -> None:
        """Export settings to a new file and switch to it as the loaded config."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Settings as...",
            str(Path(self.config_file_path).parent),
            "YAML files (*.yaml)", "",
            gui_utils.get_file_dialog_options()
        )
        if file_path:
            if not file_path.endswith('.yaml'):
                file_path += '.yaml'
            try:
                validator = ConfigValidator(self.widgets, self.config, self)
                result = validator.validate_and_extract()
                if not validator.show_validation_errors(result):
                    return
                config_io.save_yaml_config(file_path, self.config, template=False)

                # Switch the loaded config to the newly saved file
                self.config_file_path = file_path
                self._update_config_file_display()
                self._reset_dirty()

                new_name = Path(file_path).name
                QMessageBox.information(
                    self, "Exported",
                    f"Settings exported successfully!\n\n"
                    f"The loaded config file has been switched to:\n{new_name}"
                )
            except Exception as e:
                logger.error(f"Error exporting settings: {e}")
                QMessageBox.warning(self, "Export Error", f"Failed to export settings:\n{e}")

    def _import_settings(self) -> None:
        """Import settings from a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Settings from...",
            str(Path(self.config_file_path).parent),
            "YAML files (*.yaml)", "",
            gui_utils.get_file_dialog_options()
        )
        if file_path:
            try:
                new_config = gui_utils.load_yaml_config(file_path)

                schema_result = ConfigSchemaValidator.validate_and_fill_defaults(new_config)
                if not schema_result.is_valid:
                    error_msg = "Invalid config file:\n"
                    if schema_result.missing_keys:
                        error_msg += f"\nMissing keys: {', '.join(schema_result.missing_keys)}"
                    if schema_result.type_errors:
                        error_msg += f"\nType errors: {', '.join(schema_result.type_errors)}"
                    QMessageBox.warning(self, "Import Error", error_msg)
                    return

                self.config = new_config
                self._update_widgets()
                self._mark_dirty()
                QMessageBox.information(self, "Imported", "Settings imported successfully!")

            except Exception as e:
                logger.error(f"Error importing settings: {e}")
                QMessageBox.warning(self, "Import Error", f"Failed to import settings:\n{e}")

    # -------------------------------------------------------------------------
    # Widget Update Methods
    # -------------------------------------------------------------------------

    def _update_widgets(self) -> None:
        """Update all widgets with values from the config dictionary."""
        for key, widget in self.widgets.items():
            config_value = self.config.get(key)
            if config_value is None:
                continue

            try:
                if isinstance(widget, QLineEdit):
                    widget.setText(str(config_value))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(config_value)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(config_value)
                elif isinstance(widget, utils.IterSpinBox):
                    widget.setValue(config_value)
                elif isinstance(widget, dict):
                    if key == 'alignment_algorithms':
                        for sub_key, sub_widget in widget.items():
                            sub_widget.setChecked(sub_key in config_value)
                    else:
                        for sub_key, sub_widget in widget.items():
                            sub_widget.setChecked(sub_key == config_value)
                else:
                    logger.debug(f"Widget type for key '{key}' not recognized: {type(widget)}")
            except Exception as e:
                logger.warning(f"Error updating widget '{key}': {e}")

        self._update_params_to_opt_label()
        self._update_eta()

    def _toggle_technical_settings(self) -> None:
        """Toggle technical settings enabled state."""
        try:
            checkbox_on = self.widgets['technical_settings'].isChecked()
            self.config['technical_settings'] = checkbox_on
            self.tech_button.setDisabled(not checkbox_on)
            self.postgres_button.setDisabled(not checkbox_on)
        except Exception as e:
            logger.error(f"Error toggling technical settings: {e}")

    def _toggle_parallel_search_enabled(self) -> None:
        """Toggle parallel search enabled state."""
        self.widgets['parallel_trials'].setDisabled(
            not self.widgets['parallel_search_enabled'].isChecked()
        )

    def _update_iter_intervals(self, iter_interval: int) -> None:
        """Update iteration interval and adjust related spinboxes."""
        if iter_interval <= 0:
            logger.warning(f"Invalid iter_interval: {iter_interval}, using default")
            iter_interval = DEFAULT_ITER_INTERVAL

        self.iter_interval = iter_interval

        changed_values_dict = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, utils.IterSpinBox):
                widget.setSingleStep(self.iter_interval)
                if widget.value() % self.iter_interval != 0:
                    old_value = widget.value()
                    new_value = round(old_value / self.iter_interval) * self.iter_interval
                    widget.setValue(new_value)
                    changed_values_dict[key] = (old_value, new_value)
                    logger.debug(f"Updated {key} to {new_value} (was {old_value})")

        if changed_values_dict:
            message = "The following values were updated to be multiples of the new iteration interval:\n"
            for key, (old_value, new_value) in changed_values_dict.items():
                message += f"{key}: {old_value} -> {new_value}\n"
            QMessageBox.information(self, "Updated values", message)

    def _update_params_to_opt_label(self) -> None:
        """Update the parameters-to-optimize label."""
        self.params_to_opt_label.setText(self._get_params_to_opt_string())

    def _get_params_to_opt_string(self) -> str:
        """Generate display string for parameters being optimized."""
        return format_params_to_optimize(
            param_names_all=self._safe_get_config('param_names_all', []),
            params_to_optimize=self._safe_get_config('params_to_optimize', []),
            bias_param_config=self._safe_get_config('bias_param_config', {}),
        )

    def _on_target_mesh_changed(self) -> None:
        """Handle target mesh path change."""
        target_shape_path = self.widgets['target_shape_path'].path

        self.config['target_shape_path'] = target_shape_path
        self.config['min_cut_height'] = None

        target_path = Path(target_shape_path) if target_shape_path else None
        path_valid = target_path is not None and target_path.exists() and target_path.is_file()
        self.simplifier_button.setDisabled(not path_valid)
        self.cut_height_button.setDisabled(not path_valid)

        if not path_valid and target_shape_path:
            QMessageBox.warning(
                self, "Invalid Path",
                "The specified target shape path is invalid.\n"
                "Simplify Mesh and Set Cut Height functionalities may not work correctly."
            )

    def _update_eta(self) -> None:
        """Update the ETA label based on current settings."""
        try:
            tm_timeout = safe_parse_int(self.widgets['tm_timeout'].text(), 0)
            max_trials = safe_parse_int(self.widgets['max_trials'].text(), 0)
            parallel_enabled = self.widgets['parallel_search_enabled'].isChecked()
            parallel_count = self.widgets['parallel_trials'].value()

            result, is_run_timeout = calculate_eta(
                tm_timeout=tm_timeout,
                max_trials=max_trials,
                parallel_enabled=parallel_enabled,
                parallel_count=parallel_count,
                study_timeout_text=self.widgets['study_timeout'].text(),
            )

            if is_run_timeout:
                self.eta_label.setText(f"{utils.format_elapsed_time(result)} (Optimization run timeout)")
            else:
                self.eta_label.setText(utils.format_elapsed_time(result))

        except Exception as e:
            self.eta_label.setText("--")
            logger.debug(f"Error calculating ETA: {e}")

    # -------------------------------------------------------------------------
    # Window Events
    # -------------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        if self._is_dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save
            )

            if reply == QMessageBox.StandardButton.Save:
                self._save_settings()
                if self._is_dirty:
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        # Terminate any running VisPy tool subprocesses (simplifier, cut tester)
        self.process_manager.cleanup_all()

        # GeometryPersistenceMixin.closeEvent saves geometry automatically
        super().closeEvent(event)


# -----------------------------------------------------------------------------
# Main Entry Point (for testing)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    from PySide6.QtCore import QSettings

    logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)

    settings = QSettings("BITES", "ConfigEditor")

    try:
        config_window = ConfigWindow('configs/optimization_config.yaml', settings)
        config_window.show()
        sys.exit(app.exec())
    except ConfigLoadError as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
