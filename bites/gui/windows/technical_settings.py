from pathlib import Path
from PySide6.QtWidgets import (QVBoxLayout, QPushButton, QLabel, QDialog, 
                               QLineEdit, QMessageBox, QApplication)

import bites.gui.windows.config_editor_utils as config_editor_utils
from bites.gui.mixins import DirtyTrackingMixin, GeometryPersistenceMixin
from bites.gui.widgets import PathInputWidget, PathType
from bites.constants import DEFAULT_MODEL_EXECUTABLE, DEFAULT_TM_SUBFOLDER_NAME, DEFAULT_RESULTS_BACKUP_FOLDER
from bites.paths import get_bin_folder, get_configs_folder, get_assets_folder, get_results_folder

import logging
logger = logging.getLogger(__name__)


class TechnicalSettingsWindow(GeometryPersistenceMixin, DirtyTrackingMixin, QDialog):
    GEOMETRY_KEY = "technical_settings"
    DEFAULT_GEOMETRY = None  # Use Qt's auto-sizing

    def __init__(self, config, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Technical settings')
        self.layouts = {}
        self.widgets = {}

        # Load values from the config.
        # All path/folder fields are optional overrides (keys end with '_override').
        # Empty string means "use default" - resolved at runtime in config_loader.py or paths module.
        try:
            self.model_executable_path_override = config.get('model_executable_path_override', '')
            self.tm_subfolder_name_override = config.get('tm_subfolder_name_override', '')
            self.iter_interval = config.get('iter_interval', 1000)  # fallback for older configs
            self.results_folder_override = config.get('results_folder_override', '')
            self.results_backup_folder_override = config.get('results_backup_folder_override', '')
            self.configs_folder_override = config.get('configs_folder_override', '')
            self.assets_folder_override = config.get('assets_folder_override', '')
        except KeyError as e:
            raise ValueError(f"Error loading technical settings from config: {e}")

        self.initUI()

        # Initialize dirty tracking after UI is built
        self._init_dirty_tracking()
        self._connect_dirty_signals(self.widgets)

        # Initialize geometry persistence (restores saved position/size)
        self._init_geometry_persistence()

    def initUI(self) -> None:
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Compute default paths for placeholder text
        DEFAULT_EXECUTABLE_PATH = str(get_bin_folder() / DEFAULT_MODEL_EXECUTABLE)
        DEFAULT_RESULTS_FOLDER = str(get_results_folder())
        DEFAULT_CONFIGS_FOLDER = str(get_configs_folder())
        DEFAULT_ASSETS_FOLDER = str(get_assets_folder())

        # File and folder paths using PathInputWidget
        # All path fields are optional overrides - empty means use default
        self.widgets['model_executable_path_override'] = PathInputWidget(
            parent=self,
            path_type=PathType.FILE,
            label="Model executable path (optional override):",
            default_value=self.model_executable_path_override,
            use_relative_paths=False,
        )
        self.widgets['model_executable_path_override'].line_edit.setPlaceholderText(DEFAULT_EXECUTABLE_PATH)
        self.widgets['model_executable_path_override'].setToolTip(
            'Path to the tooth model executable.\n'
            f'Leave empty to use the default: {DEFAULT_EXECUTABLE_PATH}'
        )

        # Results folder - where all output folders are created
        self.widgets['results_folder_override'] = PathInputWidget(
            parent=self,
            path_type=PathType.FOLDER,
            label="Results folder (optional override):",
            default_value=self.results_folder_override,
            use_relative_paths=False,
        )
        self.widgets['results_folder_override'].line_edit.setPlaceholderText(DEFAULT_RESULTS_FOLDER)
        self.widgets['results_folder_override'].setToolTip(
            'Folder where output folders are created for each optimization run.\n\n'
            'Each run creates an output folder here named after the study name.\n'
            f'Leave empty to use the default: {DEFAULT_RESULTS_FOLDER}'
        )

        # Path overrides for configs and assets folders
        self.widgets['configs_folder_override'] = PathInputWidget(
            parent=self,
            path_type=PathType.FOLDER,
            label="Configs folder (optional override):",
            default_value=self.configs_folder_override,
            use_relative_paths=False,
        )
        self.widgets['configs_folder_override'].line_edit.setPlaceholderText(DEFAULT_CONFIGS_FOLDER)
        self.widgets['configs_folder_override'].setToolTip(
            'Override the default configs folder location.\n'
            f'Leave empty to use the default: {DEFAULT_CONFIGS_FOLDER}'
        )

        self.widgets['assets_folder_override'] = PathInputWidget(
            parent=self,
            path_type=PathType.FOLDER,
            label="Assets folder (optional override):",
            default_value=self.assets_folder_override,
            use_relative_paths=False,
        )
        self.widgets['assets_folder_override'].line_edit.setPlaceholderText(DEFAULT_ASSETS_FOLDER)
        self.widgets['assets_folder_override'].setToolTip(
            'Override the default assets folder location.\n'
            f'Leave empty to use the default: {DEFAULT_ASSETS_FOLDER}'
        )

        # Centralized backup folder for CSV and database files
        self.widgets['results_backup_folder_override'] = PathInputWidget(
            parent=self,
            path_type=PathType.FOLDER,
            label="Centralized results backup folder (optional override):",
            default_value=self.results_backup_folder_override,
            use_relative_paths=False,
        )
        self.widgets['results_backup_folder_override'].line_edit.setPlaceholderText(DEFAULT_RESULTS_BACKUP_FOLDER)
        self.widgets['results_backup_folder_override'].setToolTip(
            'Centralized location for collecting copies of result files.\n\n'
            'WHAT THIS DOES:\n'
            f'Each optimization run creates its own output subfolder inside "{DEFAULT_RESULTS_FOLDER}/" directory (by default).\n'
            'This setting copies key (and compact) files (CSV summary, database)\n'
            'to a single shared folder as a backup.\n\n'
            f'Leave empty to use the default: {DEFAULT_RESULTS_BACKUP_FOLDER}'
        )

        # NOTE self.tm_subfolder_name_override is not a path, but a string:
        self.widgets['tm_subfolder_name_override'] = QLineEdit(self.tm_subfolder_name_override)
        self.widgets['tm_subfolder_name_override'].setPlaceholderText(DEFAULT_TM_SUBFOLDER_NAME)
        self.widgets['tm_subfolder_name_override'].setToolTip(
            'Override the default tooth model output subfolder name.\n'
            f'Leave empty to use the default: {DEFAULT_TM_SUBFOLDER_NAME}'
        )
        self.layouts['tm_subfolder_name_override'] = QVBoxLayout()
        self.layouts['tm_subfolder_name_override'].addWidget(QLabel("Tooth model output subfolder name (optional override):"))
        self.layouts['tm_subfolder_name_override'].addWidget(self.widgets['tm_subfolder_name_override'])

        # Option: iter_interval
        self.layouts['iter_interval'], self.widgets['iter_interval'] = config_editor_utils.create_iter_spin_box(
            self.iter_interval,
            'Iteration interval (default: 1000)\n(other values currently not supported)',
            step=100,
            min_val=100,
        )
        self.widgets['iter_interval'].setToolTip(
            'Step size for iteration values of the tooth model outputs.\n\n'
            'This also affects how min/max iterations are rounded.\n'
            'For example, with interval 1000, iterations are always\n'
            'multiples of 1000.\n\n'
            'DEFAULT: 1000\n'
            'NOTE: Values other than 1000 are not currently supported\n'
            'and may cause issues.'
        )

        # Button to save the settings
        save_button = QPushButton("Save settings")
        save_button.setToolTip(
            'Save technical settings and return to the Config Editor.\n\n'
            'Note: Path validity is not fully checked - the settings\n'
            'will be saved even if paths don\'t exist. This allows\n'
            'creating configurations for use on different systems.'
        )
        save_button.clicked.connect(self.save_settings)

        layout.addWidget(self.widgets['model_executable_path_override'])
        layout.addWidget(self.widgets['results_folder_override'])
        layout.addWidget(self.widgets['results_backup_folder_override'])
        layout.addWidget(self.widgets['configs_folder_override'])
        layout.addWidget(self.widgets['assets_folder_override'])
        layout.addLayout(self.layouts['tm_subfolder_name_override'])
        layout.addLayout(self.layouts['iter_interval'])
        layout.addStretch()
        layout.addWidget(save_button)

        self.setLayout(layout)

        # Apply screen-aware sizing
        self._apply_screen_aware_sizing()

    def _apply_screen_aware_sizing(self) -> None:
        """Set window size constraints based on available screen space."""
        screen = QApplication.primaryScreen().availableGeometry()

        # Set minimum size so window remains usable
        self.setMinimumSize(320, 365)

        # Set maximum height to 70% of screen height (settings dialog)
        self.setMaximumHeight(int(screen.height() * 0.70))

        # Auto-size to content, then apply constraints
        self.adjustSize()

    def save_settings(self) -> None:

        valid = True

        for widget_name, widget in self.widgets.items():
            if widget_name == 'model_executable_path_override':
                self.model_executable_path_override = widget.path
            elif widget_name == 'results_folder_override':
                self.results_folder_override = widget.path
            elif widget_name == 'results_backup_folder_override':
                self.results_backup_folder_override = widget.path
            elif widget_name == 'configs_folder_override':
                self.configs_folder_override = widget.path
            elif widget_name == 'assets_folder_override':
                self.assets_folder_override = widget.path
            elif widget_name == 'tm_subfolder_name_override':
                self.tm_subfolder_name_override = widget.text()
            elif widget_name == 'iter_interval':
                try:
                    value = int(widget.text())
                    if value < 100:
                        raise ValueError("Iteration interval must be at least 100.")
                    self.iter_interval = value
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", "Invalid input for Iteration Interval. Please enter a number greater than or equal to 100.")
                    valid = False

        ''' # maybe skip checking the paths for now. idea: prompt the user to confirm the paths if they seem* invalid? (*maybe config is being created to be used in another system/location)
        exe_path = Path(self.config['model_executable_path'])
        if exe_path.exists() and not exe_path.is_file():
            logger.warning(f"Invalid path for model_executable_path: {self.config['model_executable_path']}")
            self.widgets['model_executable_path'].setStyleSheet("background-color: pink;")
            valid = False
        else:
            self.widgets['model_executable_path'].setStyleSheet("")
        '''
        # warn the user if a custom path is specified but does not exist
        if self.model_executable_path_override and not Path(self.model_executable_path_override).exists():
            logger.warning(f"Invalid path for model_executable_path_override: {self.model_executable_path_override}")
        
        if valid:
            QMessageBox.information(self, "Settings saved", "Settings saved successfully.\n\nNOTE: the validity of the paths has not been checked.")
            self._reset_dirty()
            self.accept() # Close the dialog window

    def closeEvent(self, event) -> None:
        """Handle window close - prompt if unsaved changes."""
        if self._prompt_unsaved_changes(event):
            # GeometryPersistenceMixin.closeEvent saves geometry automatically
            super().closeEvent(event)