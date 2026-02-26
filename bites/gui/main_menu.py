"""
Main Menu Window - Main application window for BITES.

This module provides the main window with tabs for setup/optimization
and results visualization.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QDialog, QMessageBox, QLabel, QTabWidget, QToolButton, QCheckBox
)

# Import tab widgets from extracted modules
from bites.gui.tabs.setup_tab import SetupTab
from bites.gui.tabs.visualize_tab import VisualizeTab

# Import other GUI modules
import bites.gui.gui_utils as gui_utils
import bites.gui.windows.config_editor as config_editor
from bites.config.config_validation import ConfigLoadError
import bites.gui.windows.optimization_console as optimization_console
from bites.gui.mixins import GeometryPersistenceMixin

if TYPE_CHECKING:
    from bites.core.application_controller import ApplicationController

logger = logging.getLogger(__name__)

# ============================================================================
# NOTE: SetupTab and VisualizeTab have been extracted to separate files
# in gui/tabs/ directory. They are imported at the top of this file.
# ============================================================================

class MainMenuWindow(GeometryPersistenceMixin, QMainWindow):
    GEOMETRY_KEY = "main_window"
    DEFAULT_GEOMETRY = None  # Use Qt's auto-sizing

    def __init__(self, controller: 'ApplicationController'):
        super().__init__()
        self.controller = controller
        self.widgets = {}

        logger.info("Initializing MainMenuWindow")
        self.init_ui()

        # Apply screen-aware sizing with margins
        self._apply_screen_aware_sizing()

        # Initialize geometry persistence (restores saved position/size)
        self._init_geometry_persistence(controller.settings)

    def init_ui(self) -> None:

        self.setWindowTitle("Main Menu")

        layout = QVBoxLayout()

        # Top row with native dialog toggle and guide button
        top_row = QHBoxLayout()

        top_row.addStretch()

        # Native dialog toggle
        native_dialog_checkbox = QCheckBox("Use native file dialogs")
        native_dialog_checkbox.setToolTip(
            "Toggle between the OS native file dialog and Qt's built-in dialog.\n"
            "(Dialogs that open when the 'Browse' buttons are clicked)\n\n"
            "Native dialogs match the OS look but may briefly freeze the UI.\n"
            "Qt dialogs avoid this delay."
        )
        native_dialog_checkbox.setChecked(self.controller.settings.use_native_dialog)
        native_dialog_checkbox.toggled.connect(self._on_native_dialog_toggled)
        top_row.addWidget(native_dialog_checkbox)

        # Guide button
        guide_button = QToolButton()
        guide_button.setText("Guide")
        guide_button.setToolTip(
            "Open User's Guide\n\n"
            "Opens the PDF documentation in your\n"
            "system's default PDF viewer."
        )
        guide_button.clicked.connect(lambda: gui_utils.open_users_guide(self, page=22))
        top_row.addWidget(guide_button)

        layout.addLayout(top_row)

        self.status_label = QLabel("Main Menu Ready")
        self.status_label.setWordWrap(True)  # Allow text to wrap instead of forcing window width
        self.status_label.setMinimumWidth(50)  # Allow shrinking

        tabs = QTabWidget()

        # Tab 1: Setup and Run
        self.setup_tab = SetupTab(self.controller)
        tabs.addTab(self.setup_tab, "Setup and run")

        self.setup_tab.config_setup_button.clicked.connect(lambda: self.run_config_editor())
        self.setup_tab.start_new_button.clicked.connect(lambda: self.run_optimization_script())

        # Tab 2: Visualize results
        self.visualize_tab = VisualizeTab(self.controller)
        #self.update_results_folder_path()  # Load the output folder path from settings # NOTE currently done on showEvent anyway, so this is not needed here
        tabs.addTab(self.visualize_tab, "Visualize results")

        self.visualize_tab.vis_shape_button.clicked.connect(lambda: self.run_shape_browser())

        self.visualize_tab.vis_optuna_button.clicked.connect(lambda: self.run_optuna_visualizations())

        layout.addWidget(tabs)

        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _apply_screen_aware_sizing(self) -> None:
        """Set window size with margins based on content."""
        # Set minimum size so window remains usable
        self.setMinimumSize(300, 200)

        # Get Qt's calculated size based on content
        hint = self.sizeHint()

        # Add margins for visual comfort
        margin = 25
        self.resize(hint.width() + margin, hint.height() + margin)

    def _on_native_dialog_toggled(self, checked: bool) -> None:
        """Save the native dialog preference."""
        self.controller.settings.use_native_dialog = checked

    def closeEvent(self, event) -> None:
        # GeometryPersistenceMixin.closeEvent saves geometry automatically
        super().closeEvent(event)

    def showEvent(self, event): # Override showEvent to call on_show each time the window is shown
        super().showEvent(event)
        self.on_show()

    def on_show(self) -> None:
        self.setup_tab.update_setup_info_text()  # Update the info text in the setup tab
        self.update_results_folder_path()  # Also triggers update_results_info_text via textChanged

    def update_results_folder_path(self) -> None:
        # Get the output folder path from settings
        results_folder_path = self.controller.settings.results_folder_path
        if not results_folder_path:
            logger.debug("No output folder path found in settings")
            results_folder_path = "None"

        # Setting .path triggers textChanged -> save_results_folder_path -> update_results_info_text
        self.visualize_tab.widgets['results_folder_path'].path = results_folder_path

    def run_config_editor(self) -> None:
        logger.info("Opening Config Editor")
        self.hide()

        # Use controller to show config window
        try:
            dialog = config_editor.ConfigWindow(
                self.controller.settings.optimization_config_file,
                self.controller.settings._settings,  # Pass QSettings for backwards compatibility
                #parent=self,
                parent=None # parent=None so config window gets its own taskbar button
            )
        except ConfigLoadError as e:
            # User declined to create a default config file - return to Main Menu
            logger.info(f"Config Editor closed: {e}")
            self.show()
            return

        dialog.exec()

        config_status = None

        if dialog.result() == QDialog.Accepted:
            # If the dialog was accepted, update the config path in settings
            self.controller.settings.optimization_config_file = dialog.config_file_path
            self.setup_tab.widgets['optimization_config_file'].path = str(dialog.config_file_path)
            config_status = f"Config file path set to: {dialog.config_file_path}"
            logger.info(f"Config file updated: {dialog.config_file_path}")

        # Always sync the Main Menu's config file to whatever was last loaded
        # in the Config Editor (e.g. after "Save as..." switched the file)
        previous_path = self.controller.settings.optimization_config_file
        current_path = dialog.config_file_path
        if str(current_path) != str(previous_path):
            self.controller.settings.optimization_config_file = current_path
            self.setup_tab.widgets['optimization_config_file'].path = str(current_path)
            config_status = f"Config file path set to: {current_path}"
            logger.info(f"Config file synced from Config Editor: {current_path}")
            QMessageBox.information(
                self, "Config File Updated",
                f"The selected config file has been updated to:\n"
                f"{Path(current_path).name}"
            )

        self.show()

        # Set status after show() so it isn't overwritten by showEvent -> on_show
        if config_status:
            self.status_label.setText(config_status)


    def run_optimization_script(self) -> None:
        logger.info("Starting optimization workflow")

        config_file_path = self.controller.settings.optimization_config_file

        if not config_file_path:
            QMessageBox.warning(self, "No Config", "Please select a configuration file first.")
            return

        # Use factory method which returns None on initialization failure
        output_window = optimization_console.OptimizationOutputWindow.create(
            config_file_path, parent=self
        )

        if output_window is None:
            # Initialization failed - error was already shown to user
            logger.error("Failed to create optimization output window")
            return

        # Store reference and show the window
        self.output_window = output_window
        self.output_window.show()

        # Connect to finished signal to update output folder after optimization
        self.output_window.finished.connect(self.on_optimization_finished)
        self.output_window.optimization_completed.connect(self._on_optimization_completed)
        
        # ------------------------------------------------------

    def _on_optimization_completed(self, results_folder_path: str) -> None:
        """Called when the optimization subprocess finishes successfully.
        Saves the new results folder path to settings and updates the UI."""
        logger.info(f"Optimization completed, updating results folder: {results_folder_path}")
        self.controller.settings.results_folder_path = results_folder_path
        self.update_results_folder_path()

    def on_optimization_finished(self) -> None:
        """Called when the optimization output window is closed."""
        self.update_results_folder_path()

    def run_shape_browser(self) -> None:
        logger.info("Launching shape browser")

        folder_path = self.controller.settings.results_folder_path
        if not folder_path:
            QMessageBox.warning(self, "No Output Folder", "Please select an output folder first.")
            return

        # Validate the folder contains the expected output files
        from bites.gui.visualization.shape_browser.utils import validate_folder_for_shape_browser
        problems = validate_folder_for_shape_browser(folder_path)
        if problems:
            bullet_list = "\n".join(f"  • {p}" for p in problems)
            QMessageBox.warning(
                self,
                "Invalid Output Folder",
                f"The selected folder does not look like a completed "
                f"optimization output folder:\n\n"
                f"{bullet_list}\n\n"
                f"Folder: {folder_path}\n\n"
                f"Please select a folder that contains the results of a "
                f"completed optimization run."
            )
            return

        self.hide()  # Hide main menu

        try:
            # Use controller to launch shape browser subprocess
            proc = self.controller.subprocess_mgr.launch_shape_browser(folder_path, wait=True)
            logger.info(f"Shape browser completed (return code: {proc.returncode})")

            # Check if subprocess failed and show error details
            if proc.returncode != 0:
                error_details = self._get_shape_browser_error_details()
                QMessageBox.warning(
                    self,
                    "Shape Browser Error",
                    f"Shape browser exited with code {proc.returncode}.\n\n"
                    f"{error_details}"
                )

        except Exception as e:
            logger.error(f"Failed to launch shape browser: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to run shape browser: {e}")
        finally:
            self.show()  # Show main menu again

    def _get_shape_browser_error_details(self) -> str:
        """Read error details from shape browser log files."""
        from bites.paths import get_logs_folder

        logs_folder = get_logs_folder()
        error_details = []

        # Check both log files
        log_files = [
            ("Subprocess output", logs_folder / "shape_browser_subprocess_output.log"),
            ("Subprocess errors", logs_folder / "shape_browser_subprocess.log"),
        ]

        for name, log_path in log_files:
            if log_path.exists():
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content:
                        # Only show last 1000 chars to avoid huge dialogs
                        if len(content) > 1000:
                            content = "..." + content[-1000:]
                        error_details.append(f"=== {name} ===\n{content}")
                except Exception as e:
                    logger.warning(f"Could not read {log_path}: {e}")

        if error_details:
            return "\n\n".join(error_details)
        else:
            return f"Check log files in:\n{logs_folder}"


    def run_optuna_visualizations(self) -> None:
        logger.info("Opening Optuna visualizations dialog")

        folder_path = self.controller.settings.results_folder_path
        if not folder_path:
            QMessageBox.warning(self, "No Output Folder", "Please select an output folder first.")
            return

        # Validate folder exists
        from pathlib import Path
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(
                self,
                "Invalid Folder",
                f"The output folder does not exist:\n{folder_path}"
            )
            return

        # Check for config file (basic validation)
        config_files = list(folder.glob("config_*.yaml"))
        if not config_files:
            QMessageBox.warning(
                self,
                "No Study Found",
                f"No optimization config file found in:\n{folder_path}\n\n"
                "Please select a folder containing optimization results."
            )
            return

        try:
            # Import and show the new dialog
            from bites.gui.visualization.optuna_plots.dialog import OptunaPlotDialog

            dialog = OptunaPlotDialog(
                output_folder=folder,
                parent=self
            )
            dialog.exec()

            logger.info("Optuna visualizations dialog closed")

        except Exception as e:
            logger.error(f"Failed to open Optuna visualizations dialog: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open visualizations dialog: {e}")

        self.visualize_tab.update_results_info_text()