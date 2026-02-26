"""
Visualize Tab - Results visualization and analysis interface.

This module provides the Visualize tab for the main application window,
allowing users to browse and analyze optimization results.
"""

import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QSizePolicy, QApplication
)

from bites.gui.widgets import PathInputWidget, PathType

# Import for folder analysis
from bites.gui.analyze_output_folder import FolderAnalysisWorker

if TYPE_CHECKING:
    from bites.core.application_controller import ApplicationController

logger = logging.getLogger(__name__)


class VisualizeTab(QWidget):
    """
    Visualize tab for results browsing and analysis.

    This tab provides:
    - Output folder selection and viewing
    - Shape browser launch
    - Optuna visualization generation
    - Output folder analysis
    """

    def __init__(self, controller: 'ApplicationController'):
        """
        Initialize the visualize tab.

        Args:
            controller: Application controller instance
        """
        super().__init__()
        self.controller = controller
        self.widgets = {}  # Store widgets for later access
        self._analysis_worker = None  # Track current analysis worker
        self._is_busy = False

        logger.debug("Initializing VisualizeTab")
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Row 1: Output folder path input
        self.widgets['results_folder_path'] = PathInputWidget(
            parent=self,
            path_type=PathType.FOLDER,
            label="Selected output folder:",
            default_value="placeholder_path",  # Updated by parent window
            vertical_layout=False,
        )
        self.widgets['results_folder_path'].line_edit.setReadOnly(True)  # Text field read-only
        # Apply minimal stylesheet to prevent Windows "OpenThemeData() failed" error on hover
        self.widgets['results_folder_path'].line_edit.setStyleSheet(
            "QLineEdit { background-color: palette(window); }"
        )
        self.widgets['results_folder_path'].setToolTip(
            'The folder containing optimization results to visualize.\n\n'
            'This folder is automatically set after running an optimization.\n'
            'Use "Browse" to select a different output folder.\n\n'
            'An output folder typically contains:\n'
            '• Aligned mesh files (.off) from each trial\n'
            '• Parameter value files (.txt) for each trial\n'
            '• A database copy file (.db) with trial history\n'
            '• A CSV summary of all trials'
        )
        self.widgets['results_folder_path'].line_edit.textChanged.connect(
            lambda text: self.save_results_folder_path(text)
        )
        layout.addWidget(self.widgets['results_folder_path'])

        # Row 2: Action buttons (allow shrinking to text width)
        self.vis_shape_button = QPushButton('Shape browser', self)
        self.vis_shape_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.vis_shape_button.setToolTip(
            'Opens an interactive 3D viewer to browse optimization results.\n\n'
            'The shape browser allows you to:\n'
            '• View 3D meshes of simulated teeth from all trials\n'
            '• Compare different results side by side with the target\n'
            '• See parameter values and loss scores for each trial\n'
            '• Navigate through trials sorted by loss or other properties'
        )
        self.vis_optuna_button = QPushButton('Generate Optuna plots', self)
        self.vis_optuna_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.vis_optuna_button.setToolTip(
            'Generate statistical visualization plots from the optimization study.\n\n'
            'Creates interactive HTML plots showing:\n'
            '• Optimization history: How scores improved over trials\n'
            '• Parameter importance: Which parameters most affect the results\n'
            '• Parameter relationships: Correlations between parameters\n'
            '• Slice plots: How each parameter affects the score\n\n'
            'Plots are saved as HTML files in the output folder\n'
            'and automatically opened in your web browser.'
        )
        self.open_output_folder_button = QPushButton('Open output folder', self)
        self.open_output_folder_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.open_output_folder_button.setToolTip(
            'Open the selected output folder in your system file explorer.'
        )

        # Open folder in file explorer
        self.open_output_folder_button.clicked.connect(self._open_output_folder)

        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.vis_shape_button)
        horizontal_layout.addWidget(self.vis_optuna_button)
        horizontal_layout.addWidget(self.open_output_folder_button)
        layout.addLayout(horizontal_layout)

        # Row 3: Results info display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFrameStyle(QTextEdit.NoFrame)
        self.text_display.setStyleSheet("background-color: palette(window);")  # Use system color instead of transparent
        self.text_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.text_display.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevent focus stealing
        self.text_display.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)  # Disable selection

        self.update_results_info_text()  # Load initial info
        layout.addWidget(self.text_display)

        self.setLayout(layout)

    def _open_output_folder(self) -> None:
        """Open the output folder in the system file explorer."""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        folder = self.controller.settings.results_folder_path or os.getcwd()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def save_results_folder_path(self, path: str) -> None:
        """
        Save output folder path to settings.

        Args:
            path: Output folder path
        """
        self.controller.settings.results_folder_path = path
        self.update_results_info_text()

    def update_results_info_text(self) -> None:
        """Update the results info display text using background worker."""
        results_folder_path = self.controller.settings.results_folder_path

        if not results_folder_path:
            logger.debug("No output folder path in settings")
            self.text_display.setPlainText("No valid output folder path set.")
            self._cancel_pending_work()
            self._set_busy(False)
            return

        results_path = Path(results_folder_path)
        if not results_path.exists() or not results_path.is_dir():
            self.text_display.setPlainText("No valid output folder path set.")
            self._cancel_pending_work()
            self._set_busy(False)
            return

        self._cancel_pending_work()

        # Show loading state and busy cursor
        self._set_busy(True)
        self.text_display.setPlainText("Analyzing folder...")

        self._analysis_worker = FolderAnalysisWorker(results_folder_path, parent=self)
        self._analysis_worker.finished_analysis.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.start()

        logger.debug(f"Started background analysis (results_folder={results_folder_path})")

    def _cancel_pending_work(self) -> None:
        """Cancel any running worker."""
        if self._analysis_worker is not None:
            # Cancel first to prevent signal emission during disconnect
            self._analysis_worker.cancel()
            try:
                self._analysis_worker.finished_analysis.disconnect()
                self._analysis_worker.error.disconnect()
            except RuntimeError:
                pass  # Already disconnected
            self._analysis_worker = None

    def _on_analysis_finished(self, result_text: str) -> None:
        """Handle completed folder analysis."""
        self.text_display.setPlainText(result_text)
        self._analysis_worker = None
        self._set_busy(False)

    def _on_analysis_error(self, error_message: str) -> None:
        """Handle folder analysis error."""
        self.text_display.setPlainText(f"Error analyzing folder: {error_message}")
        self._analysis_worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        """Show/hide busy cursor and enable/disable buttons."""
        if busy and not self._is_busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._is_busy = True
        elif not busy and self._is_busy:
            QApplication.restoreOverrideCursor()
            self._is_busy = False
        self.vis_shape_button.setDisabled(busy)
        self.vis_optuna_button.setDisabled(busy)
        self.open_output_folder_button.setDisabled(busy)
        self.widgets['results_folder_path'].browse_button.setDisabled(busy)
