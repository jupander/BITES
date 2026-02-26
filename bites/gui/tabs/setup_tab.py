"""
Setup Tab - Configuration and optimization setup interface.

This module provides the Setup tab for the main application window,
allowing users to configure and launch optimization runs.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QSizePolicy
)

from bites.config.config_loader import Config
from bites.gui.widgets import PathInputWidget, PathType

if TYPE_CHECKING:
    from bites.core.application_controller import ApplicationController

logger = logging.getLogger(__name__)


class SetupTab(QWidget):
    """
    Setup tab for configuration and optimization launch.

    This tab provides:
    - Config file selection and viewing
    - Optimization launch button
    - Quick view of config parameters
    """

    def __init__(self, controller: 'ApplicationController'):
        """
        Initialize the setup tab.

        Args:
            controller: Application controller instance
        """
        super().__init__()
        self.controller = controller
        self.widgets = {}  # Store widgets for later access

        logger.debug("Initializing SetupTab")
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Buttons (allow shrinking to text width)
        self.config_setup_button = QPushButton('Edit loaded config file', self)
        self.config_setup_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.config_setup_button.setToolTip(
            'Opens the Config Editor to modify optimization settings.\n\n'
            'Changes are saved to the configuration file when you click Save.'
        )
        self.start_new_button = QPushButton('Start optimization', self)
        self.start_new_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.start_new_button.setToolTip(
            "Launch an optimization run using the currently loaded config file.\n\n"
        )

        # Get config path from controller settings
        config_path = self.controller.settings.optimization_config_file

        # Row 1: Config file path input
        self.widgets['optimization_config_file'] = PathInputWidget(
            parent=self,
            path_type=PathType.YAML,
            label="Currently loaded config file:",
            default_value=config_path if config_path else "None selected",
            vertical_layout=False,
        )
        self.widgets['optimization_config_file'].line_edit.setReadOnly(True)  # Text field read-only
        # Apply minimal stylesheet to prevent Windows "OpenThemeData() failed" error on hover
        self.widgets['optimization_config_file'].line_edit.setStyleSheet(
            "QLineEdit { background-color: palette(window); }"
        )
        self.widgets['optimization_config_file'].setToolTip(
            'The configuration file that defines all optimization settings.\n\n'
            'Use the "Browse" button to select a different config file,\n'
            'or "Edit loaded config file" to edit the current one.'
        )
        self.widgets['optimization_config_file'].line_edit.textChanged.connect(
            lambda text: self.save_config_path(text)
        )
        layout.addWidget(self.widgets['optimization_config_file'])

        # Row 2: Action buttons
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.config_setup_button)
        horizontal_layout.addWidget(self.start_new_button)
        layout.addLayout(horizontal_layout)

        # Row 3: Config info display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFrameStyle(QTextEdit.NoFrame)
        self.text_display.setStyleSheet("background-color: palette(window);")  # Use system color instead of transparent
        self.text_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.text_display.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Prevent focus stealing
        self.text_display.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)  # Disable selection
        self.text_display.setToolTip(
            'Brief summary of the current configuration settings.'
            'This shows key information from your config file:\n'
            '• Study name: Identifies the optimization study\n'
            '• Target: The mesh you\'re trying to match\n'
            '• Search space dimensions: Number of parameters being optimized\n'
            '• Estimated duration: Worst-case runtime estimate\n\n'
            'Click "Edit loaded config file" to change these settings.'
        )

        self.update_setup_info_text()  # Load initial info
        layout.addWidget(self.text_display)

        self.setLayout(layout)

    @staticmethod
    def get_info_text_from_config_file(config_file_path: str) -> str:
        """
        Read and format information from a config file.

        Args:
            config_file_path: Path to the config file

        Returns:
            str: Formatted config information
        """
        config_path = Path(config_file_path)
        if not config_path.exists() or not config_path.is_file():
            return f"Config file not found at the specified path:\n{config_file_path}"

        try:
            config = Config.from_yaml(config_file_path)
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            return f"Error reading config file: {e}"

        # Format config information
        text = (
            f"Study:\n{config.study_name}\n\n" +
            f"Target:\n{Path(config.target_shape_path).name}\n\n" +
            f"Search space dimensions:\n{len(config.params_to_optimize)-(1 if config.bias_param_config.Bbi_value_for_Lbi_too else 0)-(1 if config.bias_param_config.Abi_value_for_Pbi_too else 0)+(0 if config.min_iterations == config.max_iterations else 1)}\n\n" +
            f"Estimated max duration:\n{config.max_trials} trials * {config.tm_timeout/60:.1f} mins / {config.parallel_trials} cores -> ~{config.max_trials * config.tm_timeout / 3600 / config.parallel_trials:.2f} hours (timeout {config.study_timeout} hours)\n\n"
        )

        return text

    def save_config_path(self, path: str) -> None:
        """
        Save config path to settings.

        Args:
            path: Config file path
        """
        self.controller.settings.optimization_config_file = path
        self.update_setup_info_text()

    def update_setup_info_text(self) -> None:
        """Update the config info display text (synchronous — takes ~12ms)."""
        config_path_str = self.controller.settings.optimization_config_file
        config_path = Path(config_path_str) if config_path_str else None

        if config_path and config_path.exists():
            text = self.get_info_text_from_config_file(config_path_str)
        else:
            text = "No valid config file loaded."

        self.text_display.setPlainText(text)
