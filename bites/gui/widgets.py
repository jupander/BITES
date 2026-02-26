"""
Reusable custom widgets for PySide6.

Provides:
- PathInputWidget: Text field with browse button for file/folder selection
- PathType: Enum for supported path types
"""

import logging
from enum import Enum, auto
from pathlib import Path

from PySide6.QtWidgets import (
    QLabel, QHBoxLayout, QVBoxLayout,
    QLineEdit, QPushButton, QFileDialog,
    QWidget, QSizePolicy
)

logger = logging.getLogger(__name__)

import bites.gui.gui_utils as gui_utils


class PathType(Enum):
    """Types of paths that can be selected via file dialog."""
    FOLDER = auto()
    FILE = auto()
    MESH = auto()
    YAML = auto()
    TXT = auto()


class PathInputWidget(QWidget):
    """
    Reusable widget for path input with browse button.

    Provides a text field with an optional label and a browse button that
    opens the appropriate file dialog based on the path type.

    Usage:
        # For folder selection
        path_widget = PathInputWidget(
            parent=self,
            path_type=PathType.FOLDER,
            label="Output Folder:",
            default_value="./output"
        )
        layout.addWidget(path_widget)

        # Get the current path
        current_path = path_widget.path

        # Connect to changes
        path_widget.line_edit.textChanged.connect(my_handler)
    """

    # File filters for different path types
    FILE_FILTERS = {
        PathType.MESH: "Mesh files (*.ply *.off *.stl *.obj *.glb *.gltf *.3mf)",
        PathType.YAML: "Config files (*.yaml)",
        PathType.TXT: "Text files (*.txt)",
        PathType.FILE: "All Files (*)",
    }

    # Dialog titles for different path types
    DIALOG_TITLES = {
        PathType.FOLDER: "Select folder",
        PathType.FILE: "Select file",
        PathType.MESH: "Select mesh file",
        PathType.YAML: "Select config file",
        PathType.TXT: "Select text file",
    }

    def __init__(
        self,
        parent: QWidget = None,
        path_type: PathType = PathType.FILE,
        label: str = None,
        default_value: str = "",
        use_relative_paths: bool = True,
        vertical_layout: bool = True,
    ):
        """
        Initialize PathInputWidget.

        Args:
            parent: Parent widget
            path_type: Type of path to select (folder, file, mesh, yaml)
            label: Optional label text (if None, no label is shown)
            default_value: Initial path value
            use_relative_paths: If True, convert selected paths to relative paths
            vertical_layout: If True, label is above input; if False, label is beside input
        """
        super().__init__(parent)

        self._path_type = path_type
        self._use_relative_paths = use_relative_paths
        self._parent_dialog = parent
        self._active_dialog = None  # prevent GC of non-blocking dialog

        # Create widgets
        self.label_widget = QLabel(label) if label else None
        self.line_edit = QLineEdit(default_value)
        self.line_edit.setMinimumWidth(50)  # Allow shrinking
        self.browse_button = QPushButton("Browse")
        self.browse_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Connect browse button
        self.browse_button.clicked.connect(self._on_browse_clicked)

        # Build layout
        if vertical_layout:
            self._build_vertical_layout()
        else:
            self._build_horizontal_layout()

    def _build_vertical_layout(self) -> None:
        """Build layout with label above the input field."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        if self.label_widget:
            main_layout.addWidget(self.label_widget)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.line_edit)
        input_layout.addWidget(self.browse_button)
        main_layout.addLayout(input_layout)

        self.setLayout(main_layout)

    def _build_horizontal_layout(self) -> None:
        """Build layout with label beside the input field."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        if self.label_widget:
            main_layout.addWidget(self.label_widget)

        main_layout.addWidget(self.line_edit)
        main_layout.addWidget(self.browse_button)

        self.setLayout(main_layout)

    def _get_default_directory(self) -> str:
        """Get the default directory for the file dialog.

        For file paths, opens the parent directory of the file.
        For folder paths, opens the parent directory so the user can
        easily pick a sibling folder (e.g. a different output run).
        If the stored path no longer exists, walks up to the nearest
        existing ancestor directory.
        """
        current_text = self.line_edit.text().strip()
        if current_text:
            try:
                path = Path(current_text).resolve()
            except OSError:
                return str(Path.cwd())
            # If it's an existing file, open its parent directory
            if path.is_file():
                return str(path.parent)
            # If it's an existing directory…
            if path.is_dir():
                # For folder selection, open the parent so the user can
                # pick a sibling folder instead of browsing inside
                if self._path_type == PathType.FOLDER:
                    parent = path.parent
                    if parent.is_dir():
                        return str(parent)
                return str(path)
            # Path doesn't exist — walk up to the nearest existing ancestor
            for ancestor in path.parents:
                if ancestor.is_dir():
                    return str(ancestor)
        return str(Path.cwd())

    def _on_browse_clicked(self) -> None:
        """Handle browse button click using a non-blocking dialog.

        Static QFileDialog methods (getOpenFileName, etc.) block the main
        thread for the entire lifetime of the native dialog — including the
        multi-second close/cleanup phase on Windows. Using an instance with
        open() keeps the event loop alive so the UI stays responsive.
        """
        dialog = QFileDialog(self._parent_dialog)
        dialog.setWindowTitle(self.DIALOG_TITLES.get(self._path_type, "Select"))
        dialog.setDirectory(self._get_default_directory())

        options = gui_utils.get_file_dialog_options()
        if options & QFileDialog.Option.DontUseNativeDialog:
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

        if self._path_type == PathType.FOLDER:
            dialog.setFileMode(QFileDialog.Directory)
            dialog.setOption(QFileDialog.ShowDirsOnly, True)
        else:
            dialog.setFileMode(QFileDialog.ExistingFile)
            file_filter = self.FILE_FILTERS.get(self._path_type, "All Files (*)")
            dialog.setNameFilter(file_filter)

        dialog.fileSelected.connect(self._on_file_selected)
        dialog.finished.connect(self._on_dialog_finished)

        # Store reference to prevent garbage collection while dialog is open
        self._active_dialog = dialog
        dialog.open()

    def _on_file_selected(self, selected_path: str) -> None:
        """Handle file selection from the non-blocking dialog."""
        if self._use_relative_paths:
            try:
                selected_path = str(Path(selected_path).relative_to(Path.cwd()))
            except ValueError:
                pass  # Keep absolute path if relative path not possible
        self.line_edit.setText(selected_path)

    def _on_dialog_finished(self) -> None:
        """Clean up dialog reference after it closes."""
        self._active_dialog = None

    @property
    def path(self) -> str:
        """Get the current path value."""
        return self.line_edit.text()

    @path.setter
    def path(self, value: str) -> None:
        """Set the path value."""
        self.line_edit.setText(value)

    def setDisabled(self, disabled: bool) -> None:
        """Enable or disable the widget."""
        super().setDisabled(disabled)
        self.line_edit.setDisabled(disabled)
        self.browse_button.setDisabled(disabled)
        if self.label_widget:
            self.label_widget.setDisabled(disabled)

    def setToolTip(self, tooltip: str) -> None:
        """Set tooltip for the line edit."""
        self.line_edit.setToolTip(tooltip)
