"""
Reusable mixin classes for PySide6 windows and dialogs.

Provides:
- GeometryPersistenceMixin: Save/restore window position and size across sessions
- DirtyTrackingMixin: Track unsaved changes and prompt before closing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bites.core.settings_manager import SettingsManager

from PySide6.QtWidgets import (
    QLineEdit, QSpinBox, QCheckBox, QRadioButton, QSlider, QMessageBox
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dirty State Tracking Mixin
# -----------------------------------------------------------------------------

class DirtyTrackingMixin:
    """
    Mixin class to add dirty state tracking to QDialog subclasses.

    Usage:
        1. Add this mixin to your dialog class: class MyDialog(DirtyTrackingMixin, QDialog)
        2. Call _init_dirty_tracking() at the end of __init__
        3. Pass widget dicts to _connect_dirty_signals() to track changes
        4. Override closeEvent() to call _prompt_unsaved_changes()
    """

    _is_dirty: bool = False

    def _init_dirty_tracking(self) -> None:
        """Initialize dirty tracking state. Call at end of __init__."""
        self._is_dirty = False

    def _mark_dirty(self) -> None:
        """Mark the dialog as having unsaved changes."""
        self._is_dirty = True

    def _reset_dirty(self) -> None:
        """Reset dirty state after successful save."""
        self._is_dirty = False

    def _connect_dirty_signals(self, *widget_collections) -> None:
        """
        Connect change signals from widgets to mark the dialog as dirty.

        Args:
            *widget_collections: Dicts or lists of widgets to track
        """
        for collection in widget_collections:
            if isinstance(collection, dict):
                widgets = collection.values()
            elif isinstance(collection, (list, tuple)):
                widgets = collection
            else:
                widgets = [collection]

            for widget in widgets:
                self._connect_single_widget(widget)

    def _connect_single_widget(self, widget) -> None:
        """Connect a single widget's change signal to _mark_dirty."""
        # Import here to avoid circular imports with widgets module
        from bites.gui.widgets import PathInputWidget

        try:
            if isinstance(widget, PathInputWidget):
                widget.line_edit.textChanged.connect(self._mark_dirty)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._mark_dirty)
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._mark_dirty)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self._mark_dirty)
            elif isinstance(widget, QRadioButton):
                widget.toggled.connect(self._mark_dirty)
            elif isinstance(widget, QSlider):
                widget.valueChanged.connect(self._mark_dirty)
            elif isinstance(widget, dict):
                # Nested dict of widgets (e.g., radio button groups)
                for sub_widget in widget.values():
                    self._connect_single_widget(sub_widget)
        except Exception as e:
            logger.debug(f"Could not connect dirty tracking for widget {type(widget)}: {e}")

    def _prompt_unsaved_changes(self, event) -> bool:
        """
        Show prompt if there are unsaved changes.

        Args:
            event: The close event

        Returns:
            True if the dialog should close, False to cancel
        """
        if not self._is_dirty:
            return True

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Do you want to discard them?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )

        if reply == QMessageBox.StandardButton.Discard:
            return True
        else:
            event.ignore()
            return False


# -----------------------------------------------------------------------------
# Geometry Persistence Mixin
# -----------------------------------------------------------------------------

class GeometryPersistenceMixin:
    """
    Mixin class to add geometry persistence to QWidget/QDialog/QMainWindow subclasses.

    Usage:
        1. Add this mixin to your window class: class MyDialog(GeometryPersistenceMixin, QDialog)
        2. Define GEOMETRY_KEY class attribute: GEOMETRY_KEY = "my_dialog"
        3. Set DEFAULT_GEOMETRY = None to let Qt auto-size based on content
        4. Call self.adjustSize() after building UI to calculate optimal initial size
        5. Call _init_geometry_persistence(settings_manager) at the end of __init__
        6. Geometry is automatically saved in closeEvent (if you override, call super())

    For QMainWindow subclasses, window state (toolbar/dock positions) is also saved.

    Note: When DEFAULT_GEOMETRY is None (recommended), the window will use Qt's layout
    system to calculate the minimum size needed to fit all content on first launch.
    Saved geometry from previous sessions takes precedence over auto-sizing.
    """

    # Subclasses MUST define this - used to construct the settings key
    GEOMETRY_KEY: str = None  # e.g., "config_editor", "mesh_cut_dialog"

    # Subclasses CAN define this for fallback - (x, y, width, height)
    DEFAULT_GEOMETRY: tuple = None

    # Internal state
    _geometry_settings: 'SettingsManager' = None
    _geometry_initialized: bool = False

    def _init_geometry_persistence(self, settings_manager: 'SettingsManager' = None) -> None:
        """
        Initialize geometry persistence. Call at end of __init__.

        Args:
            settings_manager: SettingsManager instance. If None, creates a new one.
        """
        if self.GEOMETRY_KEY is None:
            raise ValueError(
                f"{self.__class__.__name__} must define GEOMETRY_KEY class attribute"
            )

        if settings_manager is None:
            from bites.core.settings_manager import SettingsManager
            settings_manager = SettingsManager()

        self._geometry_settings = settings_manager
        self._restore_geometry()
        self._geometry_initialized = True

    def _get_geometry_key(self) -> str:
        """Get the standardized settings key for this window's geometry."""
        return f"window/{self.GEOMETRY_KEY}/geometry"

    def _get_state_key(self) -> str:
        """Get the standardized settings key for this window's state (QMainWindow only)."""
        return f"window/{self.GEOMETRY_KEY}/state"

    def _restore_geometry(self) -> None:
        """Restore window geometry from settings, falling back to defaults."""
        from PySide6.QtWidgets import QMainWindow
        from PySide6.QtCore import QByteArray

        # Try to restore saved geometry (stored as QByteArray, not string)
        geometry = self._geometry_settings.get_value(self._get_geometry_key(), value_type=QByteArray)
        if geometry:
            self.restoreGeometry(geometry)
        elif self.DEFAULT_GEOMETRY is not None:
            x, y, w, h = self.DEFAULT_GEOMETRY
            self.setGeometry(x, y, w, h)

        # For QMainWindow, also restore state (toolbar/dock positions)
        if isinstance(self, QMainWindow):
            state = self._geometry_settings.get_value(self._get_state_key(), value_type=QByteArray)
            if state:
                self.restoreState(state)

    def _save_geometry(self) -> None:
        """Save window geometry (and state for QMainWindow) to settings."""
        from PySide6.QtWidgets import QMainWindow

        if not self._geometry_initialized or self._geometry_settings is None:
            return

        self._geometry_settings.set_value(self._get_geometry_key(), self.saveGeometry())

        # For QMainWindow, also save state
        if isinstance(self, QMainWindow):
            self._geometry_settings.set_value(self._get_state_key(), self.saveState())

    def closeEvent(self, event) -> None:
        """
        Override closeEvent to save geometry.

        If your subclass overrides closeEvent, call super().closeEvent(event)
        to ensure geometry is saved.
        """
        self._save_geometry()
        super().closeEvent(event)
