"""
Settings Manager - Centralized application settings management.

This module provides a type-safe abstraction over QSettings, ensuring consistent
access to application configuration and user preferences.
"""

import logging
from pathlib import Path
from typing import Optional, Any
from PySide6.QtCore import QSettings, QByteArray

from bites.constants import DEFAULT_EXAMPLE_CONFIG_PATH, DEFAULT_EXAMPLE_RESULTS_FOLDER

logger = logging.getLogger(__name__)


class SettingsManager:
    """
    Centralized settings management with type-safe property access.

    This class wraps QSettings to provide:
    - Type-safe getters and setters
    - Default value management
    - Validation
    - Logging of setting changes

    Example:
        settings = SettingsManager()
        settings.optimization_config_file = "/path/to/config.yaml"
        config_path = settings.optimization_config_file
    """

    # Organization and application name for QSettings
    ORGANIZATION = "DiPoiLab"
    APPLICATION = "BITES"

    # Setting keys (centralized for easy reference)
    KEY_OPTIMIZATION_CONFIG = "optimization_config_file"
    KEY_RESULTS_FOLDER = "results_folder_path"
    KEY_USE_NATIVE_DIALOG = "use_native_dialog"
    KEY_WINDOW_GEOMETRY = "geometry"
    KEY_WINDOW_STATE = "windowState"

    def __init__(self):
        """Initialize the settings manager."""
        self._settings = QSettings(self.ORGANIZATION, self.APPLICATION)
        logger.info(f"Initialized SettingsManager (organization={self.ORGANIZATION}, app={self.APPLICATION})")
        logger.debug(f"Settings file location: {self._settings.fileName()}")

    # -------------------------------------------------------------------------
    # Optimization Configuration
    # -------------------------------------------------------------------------

    @property
    def optimization_config_file(self) -> Optional[str]:
        """
        Get the path to the optimization configuration file.

        Returns the stored config path if valid, falls back to the default
        example config if it exists, or None if no valid config is available.

        Returns:
            Optional[str]: Path to config file, or None if not available
        """
        value = self._settings.value(self.KEY_OPTIMIZATION_CONFIG, None, type=str)

        # If we have a stored value and the file exists, use it
        if value and Path(value).exists():
            return value

        # Try fallback to default example config
        fallback_path = Path(DEFAULT_EXAMPLE_CONFIG_PATH)
        if fallback_path.exists():
            logger.info(f"Using fallback config path: {DEFAULT_EXAMPLE_CONFIG_PATH}")
            return str(fallback_path)

        # No valid config available
        return None

    @optimization_config_file.setter
    def optimization_config_file(self, value: Optional[str]) -> None:
        """
        Set the path to the optimization configuration file.

        Args:
            value: Path to config file (or None to clear)
        """
        if value is not None:
            # Validate path exists (optional - can be removed if you want to allow non-existent paths)
            path = Path(value)
            if not path.exists():
                logger.warning(f"Config file path does not exist: {value}")

        self._settings.setValue(self.KEY_OPTIMIZATION_CONFIG, value)
        self._settings.sync()
        logger.info(f"Set optimization config file: {value}")

    # -------------------------------------------------------------------------
    # Output Folder
    # -------------------------------------------------------------------------

    @property
    def results_folder_path(self) -> Optional[str]:
        """
        Get the path to the output folder.

        Returns the stored output folder if valid, falls back to the default
        example output folder if it exists, or None if no valid folder is available.

        Returns:
            Optional[str]: Path to output folder, or None if not available
        """
        value = self._settings.value(self.KEY_RESULTS_FOLDER, None, type=str)

        # If we have a stored value and the folder exists, use it
        if value and Path(value).exists():
            return value

        # Try fallback to default example output folder
        fallback_path = Path(DEFAULT_EXAMPLE_RESULTS_FOLDER)
        if fallback_path.exists():
            logger.info(f"Using fallback output folder: {DEFAULT_EXAMPLE_RESULTS_FOLDER}")
            return str(fallback_path)

        # No valid output folder available
        return None

    @results_folder_path.setter
    def results_folder_path(self, value: Optional[str]) -> None:
        """
        Set the path to the output folder.

        Args:
            value: Path to output folder (or None to clear)
        """
        if value is not None:
            # Validate path exists
            path = Path(value)
            if not path.exists():
                logger.warning(f"Output folder path does not exist: {value}")

        self._settings.setValue(self.KEY_RESULTS_FOLDER, value)
        self._settings.sync()
        logger.info(f"Set output folder path: {value}")

    # -------------------------------------------------------------------------
    # File Dialog Preference
    # -------------------------------------------------------------------------

    @property
    def use_native_dialog(self) -> bool:
        """Whether to use the OS native file dialog (True) or Qt's built-in dialog (False)."""
        return self._settings.value(self.KEY_USE_NATIVE_DIALOG, True, type=bool)

    @use_native_dialog.setter
    def use_native_dialog(self, value: bool) -> None:
        self._settings.setValue(self.KEY_USE_NATIVE_DIALOG, value)
        self._settings.sync()
        logger.info(f"Set use_native_dialog: {value}")

    # -------------------------------------------------------------------------
    # Window Geometry and State
    # -------------------------------------------------------------------------

    def get_window_geometry(self) -> Optional[QByteArray]:
        """
        Get saved window geometry.

        Returns:
            Optional[QByteArray]: Saved geometry, or None if not set
        """
        return self._settings.value(self.KEY_WINDOW_GEOMETRY)

    def set_window_geometry(self, geometry: QByteArray) -> None:
        """
        Save window geometry.

        Args:
            geometry: QByteArray from QWidget.saveGeometry()
        """
        self._settings.setValue(self.KEY_WINDOW_GEOMETRY, geometry)
        self._settings.sync()
        logger.debug("Saved window geometry")

    def get_window_state(self) -> Optional[QByteArray]:
        """
        Get saved window state.

        Returns:
            Optional[QByteArray]: Saved state, or None if not set
        """
        return self._settings.value(self.KEY_WINDOW_STATE)

    def set_window_state(self, state: QByteArray) -> None:
        """
        Save window state.

        Args:
            state: QByteArray from QMainWindow.saveState()
        """
        self._settings.setValue(self.KEY_WINDOW_STATE, state)
        self._settings.sync()
        logger.debug("Saved window state")

    # -------------------------------------------------------------------------
    # Generic Access (for backward compatibility or custom settings)
    # -------------------------------------------------------------------------

    def get_value(self, key: str, default: Any = None, value_type: type = str) -> Any:
        """
        Get a setting value by key (generic access).

        Args:
            key: Setting key
            default: Default value if not set
            value_type: Expected type of the value

        Returns:
            The setting value or default
        """
        return self._settings.value(key, default, type=value_type)

    def set_value(self, key: str, value: Any) -> None:
        """
        Set a setting value by key (generic access).

        Args:
            key: Setting key
            value: Value to set
        """
        self._settings.setValue(key, value)
        self._settings.sync()
        logger.debug(f"Set setting '{key}' = {value}")

    def contains(self, key: str) -> bool:
        """
        Check if a setting exists.

        Args:
            key: Setting key

        Returns:
            bool: True if setting exists
        """
        return self._settings.contains(key)

    def remove(self, key: str) -> None:
        """
        Remove a setting.

        Args:
            key: Setting key
        """
        self._settings.remove(key)
        logger.debug(f"Removed setting '{key}'")

    def clear_all(self) -> None:
        """
        Clear all settings (use with caution!).
        """
        logger.warning("Clearing all application settings!")
        self._settings.clear()

    def sync(self) -> None:
        """
        Force synchronization of settings to persistent storage.
        """
        self._settings.sync()
        logger.debug("Synchronized settings to disk")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_optimization_config(self) -> bool:
        """
        Validate that the optimization config file exists and is readable.

        Returns:
            bool: True if config is valid
        """
        config_path = self.optimization_config_file
        if config_path is None:
            logger.warning("No optimization config file set")
            return False

        path = Path(config_path)
        if not path.exists():
            logger.error(f"Optimization config file does not exist: {config_path}")
            return False

        if not path.is_file():
            logger.error(f"Optimization config path is not a file: {config_path}")
            return False

        # Could add additional validation here (e.g., YAML parsing)
        logger.debug(f"Optimization config file is valid: {config_path}")
        return True

    def validate_results_folder(self) -> bool:
        """
        Validate that the output folder exists and is a directory.

        Returns:
            bool: True if output folder is valid
        """
        folder_path = self.results_folder_path
        if folder_path is None:
            logger.warning("No output folder set")
            return False

        path = Path(folder_path)
        if not path.exists():
            logger.error(f"Output folder does not exist: {folder_path}")
            return False

        if not path.is_dir():
            logger.error(f"Output path is not a directory: {folder_path}")
            return False

        logger.debug(f"Output folder is valid: {folder_path}")
        return True

    # -------------------------------------------------------------------------
    # Geometry Migration
    # -------------------------------------------------------------------------

    # Migration mappings: old_key -> new GEOMETRY_KEY suffix
    _GEOMETRY_MIGRATION_MAP = {
        "geometry": "main_window",
        "geometry_configeditor": "config_editor",
        "optimization_output_window/geometry": "optimization_console",
    }

    def migrate_geometry_keys(self) -> None:
        """
        Migrate old geometry keys to new standardized format.

        Old keys used inconsistent patterns like:
        - "geometry" (main window)
        - "geometry_configeditor"
        - "optimization_output_window/geometry"

        New format: "window/<name>/geometry" and "window/<name>/state"

        Call this once during application startup.
        """
        migrated_any = False

        for old_key, new_suffix in self._GEOMETRY_MIGRATION_MAP.items():
            new_geometry_key = f"window/{new_suffix}/geometry"
            if self.contains(old_key) and not self.contains(new_geometry_key):
                value = self._settings.value(old_key)
                if value:
                    self._settings.setValue(new_geometry_key, value)
                    logger.info(f"Migrated geometry: {old_key} -> {new_geometry_key}")
                    migrated_any = True

        # Special handling for windowState -> window/main_window/state
        if self.contains("windowState") and not self.contains("window/main_window/state"):
            value = self._settings.value("windowState")
            if value:
                self._settings.setValue("window/main_window/state", value)
                logger.info("Migrated window state: windowState -> window/main_window/state")
                migrated_any = True

        if migrated_any:
            self.sync()
            logger.info("Geometry key migration complete")

    # -------------------------------------------------------------------------
    # Debug/Info Methods
    # -------------------------------------------------------------------------

    def get_all_keys(self) -> list[str]:
        """
        Get all setting keys (useful for debugging).

        Returns:
            list[str]: List of all setting keys
        """
        return self._settings.allKeys()

    def print_all_settings(self) -> None:
        """
        Print all settings to the logger (for debugging).
        """
        logger.info("=== Current Settings ===")
        for key in self.get_all_keys():
            value = self._settings.value(key)
            logger.info(f"  {key} = {value}")
        logger.info("========================")
