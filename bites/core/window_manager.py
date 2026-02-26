"""
Window Manager - Centralized window and dialog creation and lifecycle management.

This module manages all GUI windows and dialogs in the application, providing:
- Consistent window creation patterns
- Window lifecycle tracking
- Centralized initialization
- Proper cleanup
"""

import logging
from typing import Optional, List, TYPE_CHECKING
from PySide6.QtWidgets import QWidget, QDialog

if TYPE_CHECKING:
    from .application_controller import ApplicationController

logger = logging.getLogger(__name__)


class WindowManager:
    """
    Centralized manager for window and dialog creation.

    This class provides factory methods for creating all windows and dialogs
    in the application, ensuring consistent initialization and tracking.

    Features:
    - Consistent window creation patterns
    - Tracking of open windows
    - Centralized initialization logic
    - Logging of window lifecycle events
    """

    def __init__(self, app_controller: 'ApplicationController'):
        """
        Initialize the window manager.

        Args:
            app_controller: Reference to the application controller
        """
        self.app_controller = app_controller
        self._open_windows: List[QWidget] = []
        logger.info("Initialized WindowManager")

    # -------------------------------------------------------------------------
    # Main Application Window
    # -------------------------------------------------------------------------

    def create_main_window(self) -> None:
        """
        Create and return the main application window.

        Returns:
            MainMenuWindow: The main application window

        Note:
            Import is done locally to avoid circular dependencies
        """
        logger.info("Creating main application window...")

        # Local import to avoid circular dependency
        from bites.gui.main_menu import MainMenuWindow
        


        try:
            window = MainMenuWindow(controller=self.app_controller)
            self._track_window(window)
            logger.info("Main window created successfully")
            return window
        except Exception as e:
            logger.error(f"Failed to create main window: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Configuration Window
    # -------------------------------------------------------------------------

    def show_config_window(
        self,
        config_path: Optional[str] = None,
        parent: Optional[QWidget] = None
    ) -> QDialog:
        """
        Create and show the configuration window.

        Args:
            config_path: Path to existing config file (optional)
            parent: Parent widget (optional)

        Returns:
            ConfigWindow: The configuration dialog
        """
        logger.info(f"Creating config window (config_path={config_path})...")

        # Local import to avoid circular dependency
        import bites.gui.windows.config_editor as config_editor

        try:
            # Get current config path from settings if not provided
            if config_path is None:
                config_path = self.app_controller.settings.optimization_config_file

            dialog = config_editor.ConfigWindow(
                config_path,
                self.app_controller.settings._settings,  # Pass QSettings for backwards compatibility
                parent=parent
            )

            self._track_window(dialog)
            logger.info("Config window created successfully")
            return dialog

        except Exception as e:
            logger.error(f"Failed to create config window: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Optimization Output Window
    # -------------------------------------------------------------------------

    def show_optimization_output_window(
        self,
        config_file_path: str,
        parent: Optional[QWidget] = None
    ) -> Optional[QWidget]:
        """
        Create and show the optimization output window.

        Args:
            config_file_path: Path to the optimization config file
            parent: Parent widget (optional)

        Returns:
            OptimizationOutputWindow: The output window, or None if creation failed
        """
        logger.info(f"Creating optimization output window (config={config_file_path})...")

        # Local import to avoid circular dependency
        import bites.gui.windows.optimization_console as optimization_console

        try:
            # Use factory method which handles initialization errors gracefully
            window = optimization_console.OptimizationOutputWindow.create(
                config_file_path,
                parent=parent
            )

            if window is None:
                logger.error("Optimization output window initialization failed")
                return None

            self._track_window(window)
            logger.info("Optimization output window created successfully")
            return window

        except Exception as e:
            logger.error(f"Failed to create optimization output window: {e}", exc_info=True)
            return None

    # -------------------------------------------------------------------------
    # Cut Tester Dialog
    # -------------------------------------------------------------------------

    def show_cut_tester(
        self,
        mesh_path: str,
        default_cut_height: float,
        parent: Optional[QWidget] = None
    ) -> QDialog:
        """
        Create and show the mesh cut tester dialog.

        Args:
            mesh_path: Path to the mesh file
            default_cut_height: Default cut height value
            parent: Parent widget (optional)

        Returns:
            MeshCutDialog: The cut tester dialog
        """
        logger.info(f"Creating cut tester dialog (mesh={mesh_path}, height={default_cut_height})...")

        # Local import to avoid circular dependency
        import bites.gui.windows.cut_tester as cut_tester

        try:
            dialog = cut_tester.MeshCutDialog(
                mesh_path,
                default_cut_height,
                parent=parent
            )

            self._track_window(dialog)
            logger.info("Cut tester dialog created successfully")
            return dialog

        except Exception as e:
            logger.error(f"Failed to create cut tester dialog: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Simplifier Dialog
    # -------------------------------------------------------------------------

    def show_simplifier(
        self,
        mesh_path: str,
        parent: Optional[QWidget] = None
    ) -> QDialog:
        """
        Create and show the mesh simplifier dialog.

        Args:
            mesh_path: Path to the mesh file
            parent: Parent widget (optional)

        Returns:
            MeshSimplifierDialog: The simplifier dialog
        """
        logger.info(f"Creating simplifier dialog (mesh={mesh_path})...")

        # Local import to avoid circular dependency
        import bites.gui.windows.simplifier as simplifier

        try:
            dialog = simplifier.MeshSimplifierDialog(
                mesh_path,
                parent=parent
            )

            self._track_window(dialog)
            logger.info("Simplifier dialog created successfully")
            return dialog

        except Exception as e:
            logger.error(f"Failed to create simplifier dialog: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Optuna Visualizations
    # -------------------------------------------------------------------------

    def run_optuna_visualizations(self, folder_path: str) -> None:
        """
        Run Optuna visualizations (blocking operation).

        Args:
            folder_path: Path to the output folder

        Note:
            This is a blocking operation that generates plots
        """
        logger.info(f"Running Optuna visualizations (folder={folder_path})...")

        # Local import to avoid circular dependency
        from bites.gui.visualization.optuna_plots import visuals as optuna_visuals
        from pathlib import Path

        try:
            optuna_visuals.main(Path(folder_path))
            logger.info("Optuna visualizations completed successfully")

        except Exception as e:
            logger.error(f"Failed to run Optuna visualizations: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Window Tracking and Management
    # -------------------------------------------------------------------------

    def _track_window(self, window: QWidget) -> None:
        """
        Add a window to the tracking list.

        Args:
            window: Window or dialog to track
        """
        self._open_windows.append(window)
        logger.debug(f"Tracking window: {window.__class__.__name__} (total tracked: {len(self._open_windows)})")

        # Connect to destroyed signal to remove from tracking
        # Use weakref-based approach to avoid preventing garbage collection
        # and capturing a reference to a destroyed object
        import weakref
        window_ref = weakref.ref(window)
        window.destroyed.connect(lambda: self._on_window_destroyed_ref(window_ref))

    def _on_window_destroyed_ref(self, window_ref) -> None:
        """
        Handle window destruction event using a weak reference.

        Args:
            window_ref: Weak reference to the window that was destroyed
        """
        # The weak reference will return None after the object is destroyed
        # We need to scan the list to find and remove the dead reference
        # Filter out any windows that are no longer valid
        original_count = len(self._open_windows)
        self._open_windows = [w for w in self._open_windows if w is not None]
        removed_count = original_count - len(self._open_windows)
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} destroyed window(s) (remaining: {len(self._open_windows)})")

    def _on_window_destroyed(self, window: QWidget) -> None:
        """
        Handle window destruction event (legacy method for backwards compatibility).

        Args:
            window: Window that was destroyed
        """
        if window in self._open_windows:
            self._open_windows.remove(window)
            logger.debug(f"Window destroyed: {window.__class__.__name__} (remaining: {len(self._open_windows)})")

    def get_open_windows(self) -> List[QWidget]:
        """
        Get list of currently open windows.

        Returns:
            List[QWidget]: List of open windows
        """
        # Filter out any None values (windows that were destroyed)
        self._open_windows = [w for w in self._open_windows if w is not None]
        return self._open_windows

    def close_all_windows(self) -> None:
        """
        Close all tracked windows.

        This should be called during application shutdown.
        """
        logger.info(f"Closing {len(self._open_windows)} open window(s)...")

        # Make a copy of the list since closing windows will modify it
        windows_to_close = list(self._open_windows)

        for window in windows_to_close:
            try:
                if window is not None and not window.isHidden():
                    logger.debug(f"Closing window: {window.__class__.__name__}")
                    window.close()
            except Exception as e:
                logger.error(f"Error closing window {window.__class__.__name__}: {e}")

        self._open_windows.clear()
        logger.info("All windows closed")

    # -------------------------------------------------------------------------
    # Status and Info
    # -------------------------------------------------------------------------

    def get_window_count(self) -> int:
        """
        Get number of tracked windows.

        Returns:
            int: Number of tracked windows
        """
        return len(self.get_open_windows())

    def print_status(self) -> None:
        """
        Print status of all tracked windows (for debugging).
        """
        logger.info("=== Window Manager Status ===")
        logger.info(f"Tracked windows: {self.get_window_count()}")

        for i, window in enumerate(self.get_open_windows(), 1):
            visible = "VISIBLE" if window.isVisible() else "HIDDEN"
            logger.info(f"  Window {i}: {window.__class__.__name__} ({visible})")

        logger.info("=============================")
