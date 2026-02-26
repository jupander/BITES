"""
Application Controller - Central coordinator for the BITES application.

This module provides the main application controller that coordinates all
high-level application operations and manages the lifecycle of core services.
"""

import logging
from typing import Optional
from pathlib import Path

from .settings_manager import SettingsManager
from .subprocess_manager import SubprocessManager
from .window_manager import WindowManager

logger = logging.getLogger(__name__)


class ApplicationController:
    """
    Central coordinator for the BITES application.

    This class serves as the main entry point for application-level operations,
    coordinating between settings, subprocess management, window management,
    and business logic.

    Responsibilities:
    - Initialize and manage application-wide services
    - Coordinate high-level application workflows
    - Handle application lifecycle (startup, shutdown)
    - Provide simple API for GUI components

    Example:
        controller = ApplicationController()
        controller.run()
    """

    def __init__(self):
        """
        Initialize the application controller and all core services.
        """
        logger.info("=" * 60)
        logger.info("Initializing ApplicationController")
        logger.info("=" * 60)

        # Initialize core services
        self.settings = SettingsManager()

        # Migrate old geometry keys to new standardized format (one-time operation)
        self.settings.migrate_geometry_keys()

        self.subprocess_mgr = SubprocessManager()
        self.window_mgr = WindowManager(self)

        # Main window reference (set when run() is called)
        self.main_window = None

        logger.info("ApplicationController initialized successfully")

    # -------------------------------------------------------------------------
    # Application Lifecycle
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the application by creating and showing the main window.

        Returns:
            MainMenuWindow: The main application window
        """
        logger.info("Starting application...")

        try:
            self.main_window = self.window_mgr.create_main_window()
            self.main_window.show()
            logger.info("Application started successfully")
            return self.main_window

        except Exception as e:
            logger.error(f"Failed to start application: {e}", exc_info=True)
            raise

    def shutdown(self) -> None:
        """
        Gracefully shutdown the application.

        This should be called when the application is closing to ensure
        proper cleanup of all resources.
        """
        logger.info("=" * 60)
        logger.info("Shutting down application...")
        logger.info("=" * 60)

        try:
            # Close all windows
            logger.info("Closing all windows...")
            self.window_mgr.close_all_windows()

            # Terminate subprocesses
            logger.info("Cleaning up subprocesses...")
            self.subprocess_mgr.cleanup()

            # Sync settings
            logger.info("Syncing settings...")
            self.settings.sync()

            logger.info("Application shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Optimization Workflows
    # -------------------------------------------------------------------------

    def launch_optimization(self, config_path: Optional[str] = None) -> None:
        """
        Launch optimization process.

        This is a high-level method that handles the full workflow:
        1. Validate config file
        2. Launch subprocess
        3. Show output window
        4. Update settings on completion

        Args:
            config_path: Path to config file (if None, uses settings)

        Raises:
            ValueError: If no valid config file is available
            RuntimeError: If subprocess launch fails
        """
        # Get config path from settings if not provided
        if config_path is None:
            config_path = self.settings.optimization_config_file

        if config_path is None:
            error_msg = "No optimization config file specified"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Launching optimization workflow (config={config_path})...")

        # Validate config file
        if not Path(config_path).exists():
            error_msg = f"Config file not found: {config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Show output window (it will launch the subprocess internally)
        # Note: We keep this pattern for now to maintain compatibility
        # with the existing OptimizationOutputWindow implementation
        try:
            output_window = self.window_mgr.show_optimization_output_window(
                config_path,
                parent=self.main_window
            )
            output_window.show()
            logger.info("Optimization workflow initiated successfully")

        except Exception as e:
            logger.error(f"Failed to launch optimization: {e}", exc_info=True)
            raise

    def update_results_folder(self, folder_path: str) -> None:
        """
        Update the output folder path in settings.

        Args:
            folder_path: Path to the output folder
        """
        logger.info(f"Updating output folder path: {folder_path}")
        self.settings.results_folder_path = folder_path

    # -------------------------------------------------------------------------
    # Visualization Workflows
    # -------------------------------------------------------------------------

    def launch_shape_browser(self, folder_path: Optional[str] = None) -> None:
        """
        Launch shape browser in a subprocess.

        Args:
            folder_path: Path to output folder (if None, uses settings)

        Raises:
            ValueError: If no valid folder path is available
            RuntimeError: If subprocess launch fails
        """
        # Get folder path from settings if not provided
        if folder_path is None:
            folder_path = self.settings.results_folder_path

        if folder_path is None:
            error_msg = "No output folder path specified"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Launching shape browser (folder={folder_path})...")

        try:
            # Launch subprocess (non-blocking)
            proc = self.subprocess_mgr.launch_shape_browser(folder_path, wait=False)
            logger.info(f"Shape browser launched successfully (PID: {proc.pid})")

        except Exception as e:
            logger.error(f"Failed to launch shape browser: {e}", exc_info=True)
            raise

    def run_optuna_visualizations(self, folder_path: Optional[str] = None) -> None:
        """
        Run Optuna visualizations (blocking operation).

        Args:
            folder_path: Path to output folder (if None, uses settings)

        Raises:
            ValueError: If no valid folder path is available
        """
        # Get folder path from settings if not provided
        if folder_path is None:
            folder_path = self.settings.results_folder_path

        if folder_path is None:
            error_msg = "No output folder path specified"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Running Optuna visualizations (folder={folder_path})...")

        try:
            self.window_mgr.run_optuna_visualizations(folder_path)
            logger.info("Optuna visualizations completed successfully")

        except Exception as e:
            logger.error(f"Failed to run Optuna visualizations: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Configuration Workflows
    # -------------------------------------------------------------------------

    def show_config_editor(self, parent=None) -> bool:
        """
        Show configuration editor dialog.

        Args:
            parent: Parent widget (optional)

        Returns:
            bool: True if config was accepted and saved

        Raises:
            RuntimeError: If dialog creation fails
        """
        logger.info("Showing configuration editor...")

        try:
            dialog = self.window_mgr.show_config_window(parent=parent)
            result = dialog.exec()

            if result:  # Accepted
                # Update settings with new config path
                if hasattr(dialog, 'config_file_path'):
                    self.settings.optimization_config_file = dialog.config_file_path
                    logger.info(f"Config file updated: {dialog.config_file_path}")
                return True
            else:
                logger.info("Config editor cancelled")
                return False

        except Exception as e:
            logger.error(f"Failed to show config editor: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Tool Dialogs
    # -------------------------------------------------------------------------

    def show_cut_tester(self, mesh_path: str, default_height: float, parent=None) -> None:
        """
        Show mesh cut tester dialog.

        Args:
            mesh_path: Path to mesh file
            default_height: Default cut height
            parent: Parent widget (optional)

        Returns:
            MeshCutDialog: The dialog instance
        """
        logger.info(f"Showing cut tester (mesh={mesh_path}, height={default_height})...")

        try:
            dialog = self.window_mgr.show_cut_tester(mesh_path, default_height, parent=parent)
            return dialog

        except Exception as e:
            logger.error(f"Failed to show cut tester: {e}", exc_info=True)
            raise

    def show_simplifier(self, mesh_path: str, parent=None) -> None:
        """
        Show mesh simplifier dialog.

        Args:
            mesh_path: Path to mesh file
            parent: Parent widget (optional)

        Returns:
            MeshSimplifierDialog: The dialog instance
        """
        logger.info(f"Showing simplifier (mesh={mesh_path})...")

        try:
            dialog = self.window_mgr.show_simplifier(mesh_path, parent=parent)
            return dialog

        except Exception as e:
            logger.error(f"Failed to show simplifier: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Status and Debugging
    # -------------------------------------------------------------------------

    def print_status(self) -> None:
        """
        Print comprehensive application status (for debugging).
        """
        logger.info("=" * 60)
        logger.info("APPLICATION STATUS")
        logger.info("=" * 60)

        # Settings
        logger.info("\n--- Settings ---")
        logger.info(f"Config file: {self.settings.optimization_config_file}")
        logger.info(f"Output folder: {self.settings.results_folder_path}")

        # Subprocesses
        logger.info("\n--- Subprocesses ---")
        self.subprocess_mgr.print_status()

        # Windows
        logger.info("\n--- Windows ---")
        self.window_mgr.print_status()

        logger.info("=" * 60)
