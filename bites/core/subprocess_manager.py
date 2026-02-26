"""
Subprocess Manager - Centralized subprocess lifecycle management.

This module manages all subprocess operations in the application, providing:
- Consistent subprocess launching
- Process tracking and cleanup
- Error handling
- Logging of subprocess events
"""

import os
import sys
import logging
import subprocess
import threading
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_script_path() -> str:
    """Get the resolved main script path for subprocess commands.

    For frozen executables (PyInstaller), returns None since sys.executable
    is the .exe itself. For normal Python, resolves sys.argv[0] to an
    absolute path to prevent path confusion.
    """
    return str(Path(sys.argv[0]).resolve())


class SubprocessManager:
    """
    Centralized manager for subprocess operations.

    This class handles launching and tracking subprocesses for:
    - Optimization runs
    - Shape browser visualization
    - Other heavy computation tasks

    Features:
    - Automatic process tracking
    - Proper cleanup on application exit
    - Consistent error handling
    - Logging of all subprocess events
    - Thread-safe process list management
    """

    def __init__(self):
        """Initialize the subprocess manager."""
        self._processes: List[subprocess.Popen] = []
        self._lock = threading.Lock()
        logger.info("Initialized SubprocessManager")

    def _prune_completed_processes(self) -> int:
        """
        Remove completed processes from the tracking list.

        This prevents memory leaks from accumulating dead Popen objects.
        Must be called with self._lock held.

        Returns:
            int: Number of processes removed
        """
        original_count = len(self._processes)
        self._processes = [proc for proc in self._processes if proc.poll() is None]
        removed_count = original_count - len(self._processes)
        if removed_count > 0:
            logger.debug(f"Pruned {removed_count} completed process(es) from tracking list")
        return removed_count

    # -------------------------------------------------------------------------
    # Optimization Subprocess
    # -------------------------------------------------------------------------

    def launch_optimization(
        self,
        config_path: str,
        wait: bool = False,
        capture_output: bool = False
    ) -> subprocess.Popen:
        """
        Launch optimization process in a subprocess.

        Args:
            config_path: Path to the optimization configuration file
            wait: If True, wait for process to complete
            capture_output: If True, capture stdout/stderr via PIPE.
                           WARNING: If capture_output=True and wait=False, the caller
                           MUST read from proc.stdout to avoid pipe buffer deadlock.

        Returns:
            subprocess.Popen: The launched process

        Raises:
            FileNotFoundError: If config file doesn't exist
            RuntimeError: If subprocess fails to start
        """
        # Validate config file exists and is a file
        config = Path(config_path)
        if not config.exists():
            error_msg = f"Config file not found: {config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        if not config.is_file():
            error_msg = f"Config path is not a file: {config_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Launching optimization subprocess with config: {config_path}")

        # Prepare environment
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Ensure unbuffered output

        try:
            # Configure output handling
            # Default to DEVNULL to prevent pipe buffer deadlock when output isn't consumed
            stdout_dest = subprocess.PIPE if capture_output else subprocess.DEVNULL
            stderr_dest = subprocess.STDOUT if capture_output else subprocess.DEVNULL

            # Launch subprocess
            proc = subprocess.Popen(
                [sys.executable, "-u", _get_script_path(), "--optimize", config_path],
                stdout=stdout_dest,
                stderr=stderr_dest,
                env=env
            )

            with self._lock:
                self._prune_completed_processes()
                self._processes.append(proc)
            logger.info(f"Optimization subprocess started (PID: {proc.pid})")

            if wait:
                logger.debug("Waiting for optimization subprocess to complete...")
                proc.wait()
                logger.info(f"Optimization subprocess completed (return code: {proc.returncode})")

            return proc

        except Exception as e:
            error_msg = f"Failed to launch optimization subprocess: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    # -------------------------------------------------------------------------
    # Shape Browser Subprocess
    # -------------------------------------------------------------------------

    def launch_shape_browser(
        self,
        folder_path: str,
        wait: bool = False
    ) -> subprocess.Popen:
        """
        Launch shape browser in a subprocess.

        Args:
            folder_path: Path to the output folder to browse
            wait: If True, wait for process to complete

        Returns:
            subprocess.Popen: The launched process

        Raises:
            FileNotFoundError: If folder doesn't exist
            RuntimeError: If subprocess fails to start
        """
        # Validate folder exists
        if not Path(folder_path).exists():
            error_msg = f"Output folder not found: {folder_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not Path(folder_path).is_dir():
            error_msg = f"Path is not a directory: {folder_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Launching shape browser subprocess for folder: {folder_path}")

        # Create log file path for subprocess errors
        from bites.paths import get_logs_folder, ensure_folder_exists
        logs_folder = ensure_folder_exists(get_logs_folder())
        error_log_path = logs_folder / "shape_browser_subprocess.log"

        error_log_file = None
        try:
            # Open log file for subprocess stderr
            # This captures errors that would otherwise be lost in a frozen executable
            error_log_file = open(error_log_path, 'w', encoding='utf-8')

            # For frozen executables (PyInstaller), sys.executable IS the script,
            # so we don't need sys.argv[0]. For normal Python, we need both.
            if getattr(sys, 'frozen', False):
                # Frozen executable: sys.executable is the .exe itself
                cmd = [sys.executable, "--shape-browser", folder_path]
            else:
                # Normal Python: need interpreter + script
                cmd = [sys.executable, _get_script_path(), "--shape-browser", folder_path]

            logger.info(f"Shape browser command: {cmd}")

            # Launch subprocess with stderr captured to log file
            proc = subprocess.Popen(
                cmd,
                stderr=error_log_file,
                # Don't capture stdout - let it go to console/nowhere
            )

            # Store log file handle on process for cleanup
            proc._error_log_file = error_log_file
            proc._error_log_path = error_log_path

            with self._lock:
                self._prune_completed_processes()
                self._processes.append(proc)
            logger.info(f"Shape browser subprocess started (PID: {proc.pid})")
            logger.info(f"Shape browser errors will be logged to: {error_log_path}")

            if wait:
                logger.debug("Waiting for shape browser subprocess to complete...")
                proc.wait()
                error_log_file.close()
                error_log_file = None  # Prevent double-close in except block

                # Check for errors
                if proc.returncode != 0:
                    # Read and log any errors
                    try:
                        with open(error_log_path, 'r', encoding='utf-8') as f:
                            error_content = f.read().strip()
                        if error_content:
                            logger.error(f"Shape browser subprocess errors:\n{error_content}")
                    except Exception as read_err:
                        logger.warning(f"Could not read error log: {read_err}")

                logger.info(f"Shape browser subprocess completed (return code: {proc.returncode})")

            return proc

        except Exception as e:
            # Close the log file if Popen failed before it could be attached to the process
            if error_log_file is not None:
                try:
                    error_log_file.close()
                except OSError as close_err:
                    logger.warning(f"Error closing error log file: {close_err}")
            error_msg = f"Failed to launch shape browser subprocess: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    # -------------------------------------------------------------------------
    # Generic Subprocess Launch
    # -------------------------------------------------------------------------

    def launch_generic(
        self,
        args: List[str],
        wait: bool = False,
        capture_output: bool = False
    ) -> subprocess.Popen:
        """
        Launch a generic subprocess (for future extensibility).

        Args:
            args: Command-line arguments
            wait: If True, wait for process to complete
            capture_output: If True, capture stdout/stderr

        Returns:
            subprocess.Popen: The launched process

        Raises:
            RuntimeError: If subprocess fails to start
        """
        logger.info(f"Launching generic subprocess: {' '.join(args)}")

        try:
            kwargs = {}
            if capture_output:
                kwargs['stdout'] = subprocess.PIPE
                kwargs['stderr'] = subprocess.PIPE

            proc = subprocess.Popen(args, **kwargs)
            with self._lock:
                self._prune_completed_processes()
                self._processes.append(proc)
            logger.info(f"Generic subprocess started (PID: {proc.pid})")

            if wait:
                logger.debug("Waiting for subprocess to complete...")
                proc.wait()
                logger.info(f"Subprocess completed (return code: {proc.returncode})")

            return proc

        except Exception as e:
            error_msg = f"Failed to launch subprocess: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    # -------------------------------------------------------------------------
    # Process Management
    # -------------------------------------------------------------------------

    def get_active_processes(self) -> List[subprocess.Popen]:
        """
        Get list of currently running processes.

        Also prunes completed processes from the tracking list.

        Returns:
            List[subprocess.Popen]: List of active processes
        """
        with self._lock:
            # Prune and return - after pruning, all remaining processes are active
            self._prune_completed_processes()
            logger.debug(f"Found {len(self._processes)} active subprocesses")
            return list(self._processes)  # Return a copy to prevent external modification

    def wait_for_all(self, timeout: Optional[float] = None) -> None:
        """
        Wait for all subprocesses to complete.

        Args:
            timeout: Maximum time to wait per process in seconds (None = wait forever)

        Raises:
            subprocess.TimeoutExpired: If timeout is exceeded
        """
        with self._lock:
            processes = list(self._processes)

        logger.info(f"Waiting for {len(processes)} subprocess(es) to complete...")

        for proc in processes:
            if proc.poll() is None:  # Still running
                try:
                    proc.wait(timeout=timeout)
                    logger.debug(f"Process {proc.pid} completed (return code: {proc.returncode})")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {proc.pid} did not complete within {timeout}s")
                    raise

        logger.info("All subprocesses completed")

    def terminate_all(self) -> None:
        """
        Terminate all running subprocesses gracefully.
        """
        active_processes = self.get_active_processes()

        if not active_processes:
            logger.debug("No active processes to terminate")
            return

        logger.info(f"Terminating {len(active_processes)} subprocess(es)...")

        for proc in active_processes:
            try:
                logger.debug(f"Terminating process {proc.pid}...")
                proc.terminate()

                # Give process time to terminate gracefully
                try:
                    proc.wait(timeout=5)
                    logger.debug(f"Process {proc.pid} terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if termination didn't work
                    logger.warning(f"Process {proc.pid} did not terminate, forcing kill...")
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.error(f"Process {proc.pid} could not be killed")
                    else:
                        logger.debug(f"Process {proc.pid} killed")

            except Exception as e:
                logger.error(f"Error terminating process {proc.pid}: {e}")

        # Close any error log file handles attached to processes
        with self._lock:
            for proc in self._processes:
                if hasattr(proc, '_error_log_file') and proc._error_log_file is not None:
                    try:
                        proc._error_log_file.close()
                    except OSError:
                        pass
                    proc._error_log_file = None

        logger.info("All subprocesses terminated")

    def cleanup(self) -> None:
        """
        Clean up all subprocess resources.

        This should be called when the application is shutting down.
        """
        logger.info("Cleaning up subprocess manager...")

        # Terminate any still-running processes
        self.terminate_all()

        # Clear process list
        with self._lock:
            self._processes.clear()

        logger.info("Subprocess manager cleanup complete")

    # -------------------------------------------------------------------------
    # Status and Info
    # -------------------------------------------------------------------------

    def get_process_count(self) -> int:
        """
        Get total number of tracked processes (active + completed).

        Returns:
            int: Number of tracked processes
        """
        with self._lock:
            return len(self._processes)

    def get_active_count(self) -> int:
        """
        Get number of currently running processes.

        Returns:
            int: Number of active processes
        """
        return len(self.get_active_processes())

    def print_status(self) -> None:
        """
        Print status of all tracked processes (for debugging).
        """
        logger.info("=== Subprocess Status ===")
        logger.info(f"Total processes tracked: {self.get_process_count()}")
        logger.info(f"Active processes: {self.get_active_count()}")

        with self._lock:
            for i, proc in enumerate(self._processes, 1):
                status = "RUNNING" if proc.poll() is None else f"COMPLETED (code: {proc.returncode})"
                logger.info(f"  Process {i} (PID: {proc.pid}): {status}")

        logger.info("========================")
