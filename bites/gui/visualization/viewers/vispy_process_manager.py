"""
Process manager for launching VisPy-based tools in isolated processes.

CRITICAL: VisPy windows need their own isolated QApplication to be responsive.
This module uses subprocess.Popen with command-line arguments for frozen executable
compatibility (PyInstaller, cx_Freeze, etc.).

NOTE: We use subprocess.Popen instead of multiprocessing.Process because:
- multiprocessing.Process with 'spawn' context re-executes the main entry point
- In frozen executables, this causes the main menu to launch instead of the tool
- subprocess.Popen with CLI args works correctly because main.py already has
  --simplifier and --cut-tester handlers that dispatch to the right tool
"""

import sys
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox, QWidget

logger = logging.getLogger(__name__)


class VispyProcessManager:
    """
    Manages process launching and monitoring for VisPy-based tools.

    Uses subprocess.Popen with command-line arguments for frozen executable
    compatibility. Each launched tool gets its own isolated QApplication.
    """

    def __init__(self, parent_widget: QWidget):
        self.parent = parent_widget
        self._processes = {}  # Store active processes by name
        self._timers = {}     # Store monitoring timers by name

    def launch_simplifier(
        self,
        mesh_path: str,
        on_finished: Optional[Callable[[int], None]] = None,
        check_interval_ms: int = 200
    ) -> bool:
        """
        Launch the mesh simplifier in a separate process.

        Args:
            mesh_path: Path to the mesh file to simplify
            on_finished: Optional callback when process finishes, receives return code
            check_interval_ms: How often to check if process has finished

        Returns:
            True if launch succeeded, False otherwise
        """
        try:
            logger.info("Launching Simplifier process...")

            # Use subprocess.Popen with CLI args - works in frozen executables
            # sys.executable is the Python interpreter (or the .exe in frozen apps)
            # sys.argv[0] is the script/executable path
            script_path = str(Path(sys.argv[0]).resolve())
            proc = subprocess.Popen(
                [sys.executable, script_path, "--simplifier", mesh_path],
                # Don't capture output - let it go to console for debugging
            )
            logger.info(f"Simplifier process started (PID: {proc.pid})")

            # Store reference
            self._processes["Simplifier"] = proc

            # Start timer to check when process finishes
            timer = QTimer()
            timer.timeout.connect(
                lambda: self._check_finished("Simplifier", on_finished)
            )
            timer.start(check_interval_ms)
            self._timers["Simplifier"] = timer

            return True

        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to launch Simplifier: {e}")
            logger.error(f"Error launching Simplifier process: {e}")
            return False

    def launch_cut_tester(
        self,
        mesh_path: str,
        cut_height: str,
        on_finished: Optional[Callable[[int], None]] = None,
        check_interval_ms: int = 200
    ) -> bool:
        """
        Launch the cut height tester in a separate process.

        Args:
            mesh_path: Path to the mesh file
            cut_height: Current cut height value (as string, can be 'None')
            on_finished: Optional callback when process finishes, receives return code
            check_interval_ms: How often to check if process has finished

        Returns:
            True if launch succeeded, False otherwise
        """
        try:
            logger.info("Launching Cut Tester process...")

            # Use subprocess.Popen with CLI args - works in frozen executables
            script_path = str(Path(sys.argv[0]).resolve())
            proc = subprocess.Popen(
                [sys.executable, script_path, "--cut-tester", mesh_path, cut_height],
            )
            logger.info(f"Cut Tester process started (PID: {proc.pid})")

            # Store reference
            self._processes["Cut Tester"] = proc

            # Start timer to check when process finishes
            timer = QTimer()
            timer.timeout.connect(
                lambda: self._check_finished("Cut Tester", on_finished)
            )
            timer.start(check_interval_ms)
            self._timers["Cut Tester"] = timer

            return True

        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to launch Cut Tester: {e}")
            logger.error(f"Error launching Cut Tester process: {e}")
            return False

    def _check_finished(self, name: str, on_finished: Optional[Callable[[int], None]]) -> None:
        """Check if a process has finished and clean up if so."""
        proc = self._processes.get(name)
        if proc is None:
            return

        # poll() returns None if process is still running, else the return code
        exit_code = proc.poll()
        if exit_code is not None:
            # Process has finished
            timer = self._timers.get(name)
            if timer:
                timer.stop()

            logger.info(f"{name} process finished (exit code: {exit_code})")

            # Call the callback if provided
            if on_finished:
                on_finished(exit_code)

            # Clean up references
            self._processes.pop(name, None)
            self._timers.pop(name, None)

    def cleanup_all(self) -> None:
        """Clean up all process resources."""
        # Stop all timers
        for name, timer in list(self._timers.items()):
            try:
                timer.stop()
            except RuntimeError as e:
                logger.debug(f"Timer stop failed for {name}: {e}")

        # Terminate any still-running processes
        for name, proc in list(self._processes.items()):
            try:
                if proc.poll() is None:  # Still running
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except OSError as e:
                logger.warning(f"Failed to terminate process {name}: {e}")

        # Clear all references
        self._timers.clear()
        self._processes.clear()


# =============================================================================
# IPC Readers (for reading results from subprocess tools)
# =============================================================================

class CutHeightIPCReader:
    """Handles reading cut height results from IPC file after Cut Tester process."""

    IPC_FILENAME = 'bites_cut_height_result.txt'

    @classmethod
    def get_ipc_path(cls) -> Path:
        return Path(tempfile.gettempdir()) / cls.IPC_FILENAME

    @classmethod
    def read_result(cls) -> Optional[int]:
        """
        Read the cut height result from IPC file.

        Returns:
            The cut height as int, or None if file doesn't exist or is invalid
        """
        ipc_file = cls.get_ipc_path()

        if not ipc_file.exists():
            logger.info("Cut Tester process completed without saving (no IPC file found)")
            return None

        try:
            cut_height_str = ipc_file.read_text().strip()
            cut_height = int(cut_height_str)

            # Clean up IPC file
            ipc_file.unlink()

            return cut_height

        except ValueError:
            logger.warning(f"Invalid cut height value in IPC file: {cut_height_str}")
            return None
        except Exception as e:
            logger.error(f"Error reading cut height from IPC file: {e}")
            return None


class SimplifierIPCReader:
    """Handles reading simplified mesh path from IPC file after Simplifier process."""

    IPC_FILENAME = 'bites_simplifier_result.txt'

    @classmethod
    def get_ipc_path(cls) -> Path:
        return Path(tempfile.gettempdir()) / cls.IPC_FILENAME

    @classmethod
    def read_result(cls) -> Optional[str]:
        """
        Read the simplified mesh path from IPC file.

        Returns:
            The simplified mesh path as string, or None if file doesn't exist
        """
        ipc_file = cls.get_ipc_path()

        if not ipc_file.exists():
            logger.info("Simplifier process completed without saving (no IPC file found)")
            return None

        try:
            mesh_path = ipc_file.read_text().strip()

            # Clean up IPC file
            ipc_file.unlink()

            # Validate the path exists
            if mesh_path and Path(mesh_path).exists():
                return mesh_path
            else:
                logger.warning(f"Simplified mesh path does not exist: {mesh_path}")
                return None

        except Exception as e:
            logger.error(f"Error reading simplified mesh path from IPC file: {e}")
            return None
