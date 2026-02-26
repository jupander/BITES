"""
Optimization Console Window - Real-time output display for optimization subprocess.

This module provides a window that displays real-time output from the optimization
subprocess, with proper process lifecycle management and cleanup.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional

import time as _time
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QElapsedTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPlainTextEdit, QMessageBox, QPushButton,
    QProgressBar, QLabel
)

from bites.paths import get_results_folder
from bites.utils.subprocess_utils import create_subprocess_with_cleanup, terminate_process_tree
from bites.gui.mixins import GeometryPersistenceMixin
import re
import bites.gui.gui_utils as gui_utils

logger = logging.getLogger(__name__)

# Maximum characters per line to prevent UI freeze on very long output lines
MAX_LINE_LENGTH = 10000

# Windows STATUS_CONTROL_C_EXIT (0xC000013A) - returned when process is terminated via Ctrl+C
# This value exceeds signed 32-bit int max, so we need to normalize it
WINDOWS_CTRL_C_EXIT = 0xC000013A


class OutputReaderThread(QThread):
    """
    Thread for reading subprocess output without blocking the UI.

    Emits line_ready signal for each line read, and finished signal when done.
    Supports cancellation via request_stop() method.
    """
    line_ready = Signal(str)
    finished_with_code = Signal(int)  # Emits return code when process exits

    def __init__(self, process):
        super().__init__()
        self.process = process
        self._stop_requested = False

    def request_stop(self) -> None:
        """Request the thread to stop reading. Thread-safe."""
        self._stop_requested = True

    def run(self) -> None:
        """Read output lines from subprocess until it exits or stop is requested."""
        return_code = -1
        try:
            # Read lines until EOF or stop requested
            while not self._stop_requested:
                line = self.process.stdout.readline()
                if not line:  # EOF
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip('\r\n')

                # Truncate very long lines to prevent UI freeze
                if len(decoded) > MAX_LINE_LENGTH:
                    decoded = decoded[:MAX_LINE_LENGTH] + "... [truncated]"

                self.line_ready.emit(decoded)

            # Wait for process to finish BEFORE closing stdout
            # This prevents writing to a closed pipe and ensures proper cleanup
            try:
                # Use longer timeout for optimization runs which can take a while to shut down
                return_code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Process did not exit within 30s timeout, attempting to terminate")
                try:
                    self.process.terminate()
                    return_code = self.process.wait(timeout=5)
                except OSError as e:
                    logger.debug(f"Failed to terminate process: {e}")
                    return_code = self.process.poll() or -1
            except OSError as e:
                logger.debug(f"Failed to wait for process: {e}")
                return_code = self.process.poll() or -1

            # Now safe to close stdout after process has exited
            try:
                self.process.stdout.close()
            except OSError as e:
                logger.debug(f"Failed to close process stdout: {e}")

        except Exception as e:
            logger.error(f"Error in output reader thread: {e}")
            return_code = -1
        finally:
            # Normalize Windows STATUS_CONTROL_C_EXIT to avoid signed int overflow in Qt signal
            # This code (0xC000013A) indicates graceful Ctrl+C termination
            if return_code == WINDOWS_CTRL_C_EXIT:
                return_code = -1  # Treat as terminated
            self.finished_with_code.emit(return_code)


class OptimizationOutputWindow(GeometryPersistenceMixin, QDialog):
    """
    Window for displaying real-time optimization subprocess output.

    This window:
    - Spawns the optimization as a subprocess
    - Displays stdout/stderr in real-time
    - Properly cleans up the subprocess on close
    - Persists window geometry across sessions

    Use the create() factory method instead of __init__ directly to handle
    initialization errors gracefully.
    """

    # Emitted when optimization completes successfully, carries the results folder path
    optimization_completed = Signal(str)

    GEOMETRY_KEY = "optimization_console"
    DEFAULT_GEOMETRY = None  # Use Qt's auto-sizing

    # Private flag to indicate successful initialization
    _initialized: bool = False
    _closing: bool = False
    _waiting_for_shutdown: bool = False

    # After sending interrupt, force-kill if no output for this many seconds.
    # This handles the case where the main thread has finished post-processing
    # but stale Optuna worker threads (still running tooth model subprocesses)
    # prevent the process from exiting.
    # Initialized from the config's tm_timeout (tooth model timeout) in __init__,
    # since that's the maximum time a stale worker thread can block.
    SHUTDOWN_INACTIVITY_TIMEOUT_S = 30  # fallback default

    def __init__(self, config_file_path: str, parent: Optional[QDialog] = None):
        """
        Initialize the optimization output window.

        Note: Prefer using the create() factory method which handles errors gracefully.

        Args:
            config_file_path: Path to the optimization config YAML file
            parent: Parent widget (used for modality, not for taskbar grouping)
        """
        # Use None as Qt parent so window gets its own taskbar button,
        # but store the logical parent for reference
        super().__init__(None)
        self._logical_parent = parent

        # Initialize attributes that closeEvent expects
        self.proc = None
        self.output_thread = None

        # Validate config file before proceeding
        config_path = Path(config_file_path)
        if not config_path.exists():
            self._show_init_error(f"Config file not found:\n{config_file_path}")
            return
        if not config_path.suffix.lower() in ('.yaml', '.yml'):
            self._show_init_error(f"Config file must be a YAML file:\n{config_file_path}")
            return

        self.config_file_path = config_file_path

        # Read optimization parameters from config for progress tracking
        try:
            config = gui_utils.load_yaml_config(config_file_path)
            self._max_trials = config.get('max_trials', 0)
            self._study_timeout = config.get('study_timeout', 0)  # in hours
            self._study_name = config.get('study_name', '')
            tm_timeout = config.get('tm_timeout', 0)
            if tm_timeout > 0:
                self.SHUTDOWN_INACTIVITY_TIMEOUT_S = tm_timeout
        except Exception as e:
            logger.warning(f"Could not read config for progress bars: {e}")
            self._max_trials = 0
            self._study_timeout = 0
            self._study_name = ''

        self._trial_pattern = re.compile(r"Trial \d+ finished")
        self._pre_existing_pattern = re.compile(r"Pre-existing trials in study: (\d+)")

        self._setup_window()
        self._setup_ui()

        # Let Qt calculate optimal size based on content
        self.adjustSize()

        # Initialize geometry persistence (restores saved position/size)
        self._init_geometry_persistence()

        if not self._start_subprocess():
            return

        self._start_output_reader()
        self._initialized = True

    @classmethod
    def create(cls, config_file_path: str, parent: Optional[QDialog] = None) -> Optional['OptimizationOutputWindow']:
        """
        Factory method to create an OptimizationOutputWindow.

        Returns None if initialization fails, allowing the caller to handle
        the error gracefully without dealing with a partially constructed object.

        Args:
            config_file_path: Path to the optimization config YAML file
            parent: Parent widget

        Returns:
            OptimizationOutputWindow instance, or None if initialization failed
        """
        window = cls(config_file_path, parent)
        if not window._initialized:
            # Window failed to initialize, clean up and return None
            window.deleteLater()
            return None
        return window

    def _show_init_error(self, message: str) -> None:
        """Show an error message during initialization."""
        logger.error(f"Optimization window init error: {message}")
        QMessageBox.critical(self, "Initialization Error", message)

    def _setup_window(self) -> None:
        """Configure window properties."""
        title = "Running parameter optimization..."
        if self._study_name:
            title = f"[{self._study_name}] {title}"
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)

        # Window flags for standard window behavior
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinMaxButtonsHint |
            Qt.WindowCloseButtonHint |
            Qt.WindowSystemMenuHint
        )

    def _setup_ui(self) -> None:
        """Create and configure UI elements."""
        layout = QVBoxLayout()

        # Button bar at top
        button_bar = QHBoxLayout()
        button_bar.setContentsMargins(0, 0, 0, 5)

        self.live_viewer_button = QPushButton("Show live preview")
        self.live_viewer_button.setToolTip(
            "Open a 3D viewer showing the best trial so far.\n"
            "Updates automatically as optimization progresses."
        )
        self.live_viewer_button.clicked.connect(self._launch_live_viewer)
        button_bar.addWidget(self.live_viewer_button)

        button_bar.addStretch()
        layout.addLayout(button_bar)

        # --- Progress bars ---
        # Trial progress bar (always shown if max_trials > 0)
        if self._max_trials > 0:
            trial_bar_layout = QHBoxLayout()
            trial_bar_layout.setContentsMargins(0, 0, 0, 0)
            self._trial_label = QLabel(f"Trials: 0 / {self._max_trials}")
            self._trial_progress = QProgressBar()
            self._trial_progress.setRange(0, self._max_trials)
            self._trial_progress.setValue(0)
            self._trial_progress.setTextVisible(True)
            #self._trial_progress.setFormat("%v / %m  (%p%)")
            self._trial_progress.setFormat("%p%")
            trial_bar_layout.addWidget(self._trial_label)
            trial_bar_layout.addWidget(self._trial_progress, 1)
            layout.addLayout(trial_bar_layout)
        else:
            self._trial_label = None
            self._trial_progress = None

        # Timeout progress bar (only shown when study_timeout > 0)
        if self._study_timeout > 0:
            timeout_bar_layout = QHBoxLayout()
            timeout_bar_layout.setContentsMargins(0, 0, 0, 0)
            timeout_seconds = int(self._study_timeout * 3600)
            self._timeout_label = QLabel("Time: 0:00:00 / " + self._format_duration(timeout_seconds))
            self._timeout_progress = QProgressBar()
            self._timeout_progress.setRange(0, timeout_seconds)
            self._timeout_progress.setValue(0)
            self._timeout_progress.setTextVisible(True)
            self._timeout_progress.setFormat("%p%")
            timeout_bar_layout.addWidget(self._timeout_label)
            timeout_bar_layout.addWidget(self._timeout_progress, 1)
            layout.addLayout(timeout_bar_layout)

            # Timer to update elapsed time every second
            self._elapsed_timer = QElapsedTimer()
            self._timeout_seconds = timeout_seconds
            self._timeout_update_timer = QTimer(self)
            self._timeout_update_timer.setInterval(1000)
            self._timeout_update_timer.timeout.connect(self._update_timeout_progress)
        else:
            self._timeout_label = None
            self._timeout_progress = None
            self._timeout_update_timer = None

        self._completed_trials = 0
        self._pre_existing_trials = 0

        self.output_view = QPlainTextEdit(self)
        self.output_view.setReadOnly(True)
        self.output_view.setFrameStyle(QPlainTextEdit.NoFrame)
        self.output_view.setStyleSheet(
            "background-color: transparent; font-family: monospace;"
        )
        self.output_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Limit the document size to prevent memory issues with very long output
        # 20000 blocks is sufficient for viewing recent logs while limiting memory usage
        # (reduced from 50000 to improve memory efficiency during long optimization runs)
        self.output_view.setMaximumBlockCount(20000)

        layout.addWidget(self.output_view)
        self.setLayout(layout)

        # Track the Live preview window
        self._live_viewer = None

    def _start_subprocess(self) -> bool:
        """
        Start the optimization subprocess.

        Returns:
            True if subprocess started successfully, False otherwise
        """
        import subprocess

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Ensure unbuffered output

        try:
            logger.info(f"Starting optimization subprocess with config: {self.config_file_path}")
            script_path = str(Path(sys.argv[0]).resolve())
            self.proc = create_subprocess_with_cleanup(
                [sys.executable, "-u", script_path, "--optimize", self.config_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            logger.info(f"Optimization subprocess started (PID: {self.proc.pid})")
            return True
        except FileNotFoundError as e:
            self._show_init_error(f"Python executable not found:\n{e}")
            return False
        except Exception as e:
            self._show_init_error(f"Failed to launch optimization subprocess:\n{e}")
            return False

    def _start_output_reader(self) -> None:
        """Start the thread that reads subprocess output."""
        self.output_thread = OutputReaderThread(self.proc)
        self.output_thread.line_ready.connect(self._append_output)
        self.output_thread.finished_with_code.connect(self._on_subprocess_finished)
        self.output_thread.start()

    def _append_output(self, line: str) -> None:
        """
        Append a line of output to the display and update progress bars.

        Args:
            line: The output line to append
        """
        self.output_view.appendPlainText(line)

        # Track activity during shutdown so the inactivity timer can detect stalls
        if self._waiting_for_shutdown:
            self._last_output_time = _time.monotonic()

        # Parse pre-existing trial count (emitted once before optimization starts)
        if self._pre_existing_trials == 0:
            pre_match = self._pre_existing_pattern.search(line)
            if pre_match:
                self._pre_existing_trials = int(pre_match.group(1))
                if self._trial_progress is not None:
                    self._update_trial_label()

        # Update trial progress bar: count each "Trial X finished" line
        # (don't use trial number, as parallel trials complete out of order)
        if self._trial_progress is not None and self._trial_pattern.search(line):
            self._completed_trials += 1
            self._trial_progress.setValue(min(self._completed_trials, self._max_trials))
            self._update_trial_label()

        # Start the timeout timer on the first meaningful output line
        # (indicates the subprocess has started running)
        if self._timeout_update_timer is not None and not self._timeout_update_timer.isActive():
            if "Starting optimization" in line:
                self._elapsed_timer.start()
                self._timeout_update_timer.start()

    def _update_trial_label(self) -> None:
        """Update the trial progress label, showing pre-existing trials if any."""
        label = f"Trials: {self._completed_trials} / {self._max_trials}"
        if self._pre_existing_trials > 0:
            label += f"  (+{self._pre_existing_trials} prior)"
        self._trial_label.setText(label)

    def _update_timeout_progress(self) -> None:
        """Update the timeout progress bar with elapsed time."""
        elapsed_ms = self._elapsed_timer.elapsed()
        elapsed_sec = int(elapsed_ms / 1000)
        clamped = min(elapsed_sec, self._timeout_seconds)

        self._timeout_progress.setValue(clamped)
        self._timeout_label.setText(
            f"Time: {self._format_duration(elapsed_sec)} / {self._format_duration(self._timeout_seconds)}"
        )

        # Stop updating once we've reached the timeout
        if elapsed_sec >= self._timeout_seconds:
            self._timeout_update_timer.stop()

    @staticmethod
    def _format_duration(seconds: int) -> str:
        """Format seconds as H:MM:SS."""
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"

    def _on_subprocess_finished(self, return_code: int) -> None:
        """
        Handle subprocess completion.

        Args:
            return_code: The subprocess exit code
        """
        # Stop the timeout timer
        if self._timeout_update_timer is not None:
            self._timeout_update_timer.stop()

        if return_code == 0:
            self.output_view.appendHtml(
                "<br><font color='green'><b>Optimization completed successfully.</b></font>"
            )
            self.setWindowTitle(f"[{self._study_name}] Optimization completed" if self._study_name else "Optimization completed")
            logger.info("Optimization subprocess completed successfully")

            # Emit results folder path so the Main Menu can update
            results_folder = self._get_results_folder()
            if results_folder:
                self.optimization_completed.emit(str(results_folder))
        elif return_code == -1:
            self.output_view.appendHtml(
                "<br><font color='orange'><b>Optimization process was terminated.</b></font>"
            )
            self.setWindowTitle(f"[{self._study_name}] Optimization terminated" if self._study_name else "Optimization terminated")
            logger.warning("Optimization subprocess was terminated")
        else:
            self.output_view.appendHtml(
                f"<br><font color='red'><b>Optimization failed with exit code {return_code}.</b></font>"
            )
            self.setWindowTitle(f"[{self._study_name}] Optimization failed (code {return_code})" if self._study_name else f"Optimization failed (code {return_code})")
            logger.error(f"Optimization subprocess failed with exit code {return_code}")

    def _get_results_folder(self) -> Optional[Path]:
        """
        Determine the results folder path from the config file.

        Returns:
            Path to the results folder, or None if it cannot be determined.
        """
        try:
            config = gui_utils.load_yaml_config(self.config_file_path)
            study_name = config.get('study_name')
            if not study_name:
                logger.warning("No study_name found in config")
                return None

            # Use centralized results folder (respects overrides and frozen exe)
            results_root = get_results_folder()
            results_folder = results_root / study_name

            return results_folder

        except Exception as e:
            logger.error(f"Error determining results folder: {e}")
            return None

    def _launch_live_viewer(self) -> None:
        """Launch the Live preview window to show the best trial."""
        # Import here to avoid circular imports
        from bites.gui.visualization.viewers.live_viewer import launch_live_viewer

        results_folder = self._get_results_folder()
        if not results_folder:
            QMessageBox.warning(
                self,
                "Cannot launch live preview",
                "Could not determine the results folder from the configuration."
            )
            return

        if not results_folder.exists():
            # Results folder may not exist yet if optimization just started
            QMessageBox.information(
                self,
                "Results Folder Not Ready",
                f"The results folder does not exist yet:\n{results_folder}\n\n"
                "This is normal if optimization just started. "
                "Wait for the first trial to complete and try again."
            )
            return

        # Close existing viewer if any
        if self._live_viewer is not None:
            try:
                self._live_viewer.close()
            except RuntimeError as e:
                logger.debug(f"Failed to close previous live preview: {e}")

        self._live_viewer = launch_live_viewer(results_folder, self)
        if self._live_viewer:
            self.output_view.appendHtml(
                #"<br><font color='cyan'>Live preview opened - showing best trial as optimization progresses.</font>"
                "<br><font>Live preview opened - showing best trial as optimization progresses.</font>"
            )

    def _send_interrupt_signal(self) -> None:
        """Send an interrupt signal to the subprocess without blocking."""
        import signal

        if sys.platform == 'win32':
            try:
                os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)
            except (OSError, ProcessLookupError) as e:
                logger.debug(f"Failed to send CTRL_BREAK_EVENT: {e}")
                if self.proc.poll() is None:
                    terminate_process_tree(self.proc)
        else:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError) as e:
                logger.debug(f"Failed to send SIGINT: {e}")
                if self.proc.poll() is None:
                    terminate_process_tree(self.proc)

    def _check_shutdown_inactivity(self) -> None:
        """Periodically check if the subprocess has gone silent during shutdown."""
        if not self._waiting_for_shutdown:
            self._inactivity_timer.stop()
            return

        # If the process already exited, the finished_with_code signal will handle it
        if self.proc.poll() is not None:
            self._inactivity_timer.stop()
            return

        idle_seconds = _time.monotonic() - self._last_output_time
        if idle_seconds >= self.SHUTDOWN_INACTIVITY_TIMEOUT_S:
            self._inactivity_timer.stop()
            logger.warning(
                f"No subprocess output for {idle_seconds:.0f}s during shutdown — "
                f"force-terminating (stale worker threads likely blocking exit)"
            )
            self.output_view.appendHtml(
                f"<br><font color='orange'><b>No activity for {idle_seconds:.0f}s after stop signal. "
                f"Force-closing (results were already saved).</b></font>"
            )
            self._on_shutdown_timeout()

    def _on_shutdown_complete(self, return_code: int) -> None:
        """Called when the subprocess exits after a graceful shutdown request."""
        if not self._waiting_for_shutdown:
            return

        # Stop both timers
        if hasattr(self, '_shutdown_timer'):
            self._shutdown_timer.stop()
        if hasattr(self, '_inactivity_timer'):
            self._inactivity_timer.stop()

        if return_code == -1:
            logger.info("Optimization subprocess exited after graceful shutdown")
        else:
            logger.info(f"Optimization subprocess exited (return code: {return_code})")

        self._finish_close()

    def _on_shutdown_timeout(self) -> None:
        """Called if the subprocess doesn't exit within the timeout after interrupt."""
        if not self._waiting_for_shutdown:
            return

        # Stop the inactivity timer if it's still running
        if hasattr(self, '_inactivity_timer'):
            self._inactivity_timer.stop()

        logger.warning("Graceful shutdown timed out, force-terminating")
        self.output_view.appendHtml(
            "<br><font color='red'><b>Graceful shutdown timed out. Force-terminating...</b></font>"
        )
        terminate_process_tree(self.proc)

        # Give it a moment to die, then close
        QTimer.singleShot(2000, self._finish_close)

    def _finish_close(self) -> None:
        """Final cleanup and close the window."""
        self._waiting_for_shutdown = False

        # Wait briefly for the output thread to finish
        if self.output_thread is not None and self.output_thread.isRunning():
            logger.debug("Waiting for output reader thread to finish...")
            if not self.output_thread.wait(5000):
                logger.warning("Output reader thread did not finish in time, forcing termination")
                self.output_thread.terminate()
                self.output_thread.wait(1000)

        logger.debug("OptimizationOutputWindow cleanup complete")
        # GeometryPersistenceMixin.closeEvent saves geometry automatically
        super().closeEvent(None)
        self.close()

    def closeEvent(self, event) -> None:
        """
        Handle window close event.

        If optimization is still running, asks for confirmation and sends
        an interrupt signal for graceful shutdown. The window stays responsive
        while waiting for the subprocess to exit.
        """
        logger.debug("OptimizationOutputWindow closeEvent triggered")

        # If already waiting for shutdown, ignore additional close attempts
        if self._waiting_for_shutdown:
            event.ignore()
            return

        # Re-entrancy guard: QMessageBox.question() pumps events, which could
        # re-trigger closeEvent if the user clicks X again during the dialog
        if self._closing:
            event.ignore()
            return
        self._closing = True

        # Stop timeout timer
        if getattr(self, '_timeout_update_timer', None) is not None:
            self._timeout_update_timer.stop()

        # Close Live preview if open
        if hasattr(self, '_live_viewer') and self._live_viewer is not None:
            try:
                self._live_viewer.close()
                self._live_viewer.deleteLater()
            except RuntimeError as e:
                logger.debug(f"Live preview cleanup: {e}")
            self._live_viewer = None

        # Check if subprocess is still running
        process_running = self.proc is not None and self.proc.poll() is None

        if process_running:
            # Ask for confirmation before closing
            reply = QMessageBox.question(
                self,
                "Optimization Running",
                "The optimization is still running.\n\n"
                "If you close now, the process will be stopped gracefully, "
                f"allowing current results to be saved (this may take up to {self.SHUTDOWN_INACTIVITY_TIMEOUT_S * 2 / 60:.1f} minutes).\n\n"
                "Do you want to stop the optimization?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                self._closing = False
                event.ignore()
                return

            # User confirmed - initiate non-blocking graceful shutdown
            logger.info(f"User requested graceful shutdown of optimization subprocess (PID: {self.proc.pid})")
            self.output_view.appendHtml(
                "<br><font color='orange'><b>Stopping optimization gracefully (saving results)...</b></font>"
            )
            self.setWindowTitle(f"[{self._study_name}] Stopping optimization..." if self._study_name else "Stopping optimization...")

            # Don't close the window yet - keep it open and responsive
            event.ignore()
            self._waiting_for_shutdown = True

            # Connect finished signal to complete the close when process exits
            # (disconnect _on_subprocess_finished to avoid conflicting handlers)
            try:
                self.output_thread.finished_with_code.disconnect(self._on_subprocess_finished)
            except RuntimeError:
                pass
            self.output_thread.finished_with_code.connect(self._on_shutdown_complete)

            # Send interrupt signal (non-blocking)
            self._send_interrupt_signal()

            # --- Inactivity-based shutdown timer ---
            # After the interrupt, the subprocess should print post-processing output.
            # Once output stops (e.g., main thread done but stale Optuna worker threads
            # are still waiting on tooth model subprocesses), force-kill after a short
            # inactivity period
            self._last_output_time = _time.monotonic()
            self._inactivity_timer = QTimer(self)
            self._inactivity_timer.setInterval(5000)  # check every 5 seconds
            self._inactivity_timer.timeout.connect(self._check_shutdown_inactivity)
            self._inactivity_timer.start()

            # Absolute maximum timeout as a safety net (2x inactivity timeout)
            self._shutdown_timer = QTimer(self)
            self._shutdown_timer.setSingleShot(True)
            self._shutdown_timer.timeout.connect(self._on_shutdown_timeout)
            self._shutdown_timer.start(self.SHUTDOWN_INACTIVITY_TIMEOUT_S * 2 * 1000)

        else:
            # Process already finished, just clean up the thread
            if self.output_thread is not None:
                self.output_thread.request_stop()

            # Wait for the output thread to finish (with timeout)
            if self.output_thread is not None and self.output_thread.isRunning():
                logger.debug("Waiting for output reader thread to finish...")
                if not self.output_thread.wait(5000):
                    logger.warning("Output reader thread did not finish in time, forcing termination")
                    self.output_thread.terminate()
                    self.output_thread.wait(1000)

            logger.debug("OptimizationOutputWindow cleanup complete")
            super().closeEvent(event)
