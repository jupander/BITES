"""
Live Best Trial Viewer - Real-time visualization of the best trial during optimization.

This module provides a viewer that monitors the ongoing optimization and displays
the aligned mesh from the best trial so far, alongside the target mesh.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QMessageBox, QApplication
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
from PySide6.QtGui import QFont

from bites.gui.visualization.viewers.vispy_mesh_viewer import VispyMeshViewer
from bites.constants import algorithm_cuts_target, get_cut_mesh_filename
from bites.gui.mixins import GeometryPersistenceMixin

logger = logging.getLogger(__name__)


class _PollWorker(QThread):
    """Background worker for polling CSV files and finding the best trial.

    Performs all file I/O and DataFrame processing off the main thread to
    prevent UI freezes during polling.
    """

    result_ready = Signal(object)  # Optional[dict] - best trial info or None

    def __init__(self, results_folder: Path, parent=None):
        super().__init__(parent)
        self.results_folder = results_folder

    def run(self) -> None:
        """Load all CSVs and find the overall best trial."""
        try:
            best = self._load_study_best()
            self.result_ready.emit(best)
        except Exception as e:
            logger.debug(f"Poll worker error: {e}", exc_info=True)
            self.result_ready.emit(None)

    def _load_study_best(self) -> Optional[dict]:
        """Load the best trial from ALL CSV files in the study folder."""
        all_csvs = self._find_all_csvs()
        if not all_csvs:
            return None

        overall_best = None
        best_source = None

        for csv_path in all_csvs:
            df = self._load_csv_data(csv_path)
            if df is None or df.empty:
                continue

            best = self._find_best_trial(df)
            if best is None:
                continue

            if overall_best is None or best['loss'] < overall_best['loss']:
                overall_best = best
                best_source = csv_path.name

        if overall_best:
            overall_best['source'] = best_source

        return overall_best

    def _find_all_csvs(self) -> list[Path]:
        if not self.results_folder.exists():
            return []
        return list(self.results_folder.glob("*.csv"))

    def _find_ongoing_csv(self) -> Optional[Path]:
        if not self.results_folder.exists():
            return None
        ongoing_files = list(self.results_folder.glob("ongoing_*.csv"))
        if not ongoing_files:
            return None
        return max(ongoing_files, key=lambda p: p.stat().st_mtime)

    def _load_csv_data(self, csv_path: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(
                csv_path,
                delimiter=";",
                quoting=csv.QUOTE_NONNUMERIC,
                escapechar='\\',
                on_bad_lines='skip'
            )
        except Exception as e:
            logger.debug(f"Error reading CSV {csv_path.name}: {e}")
            return None

    def _find_best_trial(self, df: pd.DataFrame) -> Optional[dict]:
        if 'state' in df.columns:
            df = df[df['state'] == 'COMPLETE']
        if df.empty or 'loss' not in df.columns:
            return None

        df = df.dropna(subset=['loss'])
        if df.empty:
            return None

        best_idx = df['loss'].idxmin()
        best_row = df.loc[best_idx]

        trial_number = int(best_row.get('trial_number', 0))
        loss = float(best_row['loss'])
        algorithm = best_row.get('best_algorithm', 'unknown')
        mesh_identifier = best_row.get('chosen_mesh_identifier', None)

        if not mesh_identifier:
            return None

        cut_heights = None
        if 'used_cut_heights' in best_row.index:
            cut_heights_raw = best_row['used_cut_heights']
            if pd.notna(cut_heights_raw) and isinstance(cut_heights_raw, str):
                try:
                    cut_heights = json.loads(cut_heights_raw)
                except json.JSONDecodeError:
                    pass

        return {
            'trial_number': trial_number,
            'loss': loss,
            'algorithm': algorithm,
            'mesh_identifier': mesh_identifier,
            'cut_heights': cut_heights,
        }


class LiveBestTrialViewer(GeometryPersistenceMixin, QMainWindow):
    """
    A viewer that displays the best trial's aligned mesh during optimization.

    Monitors the ongoing CSV file and updates the display when a new best trial
    is found. Shows the aligned mesh (solid) and target mesh (wireframe).
    """

    GEOMETRY_KEY = "live_viewer"
    DEFAULT_GEOMETRY = None

    # Default polling interval in seconds
    DEFAULT_POLL_INTERVAL = 10
    MIN_POLL_INTERVAL = 2
    MAX_POLL_INTERVAL = 60

    # Target mesh wireframe color (green)
    TARGET_COLOR = [0, 1, 0]

    def __init__(
        self,
        results_folder: Path | str,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the Live preview.

        Args:
            results_folder: Path to the optimization results folder
            parent: Parent widget (optional)
        """
        super().__init__(parent)

        self.results_folder = Path(results_folder)
        self._is_closed = False

        # Track the ongoing CSV file for the current run
        self._current_ongoing_csv: Optional[Path] = None

        # State tracking - best result across the ENTIRE study
        self.current_best_trial: Optional[int] = None
        self.current_best_loss: Optional[float] = None
        self.current_best_algorithm: Optional[str] = None
        self.current_mesh_identifier: Optional[str] = None
        self.current_best_source: Optional[str] = None  # Which CSV the best came from
        self.auto_refresh_enabled = True

        # History of best trials (session only)
        self._best_history: list[dict] = []
        self._history_index: int = -1
        self._viewing_latest: bool = True

        # UI components
        self.viewer: Optional[VispyMeshViewer] = None
        self.status_label: Optional[QLabel] = None
        self.refresh_button: Optional[QPushButton] = None
        self.auto_refresh_button: Optional[QPushButton] = None
        self.interval_spinbox: Optional[QSpinBox] = None
        self.prev_button: Optional[QPushButton] = None
        self.next_button: Optional[QPushButton] = None
        self.latest_button: Optional[QPushButton] = None
        self.nav_label: Optional[QLabel] = None
        self.history_info_label: Optional[QLabel] = None

        # Background poll worker
        self._poll_worker: Optional[_PollWorker] = None

        # Timer for polling
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._start_poll)

        self._setup_ui()
        self._init_geometry_persistence()

        # Initial check
        QTimer.singleShot(500, self._start_poll)

        # Start auto-refresh
        self._start_auto_refresh()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Live preview - Best trial")
        self.setMinimumSize(600, 500)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Top control bar
        control_bar = QHBoxLayout()
        control_bar.setSpacing(10)

        # Status label
        self.status_label = QLabel("Waiting for trials...")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        font = QFont()
        font.setPointSize(10)
        self.status_label.setFont(font)
        control_bar.addWidget(self.status_label, stretch=1)

        # Refresh interval
        interval_label = QLabel("Auto refresh:")
        control_bar.addWidget(interval_label)

        # Auto-refresh toggle
        self.auto_refresh_button = QPushButton("Enabled")
        self.auto_refresh_button.setCheckable(True)
        self.auto_refresh_button.setChecked(True)
        self.auto_refresh_button.setToolTip("Toggle automatic refresh")
        self.auto_refresh_button.clicked.connect(self._toggle_auto_refresh)
        self.auto_refresh_button.setMinimumWidth(80)
        control_bar.addWidget(self.auto_refresh_button)

        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(self.MIN_POLL_INTERVAL, self.MAX_POLL_INTERVAL)
        self.interval_spinbox.setValue(self.DEFAULT_POLL_INTERVAL)
        self.interval_spinbox.setSuffix("s")
        self.interval_spinbox.setStyleSheet(
            "QSpinBox::up-button, QSpinBox::down-button { width: 16px; }"
        )
        self.interval_spinbox.setToolTip("Auto-refresh interval in seconds")
        self.interval_spinbox.valueChanged.connect(self._on_interval_changed)
        control_bar.addWidget(self.interval_spinbox)

        # Manual refresh button
        self.refresh_button = QPushButton("Refresh now")
        self.refresh_button.setToolTip("Manually check for updates")
        self.refresh_button.clicked.connect(self._start_poll)
        control_bar.addWidget(self.refresh_button)

        layout.addLayout(control_bar)

        # History navigation bar
        nav_bar = QHBoxLayout()
        nav_bar.setSpacing(6)

        self.latest_button = QPushButton("Go to latest")
        self.latest_button.setToolTip("Jump to latest best (Up arrow)")
        self.latest_button.setEnabled(False)
        self.latest_button.clicked.connect(self._go_to_latest)
        nav_bar.addWidget(self.latest_button)

        self.next_button = QPushButton("\u25C0 Next")
        self.next_button.setToolTip("Next/newer best (Left arrow)")
        self.next_button.setEnabled(False)
        self.next_button.setFixedWidth(70)
        self.next_button.clicked.connect(self._go_next)
        nav_bar.addWidget(self.next_button)

        self.nav_label = QLabel("")
        self.nav_label.setAlignment(Qt.AlignCenter)
        self.nav_label.setFixedWidth(50)
        nav_bar.addWidget(self.nav_label)

        self.prev_button = QPushButton("Prev \u25B6")
        self.prev_button.setToolTip("Previous/older best (Right arrow)")
        self.prev_button.setEnabled(False)
        self.prev_button.setFixedWidth(70)
        self.prev_button.clicked.connect(self._go_previous)
        nav_bar.addWidget(self.prev_button)

        self.history_info_label = QLabel("")
        self.history_info_label.setStyleSheet("color: gray; padding-left: 8px;")
        nav_bar.addWidget(self.history_info_label, stretch=1)

        layout.addLayout(nav_bar)

        # Create VisPy viewer
        self.viewer = VispyMeshViewer(
            parent_handler=self,
            title="Live preview",
            bgcolor='black'
        )
        # Embed the VisPy canvas in our layout
        layout.addWidget(self.viewer.native, stretch=1)

        # Bottom info bar
        self.info_label = QLabel("Monitoring: " + str(self.results_folder.name))
        self.info_label.setStyleSheet("color: gray; font-size: 9px;")
        layout.addWidget(self.info_label)

    def _start_auto_refresh(self) -> None:
        """Start the auto-refresh timer."""
        interval_ms = self.interval_spinbox.value() * 1000
        self.poll_timer.start(interval_ms)
        self.auto_refresh_enabled = True
        self.auto_refresh_button.setText("Enabled")
        self.auto_refresh_button.setChecked(True)

    def _stop_auto_refresh(self) -> None:
        """Stop the auto-refresh timer."""
        self.poll_timer.stop()
        self.auto_refresh_enabled = False
        self.auto_refresh_button.setText("Disabled")
        self.auto_refresh_button.setChecked(False)

    def _toggle_auto_refresh(self) -> None:
        """Toggle auto-refresh on/off."""
        if self.auto_refresh_enabled:
            self._stop_auto_refresh()
        else:
            self._start_auto_refresh()

    def _on_interval_changed(self, value: int) -> None:
        """Handle refresh interval change."""
        if self.auto_refresh_enabled:
            self.poll_timer.setInterval(value * 1000)

    def _start_poll(self) -> None:
        """Kick off a background poll worker (skips if one is already running)."""
        if self._is_closed:
            return
        if self._poll_worker is not None and self._poll_worker.isRunning():
            return  # Previous poll still in progress

        # Disconnect old worker's signal to prevent stale callbacks
        if self._poll_worker is not None:
            try:
                self._poll_worker.result_ready.disconnect(self._on_poll_result)
            except RuntimeError:
                pass  # Already disconnected

        self._poll_worker = _PollWorker(self.results_folder, parent=self)
        self._poll_worker.result_ready.connect(self._on_poll_result)
        self._poll_worker.start()

    @Slot(object)
    def _on_poll_result(self, best: Optional[dict]) -> None:
        """Handle the result from the background poll worker (runs on main thread)."""
        if self._is_closed:
            return

        if best is None:
            if self.current_best_loss is not None:
                self._update_status_idle()
            else:
                self.status_label.setText("No optimization data found in this folder.")
            return

        is_new_study_best = (
            self.current_best_loss is None or
            best['loss'] < self.current_best_loss
        )

        if is_new_study_best:
            is_historical = self.current_best_loss is None
            self._update_display(best, is_historical=is_historical)
            self._update_status_new_best(best)
        else:
            self._update_status_monitoring(best)

    def _get_aligned_mesh_path(self, algorithm: str, mesh_identifier: str) -> Optional[Path]:
        """Get the path to the aligned mesh file."""
        folder_name = f"{algorithm}_aligned_meshes"
        filename = f"{algorithm}_aligned_{mesh_identifier}.ply"
        mesh_path = self.results_folder / folder_name / filename

        if mesh_path.exists():
            return mesh_path
        return None

    def _get_target_mesh_path(self, algorithm: str, cut_heights: Optional[dict]) -> Optional[Path]:
        """Get the path to the target mesh file."""
        # Check if this algorithm uses cut targets
        if algorithm_cuts_target(algorithm) and cut_heights:
            cut_height = cut_heights.get(algorithm)
            if cut_height is not None:
                filename = get_cut_mesh_filename(cut_height)
                cut_path = self.results_folder / "cut_target_meshes" / filename
                if cut_path.exists():
                    return cut_path

        # Fall back to original target
        original_folder = self.results_folder / "original_target_mesh"
        if original_folder.exists():
            files = list(original_folder.iterdir())
            if files:
                return files[0]

        return None

    def _poll_for_updates(self) -> None:
        """
        Check for new best trial across the entire study and update display if needed.

        On first run, loads the historical best from completed CSVs.
        Then monitors the ongoing CSV for new trials that beat the study best.
        """
        if self._is_closed:
            return

        # First time: load historical best from completed runs
        if self.current_best_loss is None:
            self.status_label.setText("Loading study history...")
            historical_best = self._load_study_best()
            if historical_best:
                logger.info(f"[Live preview] Found study best: Trial #{historical_best['trial_number']}, "
                      f"loss={historical_best['loss']:.2f} from {historical_best.get('source', 'unknown')}")
                self._update_display(historical_best, is_historical=True)
            else:
                logger.info("[Live preview] No historical best found in completed CSVs")

        # Check the ongoing CSV for new trials
        ongoing_csv = self._find_ongoing_csv()

        if not ongoing_csv:
            # No ongoing optimization, just show current best if we have one
            if self.current_best_loss is not None:
                self._update_status_idle()
            else:
                self.status_label.setText("No optimization data found in this folder.")
            return

        # Track if ongoing CSV changed (new optimization run started)
        if self._current_ongoing_csv != ongoing_csv:
            self._current_ongoing_csv = ongoing_csv
            logger.info(f"Monitoring ongoing CSV: {ongoing_csv.name}")

        df = self._load_csv_data(ongoing_csv)
        if df is None or df.empty:
            if self.current_best_loss is not None:
                self._update_status_monitoring()
            else:
                self.status_label.setText("Waiting for completed trials...")
            return

        # Find the best in the current ongoing CSV
        ongoing_best = self._find_best_trial(df)
        if not ongoing_best:
            if self.current_best_loss is not None:
                self._update_status_monitoring()
            else:
                self.status_label.setText("Waiting for completed trials...")
            return

        # Count trials for status
        n_ongoing_trials = len(df[df['state'] == 'COMPLETE']) if 'state' in df.columns else len(df)

        # Check if the ongoing best beats our current study best
        is_new_study_best = (
            self.current_best_loss is None or
            ongoing_best['loss'] < self.current_best_loss
        )

        if is_new_study_best:
            ongoing_best['source'] = ongoing_csv.name
            self._update_display(ongoing_best, is_historical=False)
            self._update_status_new_best(ongoing_best)
        else:
            # Current run hasn't beaten the study best yet
            self._update_status_monitoring(ongoing_best, n_ongoing_trials)

    def _update_status_idle(self) -> None:
        """Update status when no ongoing optimization is detected."""
        self.status_label.setText(
            f"Study best: Trial #{self.current_best_trial} | "
            f"Loss: {self.current_best_loss:.2f} | "
            f"No active optimization"
        )

    def _update_status_monitoring(self, ongoing_best: Optional[dict] = None) -> None:
        """Update status when monitoring but study best hasn't been beaten."""
        if ongoing_best:
            self.status_label.setText(
                f"Study best: {self.current_best_loss:.2f} (Trial #{self.current_best_trial}) | "
                f"Current run best: {ongoing_best['loss']:.2f}"
            )
        else:
            self.status_label.setText(
                f"Study best: {self.current_best_loss:.2f} (Trial #{self.current_best_trial}) | "
                f"Monitoring for new trials..."
            )

    def _update_status_new_best(self, best: dict) -> None:
        """Update status when a new study best was just found."""
        self.status_label.setText(
            f"Best so far: Trial #{best['trial_number']} | "
            f"Loss: {best['loss']:.2f} | "
            f"Alignment: {best['algorithm']}"
        )

    def _update_display(self, best: dict, is_historical: bool = False) -> None:
        """
        Update the viewer with the best trial.

        Args:
            best: Dict with trial info (trial_number, loss, algorithm, mesh_identifier, etc.)
            is_historical: True if this is from a completed CSV (previous run)
        """
        algorithm = best['algorithm']
        mesh_identifier = best['mesh_identifier']

        # Get mesh paths
        aligned_path = self._get_aligned_mesh_path(algorithm, mesh_identifier)
        target_path = self._get_target_mesh_path(algorithm, best.get('cut_heights'))

        if not aligned_path or not target_path:
            self.status_label.setText(f"Trial #{best['trial_number']} - Meshes not yet saved...")
            return

        # Only load the mesh visually if the user is following the latest
        if self._viewing_latest:
            success = self.viewer.load_and_display_aligned_with_target(
                str(aligned_path),
                str(target_path),
                self.TARGET_COLOR
            )
        else:
            # Still record the entry but don't change the display
            success = True

        if success:
            self.current_best_trial = best['trial_number']
            self.current_best_loss = best['loss']
            self.current_best_algorithm = algorithm
            self.current_mesh_identifier = mesh_identifier
            self.current_best_source = best.get('source', 'unknown')

            # Record in history
            history_entry = dict(best)
            history_entry['aligned_path'] = str(aligned_path)
            history_entry['target_path'] = str(target_path)
            history_entry['is_historical'] = is_historical
            self._best_history.append(history_entry)

            if self._viewing_latest:
                self._history_index = len(self._best_history) - 1

                # Update window title
                source_info = "(previous run)" if is_historical else "(current run)"
                self.setWindowTitle(
                    f"Live preview - Study best: {best['loss']:.2f} "
                    f"(Trial #{best['trial_number']}) {source_info}"
                )

                # Fit view for new mesh
                self.viewer.fit_to_meshes()

                # Update status
                if is_historical:
                    self.status_label.setText(
                        f"Study best from previous run: Trial #{best['trial_number']} | "
                        f"Loss: {best['loss']:.2f} | "
                        f"Alignment: {best['algorithm']} | "
                        f"Monitoring for improvements..."
                    )

            self._update_nav_ui()
        else:
            self.status_label.setText(f"Error loading mesh from trial #{best['trial_number']}")

    # ------------------------------------------------------------------
    # History navigation
    # ------------------------------------------------------------------

    def _go_previous(self) -> None:
        """Navigate to the previous best in history."""
        if self._history_index > 0:
            self._viewing_latest = False
            self._display_history_entry(self._history_index - 1)

    def _go_next(self) -> None:
        """Navigate to the next best in history."""
        if self._history_index < len(self._best_history) - 1:
            self._display_history_entry(self._history_index + 1)
            if self._history_index == len(self._best_history) - 1:
                self._viewing_latest = True

    def _go_to_latest(self) -> None:
        """Jump to the most recent best in history."""
        if self._best_history:
            self._viewing_latest = True
            self._display_history_entry(len(self._best_history) - 1)

    def _display_history_entry(self, index: int) -> None:
        """Load and display a specific history entry by index."""
        if index < 0 or index >= len(self._best_history):
            return

        entry = self._best_history[index]
        success = self.viewer.load_and_display_aligned_with_target(
            entry['aligned_path'],
            entry['target_path'],
            self.TARGET_COLOR
        )

        if success:
            self._history_index = index
            self.viewer.fit_to_meshes()
            self._update_history_title(entry)
            self._update_nav_ui()

    def _update_nav_ui(self) -> None:
        """Refresh navigation buttons and labels based on current history state."""
        n = len(self._best_history)
        has_history = n > 0

        self.prev_button.setEnabled(has_history and self._history_index > 0)
        self.next_button.setEnabled(has_history and self._history_index < n - 1)
        self.latest_button.setEnabled(has_history and not self._viewing_latest)

        if has_history:
            self.nav_label.setText(f"{self._history_index + 1} / {n}")
            entry = self._best_history[self._history_index]
            self.history_info_label.setText(
                f"Trial #{entry['trial_number']},  "
                f"Loss: {entry['loss']:.2f},  "
                f"{entry['algorithm']}"
            )
        else:
            self.nav_label.setText("")
            self.history_info_label.setText("")

    def _update_history_title(self, entry: dict) -> None:
        """Update the window title to reflect the currently viewed history entry."""
        n = len(self._best_history)
        pos = self._history_index + 1
        if self._viewing_latest:
            source_info = "(previous run)" if entry.get('is_historical') else "(current run)"
            self.setWindowTitle(
                f"Live preview - Study best: {entry['loss']:.2f} "
                f"(Trial #{entry['trial_number']}) {source_info}"
            )
        else:
            self.setWindowTitle(
                f"Live preview - #{pos}/{n}: Trial #{entry['trial_number']}, "
                f"Loss: {entry['loss']:.2f} ({entry['algorithm']})"
            )

    def handle_key_press(self, key: str) -> None:
        """Handle key events from the VisPy viewer."""
        if key == 'Escape':
            self.close()
        elif key == 'Left':
            self._go_next()
        elif key == 'Right':
            self._go_previous()
        elif key == 'Up':
            self._go_to_latest()
        elif key == 'R':
            self._start_poll()
        elif key == 'Space':
            self._toggle_auto_refresh()
        else:
            # Let the viewer handle other keys (X, Y, Z, F, W, S, etc.)
            if self.viewer:
                self.viewer._handle_viewer_key(key)

    def closeEvent(self, event) -> None:
        """Handle window close."""
        self._is_closed = True
        self.poll_timer.stop()

        # Clean up poll worker
        if self._poll_worker is not None and self._poll_worker.isRunning():
            if not self._poll_worker.wait(500):
                self._poll_worker.finished.connect(self._poll_worker.deleteLater)
                self._poll_worker.setParent(None)

        if self.viewer:
            try:
                self.viewer.close()
            except Exception:
                pass

        super().closeEvent(event)


def launch_live_viewer(results_folder: Path | str, parent: Optional[QWidget] = None) -> Optional[LiveBestTrialViewer]:
    """
    Launch a Live preview for the given results folder.

    Args:
        results_folder: Path to the optimization results folder
        parent: Parent widget (optional)

    Returns:
        LiveBestTrialViewer instance, or None if launch failed
    """
    results_folder = Path(results_folder)

    if not results_folder.exists():
        QMessageBox.warning(
            parent,
            "Folder Not Found",
            f"Results folder does not exist:\n{results_folder}"
        )
        return None

    try:
        viewer = LiveBestTrialViewer(results_folder, parent)
        viewer.show()
        return viewer
    except Exception as e:
        logger.error(f"Failed to launch Live preview: {e}")
        QMessageBox.critical(
            parent,
            "Error",
            f"Failed to launch Live preview:\n{e}"
        )
        return None


# Standalone test
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    # Test with a folder path argument
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        print("Usage: python live_viewer.py <results_folder>")
        sys.exit(1)

    viewer = launch_live_viewer(folder)
    if viewer:
        sys.exit(app.exec())
    else:
        sys.exit(1)
