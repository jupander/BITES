"""
Optuna Plot Dialog - UI for selecting and generating Optuna visualization plots.

This module provides a dialog for:
- Selecting which plots to generate (with computational cost indicators)
- Showing progress during generation
- Displaying results and any errors
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

from PySide6.QtCore import Slot, QThread, Signal, QUrl
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QProgressBar, QGroupBox, QWidget,
    QFrame, QTextEdit, QMessageBox
)
from PySide6.QtGui import QDesktopServices

from bites.gui.visualization.optuna_plots.worker import (
    OptunaPlotWorker, PlotResult, PlotInfo, PlotCost,
    AVAILABLE_PLOTS, DEFAULT_PLOT_ORDER, get_available_plots
)

logger = logging.getLogger(__name__)


def load_study_info(output_folder: Path) -> Dict:
    """
    Load study information for validation purposes.

    Returns:
        Dict with keys: study_name, n_trials, n_complete, n_params, error
    """
    import optuna
    from bites.config.config_loader import Config
    from bites.database.pg_db_manager import ensure_database_for_target

    result = {
        "study_name": None,
        "n_trials": 0,
        "n_complete": 0,
        "n_params": 0,
        "error": None,
    }

    try:
        # Find the most recent config file
        config_files = list(output_folder.glob("config_*.yaml"))
        if not config_files:
            result["error"] = "No config file found in output folder"
            return result

        config_file = sorted(config_files)[-1]
        config = Config.from_yaml(str(config_file))
        result["study_name"] = config.study_name

        # Get storage URL
        if config.storage_mode == "shared":
            storage_url = ensure_database_for_target(config)
        else:
            result["error"] = f"Unsupported storage mode: {config.storage_mode}"
            return result

        # Load study
        study = optuna.load_study(
            study_name=config.study_name,
            storage=storage_url
        )

        result["n_trials"] = len(study.trials)
        result["n_complete"] = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])

        # Get number of parameters from first complete trial
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result["n_params"] = len(trial.params)
                break

    except Exception as e:
        logger.error(f"Failed to load study info: {e}", exc_info=True)
        result["error"] = str(e)

    return result


def get_existing_plots(output_folder: Path) -> Dict[str, Optional[datetime]]:
    """
    Check for existing plot files and their modification times.

    Args:
        output_folder: Path to the optimization results folder

    Returns:
        Dict mapping plot_id to modification datetime (or None if not found)
    """
    vis_dir = output_folder / "optuna_visualizations"
    existing = {}

    for plot_id in AVAILABLE_PLOTS.keys():
        filepath = vis_dir / f"{plot_id}.html"
        if filepath.exists():
            try:
                mtime = filepath.stat().st_mtime
                existing[plot_id] = datetime.fromtimestamp(mtime)
            except Exception:
                existing[plot_id] = None  # File exists but can't get time
        else:
            existing[plot_id] = None

    return existing


def format_timestamp(dt: Optional[datetime]) -> str:
    """Format a datetime for display."""
    if dt is None:
        return "Not generated"

    now = datetime.now()
    diff = now - dt

    # If today, show time only
    if dt.date() == now.date():
        return f"Today at {dt.strftime('%H:%M')}"
    # If yesterday
    elif diff.days == 1:
        return f"Yesterday at {dt.strftime('%H:%M')}"
    # If within a week
    elif diff.days < 7:
        return f"{diff.days} days ago"
    # Otherwise show date
    else:
        return dt.strftime("%Y-%m-%d %H:%M")


class StudyInfoWorker(QThread):
    """Background worker for loading study info without blocking the UI."""

    finished_loading = Signal(dict)

    def __init__(self, output_folder: Path, parent=None):
        super().__init__(parent)
        self.output_folder = output_folder

    def run(self) -> None:
        result = load_study_info(self.output_folder)
        self.finished_loading.emit(result)


class PlotRowWidget(QWidget):
    """Widget containing checkbox, existing plot info, and open button for a single plot."""

    def __init__(self, plot_info: PlotInfo, existing_time: Optional[datetime] = None, parent=None):
        super().__init__(parent)
        self.plot_info = plot_info
        self.existing_time = existing_time
        self.file_path: Optional[Path] = None  # Set when plot exists

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        # Checkbox
        self.checkbox = QCheckBox()
        cost_text = f" [{plot_info.cost_indicator}]"
        self.checkbox.setText(f"{plot_info.display_name}{cost_text}")

        # Set tooltip with description
        self.checkbox.setToolTip(
            f"{plot_info.description}\n\n"
            f"Speed: {plot_info.cost_indicator}\n"
            f"Requires: {plot_info.min_trials}+ completed trials"
        )

        # Style based on cost
        if plot_info.cost == PlotCost.VERY_SLOW:
            self.checkbox.setStyleSheet("QCheckBox { color: #cc6600; }")

        layout.addWidget(self.checkbox)

        # Existing plot info label
        self.existing_label = QLabel()
        self.existing_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.existing_label)

        # Open button (hidden by default, shown when plot exists)
        self.open_btn = QPushButton("Open")
        self.open_btn.setMaximumWidth(50)
        self.open_btn.setMaximumHeight(22)
        self.open_btn.setToolTip("Open this plot in your browser")
        self.open_btn.clicked.connect(self._open_plot)
        self.open_btn.setVisible(False)
        layout.addWidget(self.open_btn)

        layout.addStretch()

        self._update_existing_label()

    def _update_existing_label(self) -> None:
        """Update the existing plot label text and open button visibility."""
        if self.existing_time:
            time_str = format_timestamp(self.existing_time)
            self.existing_label.setText(f"(exists: {time_str})")
            self.existing_label.setStyleSheet("color: #006600; font-size: 11px;")  # Green for existing
            self.open_btn.setVisible(True)
        else:
            self.existing_label.setText("")
            self.open_btn.setVisible(False)

    def _open_plot(self) -> None:
        """Open the existing plot in the browser."""
        if self.file_path and self.file_path.exists():
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.file_path)))
            except Exception as e:
                logger.warning(f"Could not open browser: {e}")

    def set_existing_time(self, dt: Optional[datetime], file_path: Optional[Path] = None) -> None:
        """Update the existing time and file path."""
        self.existing_time = dt
        self.file_path = file_path
        self._update_existing_label()

    def isChecked(self) -> bool:
        return self.checkbox.isChecked()

    def setChecked(self, checked: bool) -> None:
        self.checkbox.setChecked(checked)

    def isEnabled(self) -> bool:
        return self.checkbox.isEnabled()

    def setEnabled(self, enabled: bool) -> None:
        self.checkbox.setEnabled(enabled)
        self.existing_label.setEnabled(enabled)
        # Keep open button enabled even if checkbox is disabled
        # so user can still view existing plots

    def setToolTip(self, text: str) -> None:
        self.checkbox.setToolTip(text)


class OptunaPlotDialog(QDialog):
    """
    Dialog for selecting and generating Optuna visualization plots.

    Provides two phases:
    1. Selection phase: User selects which plots to generate
    2. Progress phase: Shows generation progress with cancel option
    """

    def __init__(
        self,
        output_folder: Path,
        parent=None
    ):
        """
        Initialize the dialog.

        Args:
            output_folder: Path to the optimization results folder
            parent: Parent widget
        """
        super().__init__(parent)
        self.output_folder = Path(output_folder).resolve()
        self.worker: Optional[OptunaPlotWorker] = None
        self.results: List[PlotResult] = []
        self._results_shown = False
        self._results_lock = threading.Lock()
        self.checkboxes: Dict[str, PlotRowWidget] = {}

        self.setWindowTitle("Generate Optuna plots")
        self.setMinimumWidth(500)
        self.setMinimumHeight(450)

        # Study info (populated during validation)
        self.study_info: Optional[Dict] = None
        self.existing_plots: Dict[str, Optional[datetime]] = {}

        self._init_ui()
        self._validate_and_show()

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        self.main_layout = QVBoxLayout(self)

        # Header
        header_label = QLabel(
            "<b style='font-size: 14px;'>Generate Optuna visualization plots</b><br>"
            "Select which plots to generate. Plots are saved as HTML files and opened in your browser."
        )
        header_label.setWordWrap(True)
        self.main_layout.addWidget(header_label)

        # Folder info
        folder_label = QLabel(f"Output folder: {self.output_folder.name}")
        folder_label.setStyleSheet("color: gray;")
        self.main_layout.addWidget(folder_label)

        # Study info (updated during validation)
        self.study_info_label = QLabel("Loading study information...")
        self.study_info_label.setStyleSheet("color: gray;")
        self.main_layout.addWidget(self.study_info_label)

        # Existing plots info (updated during validation)
        self.existing_plots_label = QLabel("")
        self.existing_plots_label.setWordWrap(True)
        self.main_layout.addWidget(self.existing_plots_label)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(line)

        # === Selection Phase Widgets ===
        self.selection_widget = QWidget()
        selection_layout = QVBoxLayout(self.selection_widget)
        selection_layout.setContentsMargins(0, 0, 0, 0)

        # Plot selection group
        selection_group = QGroupBox("Select plots to generate:")
        selection_group_layout = QVBoxLayout(selection_group)

        # Quick select buttons
        quick_select_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select all")
        select_all_btn.clicked.connect(self._select_all)
        select_fast_btn = QPushButton("Select fast only")
        select_fast_btn.clicked.connect(self._select_fast_only)
        select_none_btn = QPushButton("Select none")
        select_none_btn.clicked.connect(self._select_none)

        quick_select_layout.addWidget(select_all_btn)
        quick_select_layout.addWidget(select_fast_btn)
        quick_select_layout.addWidget(select_none_btn)
        quick_select_layout.addStretch()
        selection_group_layout.addLayout(quick_select_layout)

        # Plot rows (checkbox + existing info) for each plot
        for plot_id in DEFAULT_PLOT_ORDER:
            plot_info = AVAILABLE_PLOTS[plot_id]
            row_widget = PlotRowWidget(plot_info)

            # Check fast/medium by default, leave slow unchecked
            row_widget.setChecked(plot_info.cost in (PlotCost.FAST, PlotCost.MEDIUM))

            self.checkboxes[plot_id] = row_widget
            selection_group_layout.addWidget(row_widget)

        selection_layout.addWidget(selection_group)

        # Open in browser option
        self.open_browser_checkbox = QCheckBox("Open plots in browser after generation")
        self.open_browser_checkbox.setChecked(True)
        self.open_browser_checkbox.setToolTip(
            "When enabled, each plot will be opened in your default web browser after being saved."
        )
        selection_layout.addWidget(self.open_browser_checkbox)

        # Warning for slow plots
        self.slow_warning_label = QLabel(
            "<i>Note: 'Parameter Importances' can take several minutes for large studies.</i>"
        )
        self.slow_warning_label.setStyleSheet("color: #cc6600; font-size: 11px;")
        self.slow_warning_label.setWordWrap(True)
        selection_layout.addWidget(self.slow_warning_label)

        selection_layout.addStretch()

        # Generate button
        self.generate_btn = QPushButton("Generate selected plots")
        self.generate_btn.setMinimumHeight(35)
        self.generate_btn.clicked.connect(self._start_generation)
        selection_layout.addWidget(self.generate_btn)

        self.main_layout.addWidget(self.selection_widget)

        # === Progress Phase Widgets ===
        self.progress_widget = QWidget()
        progress_layout = QVBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        # Current plot label
        self.current_plot_label = QLabel("Preparing...")
        self.current_plot_label.setStyleSheet("font-weight: bold;")
        progress_layout.addWidget(self.current_plot_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        # Status log
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(150)
        self.status_log.setStyleSheet("background-color: palette(window);")
        progress_layout.addWidget(self.status_log)

        progress_layout.addStretch()

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_generation)
        progress_layout.addWidget(self.cancel_btn)

        self.main_layout.addWidget(self.progress_widget)

        # === Results Phase Widgets ===
        self.results_widget = QWidget()
        results_layout = QVBoxLayout(self.results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)

        self.results_label = QLabel()
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)

        # Results details
        self.results_details = QTextEdit()
        self.results_details.setReadOnly(True)
        self.results_details.setStyleSheet("background-color: palette(window);")
        results_layout.addWidget(self.results_details)

        results_layout.addStretch()

        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        results_layout.addWidget(self.close_btn)

        self.main_layout.addWidget(self.results_widget)

    def _show_selection_phase(self) -> None:
        """Show the plot selection phase."""
        self.selection_widget.setVisible(True)
        self.progress_widget.setVisible(False)
        self.results_widget.setVisible(False)

    def _show_progress_phase(self) -> None:
        """Show the progress phase."""
        self.selection_widget.setVisible(False)
        self.progress_widget.setVisible(True)
        self.results_widget.setVisible(False)
        self.status_log.clear()

    def _show_results_phase(self) -> None:
        """Show the results phase."""
        self.selection_widget.setVisible(False)
        self.progress_widget.setVisible(False)
        self.results_widget.setVisible(True)

    def _validate_and_show(self) -> None:
        """Validate study access and show appropriate phase."""
        logger.info(f"Validating study in {self.output_folder}")

        # Load existing plots first (doesn't require database - fast filesystem check)
        self.existing_plots = get_existing_plots(self.output_folder)
        existing_count = sum(1 for v in self.existing_plots.values() if v is not None)

        # Update existing plots label
        if existing_count > 0:
            # Find the most recent plot time
            existing_times = [v for v in self.existing_plots.values() if v is not None]
            most_recent = max(existing_times) if existing_times else None
            time_str = format_timestamp(most_recent) if most_recent else ""

            self.existing_plots_label.setText(
                f"<span style='color: #006600;'>{existing_count} plot(s) already exist "
                f"(most recent: {time_str})</span>"
            )
        else:
            self.existing_plots_label.setText(
                "<span style='color: gray;'>No existing plots found</span>"
            )

        # Update row widgets with existing plot times and file paths
        vis_dir = self.output_folder / "optuna_visualizations"
        for plot_id, row_widget in self.checkboxes.items():
            existing_time = self.existing_plots.get(plot_id)
            file_path = vis_dir / f"{plot_id}.html" if existing_time else None
            row_widget.set_existing_time(existing_time, file_path)

        # Load study info in a background thread to avoid freezing the UI
        # (database connections can hang if the server is unreachable)
        self.generate_btn.setEnabled(False)
        self.study_info_label.setText("Loading study information...")
        self._show_selection_phase()

        self._study_info_worker = StudyInfoWorker(self.output_folder, parent=self)
        self._study_info_worker.finished_loading.connect(self._on_study_info_loaded)
        self._study_info_worker.start()

    @Slot(dict)
    def _on_study_info_loaded(self, study_info: dict) -> None:
        """Handle study info loaded from background thread."""
        self.study_info = study_info

        if self.study_info["error"]:
            self.study_info_label.setText(
                f"<span style='color: red;'>Error: {self.study_info['error']}</span>"
            )
            QMessageBox.critical(
                self,
                "Cannot Load Study",
                f"Failed to load the optimization study:\n\n{self.study_info['error']}\n\n"
                "Please check that the database is accessible."
            )
            self.generate_btn.setEnabled(False)
            return

        # Update study info label
        self.study_info_label.setText(
            f"Study: {self.study_info['study_name']} | "
            f"Trials: {self.study_info['n_complete']} completed / {self.study_info['n_trials']} total | "
            f"Parameters: {self.study_info['n_params']}"
        )

        # Update checkboxes based on available trials
        n_complete = self.study_info["n_complete"]
        for plot_id, row_widget in self.checkboxes.items():
            plot_info = AVAILABLE_PLOTS[plot_id]
            if n_complete < plot_info.min_trials:
                row_widget.setEnabled(False)
                row_widget.setChecked(False)
                row_widget.setToolTip(
                    f"{plot_info.description}\n\n"
                    f"DISABLED: Requires {plot_info.min_trials}+ completed trials "
                    f"(you have {n_complete})"
                )
            else:
                row_widget.setEnabled(True)

        # Check if any plots are available
        available_count = sum(
            1 for p in AVAILABLE_PLOTS.values()
            if n_complete >= p.min_trials
        )

        if available_count == 0:
            QMessageBox.warning(
                self,
                "Not Enough Trials",
                f"The study only has {n_complete} completed trial(s).\n\n"
                "At least 1 completed trial is needed for any visualizations.\n"
                "Run more optimization trials first."
            )
            self.generate_btn.setEnabled(False)
        else:
            self.generate_btn.setEnabled(True)

    def _select_all(self) -> None:
        """Select all enabled plot checkboxes."""
        for checkbox in self.checkboxes.values():
            if checkbox.isEnabled():
                checkbox.setChecked(True)

    def _select_fast_only(self) -> None:
        """Select only fast and medium plots (if enabled)."""
        for plot_id, checkbox in self.checkboxes.items():
            plot_info = AVAILABLE_PLOTS[plot_id]
            should_check = (
                checkbox.isEnabled() and
                plot_info.cost in (PlotCost.FAST, PlotCost.MEDIUM)
            )
            checkbox.setChecked(should_check)

    def _select_none(self) -> None:
        """Deselect all plot checkboxes."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)

    def _get_selected_plots(self) -> List[str]:
        """Get list of selected plot IDs."""
        return [
            plot_id for plot_id, checkbox in self.checkboxes.items()
            if checkbox.isChecked()
        ]

    def _start_generation(self) -> None:
        """Start generating the selected plots."""
        selected = self._get_selected_plots()

        if not selected:
            QMessageBox.warning(
                self,
                "No Plots Selected",
                "Please select at least one plot to generate."
            )
            return

        logger.info(f"Starting generation of {len(selected)} plots: {selected}")

        # Check for slow plots and warn
        slow_selected = [
            AVAILABLE_PLOTS[p].display_name for p in selected
            if AVAILABLE_PLOTS[p].cost in (PlotCost.SLOW, PlotCost.VERY_SLOW)
        ]
        if slow_selected:
            result = QMessageBox.question(
                self,
                "Slow Plots Selected",
                f"The following plots may take a long time to generate:\n\n"
                f"  - {chr(10).join(slow_selected)}\n\n"
                f"Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if result != QMessageBox.Yes:
                return

        # Switch to progress view
        self._show_progress_phase()
        self.progress_bar.setMaximum(len(selected))
        self.progress_bar.setValue(0)

        # Create and start worker
        self.worker = OptunaPlotWorker(
            output_folder=self.output_folder,
            selected_plots=selected,
            open_in_browser=self.open_browser_checkbox.isChecked(),
            parent=self
        )

        # Connect signals
        self.worker.started_plot.connect(self._on_plot_started)
        self.worker.completed_plot.connect(self._on_plot_completed)
        self.worker.all_completed.connect(self._on_all_completed)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)

        self.results = []
        with self._results_lock:
            self._results_shown = False
        self.worker.start()

    def _cancel_generation(self) -> None:
        """Cancel the ongoing generation."""
        if self.worker and self.worker.isRunning():
            self._log_status("Cancelling...")
            self.cancel_btn.setEnabled(False)
            self.worker.cancel()

    @Slot(str, int, int)
    def _on_plot_started(self, plot_id: str, current: int, total: int) -> None:
        """Handle plot generation started."""
        plot_info = AVAILABLE_PLOTS.get(plot_id)
        name = plot_info.display_name if plot_info else plot_id

        self.current_plot_label.setText(
            f"Generating: {name} ({current}/{total})"
        )
        self._log_status(f"Starting: {name}...")

    @Slot(object)
    def _on_plot_completed(self, result: PlotResult) -> None:
        """Handle single plot completion."""
        self.results.append(result)
        self.progress_bar.setValue(len(self.results))

        plot_info = AVAILABLE_PLOTS.get(result.plot_id)
        name = plot_info.display_name if plot_info else result.plot_id

        if result.success:
            self._log_status(f"  Completed: {name}")

            # Open in browser immediately if option is enabled
            if self.open_browser_checkbox.isChecked() and result.file_path:
                self._open_in_browser(result.file_path)
        else:
            self._log_status(f"  Failed: {name} - {result.error_message}")

    def _open_in_browser(self, file_path: Path) -> None:
        """Open a plot file in the default browser."""
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(file_path)))
            logger.debug(f"Opened in browser: {file_path}")
        except Exception as e:
            logger.warning(f"Could not open browser for {file_path}: {e}")

    @Slot(list)
    def _on_all_completed(self, results: List[PlotResult]) -> None:
        """Handle all plots completed."""
        self.results = results
        with self._results_lock:
            if self._results_shown:
                return
            self._results_shown = True
        self._show_results()

    @Slot(str)
    def _on_error(self, error_message: str) -> None:
        """Handle fatal error."""
        logger.error(f"Plot generation error: {error_message}")
        self._log_status(f"ERROR: {error_message}")

        QMessageBox.critical(
            self,
            "Error",
            f"Failed to generate plots:\n\n{error_message}"
        )

    @Slot()
    def _on_worker_finished(self) -> None:
        """Handle worker thread finished."""
        logger.info("Plot worker finished")
        show = False
        with self._results_lock:
            if self.results and not self._results_shown:
                self._results_shown = True
                show = True
        if show:
            self._show_results()
        elif not self.results:
            # Worker finished but no results (likely cancelled early)
            self._show_selection_phase()

    def _log_status(self, message: str) -> None:
        """Add a message to the status log."""
        self.status_log.append(message)
        # Scroll to bottom
        scrollbar = self.status_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _show_results(self) -> None:
        """Show the results summary."""
        self._show_results_phase()

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        # Summary label
        if failed:
            self.results_label.setText(
                f"<b>Generation Complete</b><br>"
                f"<span style='color: green;'>{len(successful)} succeeded</span>, "
                f"<span style='color: red;'>{len(failed)} failed</span>"
            )
        else:
            self.results_label.setText(
                f"<b>Generation Complete</b><br>"
                f"<span style='color: green;'>All {len(successful)} plots generated successfully!</span>"
            )

        # Details
        details = []

        if successful:
            details.append("=== Successful ===")
            for result in successful:
                plot_info = AVAILABLE_PLOTS.get(result.plot_id)
                name = plot_info.display_name if plot_info else result.plot_id
                details.append(f"  {name}")
                if result.file_path:
                    details.append(f"    Saved to: {result.file_path.name}")

        if failed:
            details.append("\n=== Failed ===")
            for result in failed:
                plot_info = AVAILABLE_PLOTS.get(result.plot_id)
                name = plot_info.display_name if plot_info else result.plot_id
                details.append(f"  {name}: {result.error_message}")

        # Output folder info
        if successful:
            vis_dir = self.output_folder / "optuna_visualizations"
            details.append(f"\nPlots saved in:\n  {vis_dir}")

        self.results_details.setPlainText("\n".join(details))

    def closeEvent(self, event) -> None:
        """Handle dialog close - cancel workers without blocking."""
        # Cancel plot generation worker
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            # Disconnect signals to prevent callbacks after dialog closes
            try:
                self.worker.started_plot.disconnect()
                self.worker.completed_plot.disconnect()
                self.worker.all_completed.disconnect()
                self.worker.error.disconnect()
                self.worker.finished.disconnect()
            except RuntimeError:
                pass
            if not self.worker.wait(500):
                # Worker didn't finish in time - detach so it can finish safely
                self.worker.finished.connect(self.worker.deleteLater)
                self.worker.setParent(None)

        # Cancel study info worker
        if hasattr(self, '_study_info_worker') and self._study_info_worker.isRunning():
            try:
                self._study_info_worker.finished_loading.disconnect()
            except RuntimeError:
                pass
            if not self._study_info_worker.wait(500):
                self._study_info_worker.finished.connect(self._study_info_worker.deleteLater)
                self._study_info_worker.setParent(None)

        super().closeEvent(event)
