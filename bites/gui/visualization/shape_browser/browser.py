"""
Enhanced Shape Browser - Visualize and browse aligned .off files with rugplots.

Integrates improved VisPy visualization with PySide6 GUI.
Business logic is separated into shape_browser_model.py.
"""
from __future__ import annotations

import logging
import sys
import traceback
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QGridLayout, QInputDialog, QMessageBox, QSplitter, QToolButton
)
from PySide6.QtGui import QTextCursor, QFont
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QMutex, QMutexLocker

import bites.gui.gui_utils as gui_utils
from bites.gui.gui_utils import ErrorNotifier

_logger = logging.getLogger(__name__)
import bites.gui.visualization.shape_browser.rugplot_canvas as rugplot_canvas
import bites.gui.visualization.shape_browser.utils as shape_browser_utils
from bites.gui.visualization.shape_browser.model import (
    ShapeBrowserModel,
    FileService,
    MeshDisplayInfo,
)
from bites.gui.visualization.viewers.vispy_mesh_viewer import VispyMeshViewer
from bites.constants import ShapeBrowserConfig as Config
from bites.gui.mixins import GeometryPersistenceMixin

if TYPE_CHECKING:
    import pandas as pd


class ShapeBrowserError(Exception):
    """Base exception for shape browser errors."""
    pass


class NoDataError(ShapeBrowserError):
    """Raised when there's no data to display."""
    pass


class ConsoleOutput(QThread):
    """
    Thread for redirecting console output to GUI.

    This thread exists to keep the GUI responsive while handling
    stdout/stderr redirection. The actual text emission happens
    via Qt signals from the write() method.
    """
    new_text = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = True
        self._mutex = QMutex()

    def run(self) -> None:
        """Qt signals work without an active run loop."""
        pass

    def write(self, text: str) -> None:
        """
        Emit text to the console widget.

        Thread-safe: checks running state before emitting.
        """
        should_emit = False
        with QMutexLocker(self._mutex):
            should_emit = self._running
        if should_emit:
            self.new_text.emit(str(text))

    def flush(self) -> None:
        """Required for file-like interface compatibility."""
        pass

    def stop(self) -> None:
        """Stop the thread safely."""
        with QMutexLocker(self._mutex):
            self._running = False
        self.quit()
        if not self.wait(Config.CONSOLE_THREAD_STOP_TIMEOUT):
            self.terminate()
            self.wait(Config.CONSOLE_THREAD_TERMINATE_TIMEOUT)


class ConsoleWidget(QTextEdit):
    """
    Console widget for displaying program output.

    Redirects stdout/stderr to display in the GUI.
    """

    def __init__(self, font_size: int, console_widget_height: int):
        super().__init__()
        self.setReadOnly(True)
        self.setMinimumHeight(50)

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._streams_redirected = False

        self.console_output = ConsoleOutput()
        self.console_output.new_text.connect(self._append_text)
        self.console_output.start()

        sys.stdout = self.console_output
        sys.stderr = self.console_output
        self._streams_redirected = True

        font = self.font()
        font.setFamily(Config.DEFAULT_FONT_FAMILY)
        font.setPointSize(font_size)
        self.setFont(font)

    def _append_text(self, text: str) -> None:
        """Append text to the console display."""
        self.moveCursor(QTextCursor.End)
        self.insertPlainText(text)
        self.moveCursor(QTextCursor.End)

    def _restore_streams(self) -> None:
        """Restore original stdout/stderr."""
        if self._streams_redirected:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self._streams_redirected = False

    def closeEvent(self, event) -> None:
        """Clean up on close."""
        self._restore_streams()
        if hasattr(self, 'console_output'):
            try:
                self.console_output.new_text.disconnect()
            except (RuntimeError, TypeError):
                pass
            self.console_output.stop()
        event.accept()


class RugConsoleWindow(GeometryPersistenceMixin, QMainWindow):
    """
    Main window holding rugplots and console output.

    Displays a grid of rugplot visualizations for numerical columns
    in the dataframe, plus a console widget for status output.
    """

    GEOMETRY_KEY = "rug_console"
    DEFAULT_GEOMETRY = None

    def __init__(
        self,
        dataframe: pd.DataFrame,
        font_size: int = Config.DEFAULT_FONT_SIZE,
        window_width: int = Config.RUGPLOT_WINDOW_WIDTH,
        window_height: int = Config.RUGPLOT_WINDOW_HEIGHT,
        console_widget_height: int = Config.CONSOLE_WIDGET_HEIGHT,
        excluded_columns: list[str] | None = None,
        exclude_algorithm_columns: bool = True
    ):
        super().__init__()
        self._visualizer_ref: weakref.ref | None = None
        self._is_closed = False

        self.df = dataframe.copy()

        columns_to_exclude = set(excluded_columns or Config.DEFAULT_EXCLUDED_COLUMNS)

        if exclude_algorithm_columns:
            alignment_algorithms = {
                'ST', 'CPD-only', 'ST-CPD', 'STatac', 'STatac-CPD',
                'ICP', 'STatac-AB-CPD', 'STatac-C-CPD', 'STatac-AB', 'STatac-C'
            }
            columns_to_exclude.update(alignment_algorithms)

        numerical_columns = [
            col for col in self.df.select_dtypes(include=[np.number]).columns
            if self.df[col].nunique() > 1 and col not in columns_to_exclude
        ]

        if len(numerical_columns) == 0:
            raise NoDataError("No numerical columns with variation to display in rugplots.")

        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)

        self._setup_instructions_label(font_size)

        self.splitter = QSplitter(Qt.Orientation.Vertical)

        rugplots_container = QWidget()
        self.grid_layout = QGridLayout(rugplots_container)

        self.plot_widgets: list[rugplot_canvas.RugPlotCanvas] = []
        self._create_rugplots(dataframe, numerical_columns)

        self.console_widget = ConsoleWidget(
            font_size=font_size,
            console_widget_height=console_widget_height
        )
        self.console_widget.setToolTip(
            'Console output showing status messages and trial information.\n\n'
            'Displays:\n'
            '• Currently viewed trial and its loss score\n'
            '• Parameter values for the current trial\n'
            '• File paths and mesh statistics\n'
            '• Help information (press H in visualizer)\n\n'
            'Drag the divider above to resize this area.'
        )

        self.splitter.addWidget(rugplots_container)
        self.splitter.addWidget(self.console_widget)
        self.splitter.setSizes([window_height - console_widget_height - 100, console_widget_height])

        self.layout.addWidget(self.splitter)

        self._setup_status_bar(font_size)

        self.setCentralWidget(self.central_widget)
        self.setWindowTitle('Shape browser - Rug plots and text console')

        self.adjustSize()
        self.move(Config.RUGPLOT_WINDOW_X, Config.RUGPLOT_WINDOW_Y)
        self._init_geometry_persistence()

    @property
    def visualizer(self) -> ResultsShapeVisualizer | None:
        """Get the visualizer (using weak reference)."""
        if self._visualizer_ref is None:
            return None
        return self._visualizer_ref()

    @visualizer.setter
    def visualizer(self, value: ResultsShapeVisualizer | None) -> None:
        """Set the visualizer using a weak reference."""
        if value is None:
            self._visualizer_ref = None
        else:
            self._visualizer_ref = weakref.ref(value)

    def _setup_instructions_label(self, font_size: int) -> None:
        """Create and configure the instructions label with guide button."""
        top_row = QHBoxLayout()

        instructions = "Press H to print help (click visualizer window for controls to work)"
        self.instructions_label = QLabel(instructions)
        self.instructions_label.setToolTip(
            'Keyboard shortcuts for the 3D visualizer window:\n\n'
            'NAVIGATION:\n'
            '• Left / Right Arrow: Navigate shapes\n'
            '• Up Arrow: Go to first shape\n'
            '• J: Jump to specific index\n'
            '• 0-9: Sort by property/alignment\n\n'
            'SLIDESHOW:\n'
            '• Space: Pause/Resume slideshow\n'
            '• +/-: Increase/decrease slideshow speed\n'
            '• R: Reverse slideshow direction\n\n'
            'VIEW CONTROLS:\n'
            '• X / Y / Z: Snap to axis views\n'
            '• F: Fit meshes to view\n'
            '• W: Toggle source wireframe\n'
            '• T: Toggle target wireframe\n'
            '• S: Cycle shading (smooth/flat/uniform)\n'
            '• D: Cycle diffuse intensity\n'
            '• G: Toggle specular highlights\n'
            '• B: Toggle background (black/white)\n'
            '• C: Toggle backface culling\n'
            '• A: Toggle coordinate axis\n\n'
            'ACTIONS:\n'
            '• Down Arrow: Save current mesh and screenshot\n'
            '• U: Toggle rugplot updating\n'
            '• H: Print full help to console\n\n'
            'Note: Click the 3D visualizer window first for keyboard controls to work.'
        )

        font = QFont()
        font.setFamily(Config.DEFAULT_FONT_FAMILY)
        font.setPointSize(font_size)
        self.instructions_label.setFont(font)

        top_row.addWidget(self.instructions_label)
        top_row.addStretch()

        guide_button = QToolButton()
        guide_button.setText("Guide")
        guide_button.setToolTip(
            "Open User's Guide\n\n"
            "Opens the PDF documentation in your\n"
            "system's default PDF viewer."
        )
        guide_button.clicked.connect(lambda: gui_utils.open_users_guide(self, page=40))
        top_row.addWidget(guide_button)

        self.layout.addLayout(top_row)

    def _setup_status_bar(self, font_size: int) -> None:
        """Configure the status bar."""
        self.status_bar = self.statusBar()
        font = self.status_bar.font()
        font.setFamily(Config.DEFAULT_FONT_FAMILY)
        font.setPointSize(font_size)
        self.status_bar.setFont(font)

    def _create_rugplots(self, dataframe: pd.DataFrame, columns: list[str]) -> None:
        """Create rugplot widgets in a grid layout."""
        n_columns = len(columns)
        if n_columns == 0:
            return

        max_rows = Config.MAX_RUGPLOTS_PER_COLUMN
        n_grid_cols = max(1, (n_columns - 1) // max_rows + 1)
        n_grid_rows = max(1, (n_columns - 1) // n_grid_cols + 1)

        for i, column in enumerate(columns):
            try:
                canvas = rugplot_canvas.RugPlotCanvas(dataframe[column], column)
                self.plot_widgets.append(canvas)
                self.grid_layout.addWidget(canvas, i % n_grid_rows, i // n_grid_rows)
            except ValueError as e:
                _logger.warning(f"Could not create rugplot for column '{column}': {e}")

    def update_plots(self, row_index: int | None = None) -> None:
        """
        Update plots to highlight the current row.

        Uses batched drawing for better performance.
        """
        if self._is_closed:
            return

        for widget in self.plot_widgets:
            if widget and not widget._is_destroyed:
                widget.update_highlight(highlight_index=row_index, draw=False)

        for widget in self.plot_widgets:
            if widget and not widget._is_destroyed:
                try:
                    widget.draw_idle()
                except RuntimeError:
                    # Widget destroyed during navigation - expected race condition
                    _logger.debug("Widget destroyed during draw_idle")

        if row_index is not None:
            try:
                row = self.df.loc[self.df['id'] == row_index]
                if not row.empty:
                    filename = row.iloc[0]['chosen_mesh_identifier']
                    self.status_bar.showMessage(f"Current File: {filename}")
                else:
                    self.status_bar.showMessage(f"Row index: {row_index}")
            except (KeyError, IndexError):
                self.status_bar.showMessage(f"Row index: {row_index}")

    def closeEvent(self, event) -> None:
        """Handle window close event with proper cleanup."""
        if self._is_closed:
            event.accept()
            return

        self._is_closed = True
        _logger.info("RugConsoleWindow closing")
        print("RugConsoleWindow is closing...")

        # Close the visualizer (and its 3D viewer) if still alive
        visualizer = self.visualizer
        if visualizer is not None:
            visualizer.on_close()

        for widget in self.plot_widgets:
            try:
                widget.close()
            except Exception as e:
                _logger.debug(f"Error closing plot widget: {e}")
        self.plot_widgets.clear()

        if hasattr(self, "console_widget"):
            try:
                self.console_widget.close()
            except Exception as e:
                _logger.debug(f"Error closing console widget: {e}")

        self._visualizer_ref = None
        super().closeEvent(event)


class ResultsShapeVisualizer:
    """
    Controller for browsing shape optimization results.

    Thin layer that connects the model (business logic) to the view (VisPy viewer).
    Handles keyboard input routing and timer management.
    """

    def __init__(
        self,
        folder_path: str | Path,
        rugplot_console_window: RugConsoleWindow,
        preloaded_dataframe: pd.DataFrame | None = None,
        display_time: float = Config.SLIDESHOW_DEFAULT_DISPLAY_TIME
    ):
        self.folder_path = Path(folder_path)
        self.rugplot_console_window = rugplot_console_window
        self._is_closing = False

        # Create model (business logic)
        self.model = ShapeBrowserModel(
            folder_path=folder_path,
            preloaded_dataframe=preloaded_dataframe,
            display_time=display_time
        )

        # Create file service
        self.file_service = FileService(folder_path)

        # View components
        self.viewer: VispyMeshViewer | None = None

        # Timer for slideshow (view concern - needs QTimer)
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)

        # Connect model signals
        self._connect_model_signals()

    def _connect_model_signals(self) -> None:
        """Connect model signals to view updates."""
        self.model.navigation_wrapped.connect(self._on_navigation_wrapped)
        self.model.slideshow_state_changed.connect(self._on_slideshow_state_changed)
        self.model.error_occurred.connect(self._on_error)

    def _is_viewer_alive(self) -> bool:
        """Check if viewer is still usable."""
        if self.viewer is None:
            return False
        return self.viewer.is_alive

    def _on_viewer_closed(self) -> None:
        """Handle viewer being closed (by user or error)."""
        if self._is_closing:
            return
        _logger.info("Viewer window closed")
        self.viewer = None
        self.on_close()

    def _on_viewer_error(self, message: str) -> None:
        """Handle viewer errors."""
        _logger.error(f"Viewer error: {message}")
        self.timer.stop()
        ErrorNotifier.show_error(
            self.rugplot_console_window,
            "3D Viewer Error",
            "The 3D viewer encountered an error and may need to be restarted.",
            details=message
        )

    def _on_navigation_wrapped(self, direction: str) -> None:
        """Handle navigation wrap-around."""
        if direction == "first":
            print(f"{'-'*20} Looping to first file {'-'*20}")
        else:
            print(f"{'-'*20} Looping to last file {'-'*20}")

    def _on_slideshow_state_changed(self, is_playing: bool, is_reversed: bool, display_time: float) -> None:
        """Handle slideshow state changes."""
        if is_playing:
            self.timer.start(int(display_time * 1000))
        else:
            self.timer.stop()

    def _on_error(self, message: str) -> None:
        """Handle errors from model."""
        _logger.warning(f"Model error: {message}")
        ErrorNotifier.show_warning(
            self.rugplot_console_window,
            "Warning",
            message
        )

    def _on_timer_tick(self) -> None:
        """Handle slideshow timer tick."""
        if self._is_closing or not self.model.slideshow.is_playing:
            return

        if not self._is_viewer_alive():
            _logger.warning("Slideshow stopped: viewer no longer available")
            self.model.slideshow.is_playing = False
            self.timer.stop()
            return

        try:
            mesh_info = self.model.advance_slideshow()
            if mesh_info:
                self._display_mesh(mesh_info)
        except Exception as e:
            details = ErrorNotifier.log_exception("Slideshow error", e)
            self.model.slideshow.is_playing = False
            self.timer.stop()
            ErrorNotifier.show_error(
                self.rugplot_console_window,
                "Slideshow Error",
                "An error occurred during slideshow playback.",
                details=details
            )

    def run(self, stack_key: str = 'loss') -> None:
        """Start the visualizer with the specified sorting key."""
        if not self.model.set_stack(stack_key):
            ErrorNotifier.show_warning(
                self.rugplot_console_window,
                "No Files",
                f"No files found for sort key '{stack_key}' in {self.folder_path}"
            )
            return

        if self.viewer is not None:
            self.viewer.close()

        self.viewer = VispyMeshViewer(
            parent_handler=self,
            title=f"Shape browser - Viewer (shapes sorted by '{stack_key}')"
        )

        # Register lifecycle callbacks
        self.viewer.on_close_callback(self._on_viewer_closed)
        self.viewer.on_error_callback(self._on_viewer_error)

        self.viewer.show_and_activate()
        self.print_controls()
        print("^ (See above for controls) ^\n")
        print(f"{'-'*20} Shapes sorted by '{stack_key}' {'-'*20}")

        mesh_info = self.model.go_to_file(0)
        if mesh_info:
            self._display_mesh(mesh_info)

    def _display_mesh(self, mesh_info: MeshDisplayInfo) -> None:
        """Display mesh in the viewer."""
        if not self._is_viewer_alive():
            _logger.debug("Cannot display mesh: viewer not alive")
            return

        color_value, color_name = self.model.get_color_for_algorithm(mesh_info.algorithm)

        try:
            success = self.viewer.load_and_display_aligned_with_target(
                mesh_info.aligned_file_path,
                mesh_info.target_mesh_path,
                color_value
            )
        except Exception as e:
            details = ErrorNotifier.log_exception("Mesh loading error", e)
            ErrorNotifier.show_error(
                self.rugplot_console_window,
                "Failed to Load Mesh",
                f"Could not load mesh: {mesh_info.mesh_identifier}",
                details=details
            )
            return

        if not success:
            _logger.warning(f"Failed to load mesh from {mesh_info.aligned_file_path}")
            ErrorNotifier.show_warning(
                self.rugplot_console_window,
                "Mesh Load Failed",
                f"Could not display mesh file:\n{Path(mesh_info.aligned_file_path).name}\n\n"
                "The file may be missing or corrupted."
            )
            return

        self._print_mesh_info(mesh_info, color_name)

        if self.model.update_rugplots:
            self.rugplot_console_window.update_plots(row_index=mesh_info.row_id)

    def _print_mesh_info(self, mesh_info: MeshDisplayInfo, color_name: str) -> None:
        """Print information about the current mesh."""
        idx_str = f"{mesh_info.index_in_stack + 1}/{mesh_info.stack_size}"

        loss_str = f"{mesh_info.loss:.2f}" if mesh_info.loss is not None else "N/A"
        chamfer_str = f"{mesh_info.chamfer_distance:.2f}" if mesh_info.chamfer_distance is not None else "N/A"

        if mesh_info.stack_key == 'tm_run_duration':
            if mesh_info.run_duration is None:
                info = "Run duration: N/A"
            elif mesh_info.run_duration > 60:
                info = f"Run duration: {mesh_info.run_duration / 60:.1f} min"
            else:
                info = f"Run duration: {mesh_info.run_duration:.2f} sec"
        elif mesh_info.stack_key in self.model.alignment_algorithms:
            info = (f"Best loss: {loss_str} ({mesh_info.best_algorithm}) | "
                    f"{mesh_info.algorithm} chamfer: {chamfer_str}")
        else:
            info = f"Best loss: {loss_str} | Algorithm: {mesh_info.algorithm} ({color_name})"

        filename = Path(mesh_info.aligned_file_path).name
        print(f"Displaying: {filename}\n{idx_str} (sorted by {mesh_info.stack_key}) | {info}")

    def handle_key_press(self, key: str) -> None:
        """Process key press events from the viewer."""
        handlers = {
            'Escape': self.on_close,
            'Space': self._toggle_slideshow,
            'R': self._toggle_direction,
            '+': self._increase_speed,
            '=': self._increase_speed,
            '-': self._decrease_speed,
            'Left': self._go_previous,
            'Right': self._go_next,
            'Up': self._go_first,
            'U': self._toggle_rugplot_update,
            'H': self._show_help,
            'Down': self._save_current,
            'J': self._jump_to_file,
        }

        if key in handlers:
            handlers[key]()
        elif key in '0123456789':
            digit = int(key)
            if digit in self.model.numkey_to_column:
                self.run(stack_key=self.model.numkey_to_column[digit])
        elif self.viewer:
            self.viewer._handle_viewer_key(key)

    def _go_next(self) -> None:
        mesh_info = self.model.go_to_next()
        if mesh_info:
            self._display_mesh(mesh_info)

    def _go_previous(self) -> None:
        mesh_info = self.model.go_to_previous()
        if mesh_info:
            self._display_mesh(mesh_info)

    def _go_first(self) -> None:
        mesh_info = self.model.go_to_first()
        if mesh_info:
            self._display_mesh(mesh_info)

    def _toggle_slideshow(self) -> None:
        is_playing = self.model.toggle_slideshow()
        print(f"{'-'*20} Slideshow: {'ON' if is_playing else 'OFF'} {'-'*20}")

    def _toggle_direction(self) -> None:
        is_reversed = self.model.toggle_direction()
        direction = "Descending" if is_reversed else "Ascending"
        print(f"{'-'*20} Direction: {direction} {'-'*20}")

    def _increase_speed(self) -> None:
        display_time = self.model.increase_speed()
        if self.model.slideshow.is_playing:
            self.timer.setInterval(int(display_time * 1000))
        print(f"{'-'*20} Display time: {display_time}s {'-'*20}")

    def _decrease_speed(self) -> None:
        display_time = self.model.decrease_speed()
        if self.model.slideshow.is_playing:
            self.timer.setInterval(int(display_time * 1000))
        print(f"{'-'*20} Display time: {display_time}s {'-'*20}")

    def _toggle_rugplot_update(self) -> None:
        state = self.model.toggle_rugplot_update()
        print(f"{'-'*20} Rugplot updating: {'ON' if state else 'OFF'} {'-'*20}")

    def _save_current(self) -> None:
        mesh_info = self.model.get_current_mesh_info()
        if not mesh_info:
            return

        screenshot_data = None
        if self.viewer:
            try:
                screenshot_data = self.viewer.render()
            except Exception as e:
                print(f"Error capturing screenshot: {e}")

        self.file_service.save_mesh_with_screenshot(mesh_info, screenshot_data)

    def _jump_to_file(self) -> None:
        if not self.model.current_file_stack:
            return

        try:
            file_index, ok = QInputDialog.getInt(
                self.rugplot_console_window,
                "Jump to file",
                f"Enter file index (1-{self.model.stack_size}):",
                value=self.model.current_file_index + 1,
                minValue=1,
                maxValue=self.model.stack_size
            )
            if ok:
                mesh_info = self.model.go_to_file(file_index - 1)
                if mesh_info:
                    self._display_mesh(mesh_info)
        except Exception as e:
            details = ErrorNotifier.log_exception("Jump to file error", e)
            ErrorNotifier.show_error(
                self.rugplot_console_window,
                "Navigation Error",
                "Could not jump to the specified file.",
                details=details
            )

    def _show_help(self) -> None:
        """Display help information."""
        if self._is_viewer_alive():
            self.viewer.print_help()
        self.print_controls()

    def print_controls(self) -> None:
        """Print shape browser specific control information."""
        if self.model.slideshow.is_playing:
            self._toggle_slideshow()

        print(f"""
{"="*60}
SHAPE BROWSER KEYBOARD CONTROLS (the 3D viewer window must be in focus for these to work)
{"="*60}
Left/Right  : Navigate shapes
Up            : Go to first shape
Down        : Save current mesh, parameters, and screenshot
J               : Jump to specific index
Space        : Pause/Resume slideshow
+/-           : Increase/decrease slideshow speed
R              : Reverse slideshow direction
U              : Toggle rugplot updating (off improves performance)
H              : Print this help message
0-9           : Sort by property/alignment ({self.model.numkey_info_str})
{"="*60}
""")

    def on_close(self) -> None:
        """Clean up and close the visualizer."""
        if self._is_closing:
            return
        self._is_closing = True

        _logger.info("Visualizer closing")
        print("Visualizer closing.")

        # Stop timer first to prevent callbacks during cleanup
        self.timer.stop()
        try:
            self.timer.timeout.disconnect(self._on_timer_tick)
        except (RuntimeError, TypeError):
            pass

        # Close viewer
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception as e:
                _logger.debug(f"Error closing viewer: {e}")
            self.viewer = None

        # Close main window
        try:
            self.rugplot_console_window.close()
        except Exception as e:
            _logger.debug(f"Error closing console window: {e}")


class ResultsViewer:
    """Main application controller for the shape browser."""

    def __init__(self, folder_path: str | Path | None = None):
        self.folder_path = Path(folder_path) if folder_path else None

    def select_results_folder(self) -> None:
        """Prompt user to select an output folder if needed."""
        if not self.folder_path or not self.folder_path.exists():
            if self.folder_path:
                print(f"Error: Folder '{self.folder_path}' does not exist.")

            selected = gui_utils.select_folder()
            self.folder_path = Path(selected) if selected else None

            if not self.folder_path or not self.folder_path.exists():
                raise FileNotFoundError("No valid folder selected.")

        print("-" * 40)
        print(f"Selected output folder: {self.folder_path}")

    def run(self) -> None:
        """Run the shape browser application."""
        try:
            dataframe = shape_browser_utils.get_dataframe_from_csv(self.folder_path)
            dataframe.insert(0, 'id', dataframe.index.to_series())

            main_window = RugConsoleWindow(
                dataframe,
                font_size=Config.DEFAULT_FONT_SIZE - 1
            )
            main_window.show()

            visualizer = ResultsShapeVisualizer(
                folder_path=self.folder_path,
                rugplot_console_window=main_window,
                preloaded_dataframe=dataframe
            )
            main_window.visualizer = visualizer
            visualizer.run()

        except NoDataError as e:
            self._show_error("No Data", str(e))
            raise
        except shape_browser_utils.ShapeBrowserUtilsError as e:
            self._show_error(
                "Invalid Output Folder",
                f"The selected folder does not appear to contain valid "
                f"optimization results:\n\n{e}\n\n"
                f"Folder: {self.folder_path}\n\n"
                f"Please select a folder that contains the results of a "
                f"completed optimization run."
            )
            raise
        except Exception as e:
            self._show_error("Error", f"Failed to start shape browser: {e}")
            raise

    def _show_error(self, title: str, message: str) -> None:
        """Show an error message box."""
        app = QApplication.instance()
        if app:
            QMessageBox.critical(None, title, message)
        else:
            print(f"ERROR - {title}: {message}")


def run_results_viewer(folder_path: str | Path | None = None) -> None:
    """
    Run the results viewer.

    This function works with a QApplication created externally.
    """
    viewer = ResultsViewer(folder_path)
    folder_as_path = Path(folder_path) if folder_path else None
    if not folder_as_path or not folder_as_path.exists():
        viewer.select_results_folder()
    viewer.run()


if __name__ == "__main__":
    folder_path = None
    if len(sys.argv) > 1:
        folder_path = Path(sys.argv[1])
        print(f"Using folder path: {folder_path}")

    app = QApplication(sys.argv)

    try:
        run_results_viewer(folder_path)
        sys.exit(app.exec())
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
