"""
RugPlotCanvas - A matplotlib-based rugplot widget for PySide6.

Displays a horizontal rugplot showing data distribution with optional highlighting.
"""
# Import PySide6 first so matplotlib knows which Qt binding to use
from PySide6 import QtWidgets

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import bites.gui.gui_utils as gui_utils
from bites.constants import ShapeBrowserConfig as Config

# Fix 3: Move global rcParams to module level (run once at import, not per-instance)
plt.rcParams.update({'figure.max_open_warning': 0})


class RugPlotCanvas(FigureCanvas):
    """
    A canvas that displays a rug plot using matplotlib inside a PySide6 widget.

    Attributes:
        data: The pandas Series or array of values to display.
        column_name: The name of the data column (shown as label).
    """

    def __init__(self, data: pd.Series | np.ndarray, column_name: str, parent=None):
        """
        Initialize the rugplot canvas.

        Args:
            data: Numeric data to visualize as a rugplot.
            column_name: Name to display as the plot label.
            parent: Optional parent widget.
        """
        self._is_destroyed = False

        # Validate data
        if data is None or len(data) == 0:
            raise ValueError("RugPlotCanvas requires non-empty data")

        # Store data, dropping NaN values to prevent plotting issues
        if isinstance(data, pd.Series):
            self._original_data = data
            self.data = data.dropna()
        else:
            self._original_data = data
            self.data = pd.Series(data).dropna()

        self.column_name = column_name
        self.highlight_label = None

        # Cached artists for performance (avoid redrawing static elements)
        self._orange_ticks_artist = None
        self._axis_line_artist = None
        self._highlight_artist = None
        self._static_initialized = False
        self._orange_ticks_visible = True

        # Fix 4: Wrap figure creation in try-finally to prevent leaks on exception
        fig = None
        try:
            fig, self.ax = plt.subplots()
            fig.tight_layout(pad=0, w_pad=0, h_pad=0)
            super().__init__(fig)
        except Exception:
            if fig is not None:
                plt.close(fig)
            raise

        if parent is not None:
            self.setParent(parent)

        # Create persistent labels
        self.min_label = self.figure.text(
            Config.RUGPLOT_XAXIS_MIN, Config.RUGPLOT_LABEL_Y, '',
            verticalalignment='baseline', horizontalalignment='left',
            transform=self.ax.transAxes
        )
        self.max_label = self.figure.text(
            1 - Config.RUGPLOT_XAXIS_MIN, Config.RUGPLOT_LABEL_Y, '',
            verticalalignment='baseline', horizontalalignment='right',
            transform=self.ax.transAxes
        )
        self.name_label = self.figure.text(
            0.5, Config.RUGPLOT_LABEL_Y, self.column_name,
            verticalalignment='baseline', horizontalalignment='center',
            transform=self.ax.transAxes
        )

        # Hide spines
        for position in ['left', 'bottom', 'right', 'top']:
            self.ax.spines[position].set_visible(False)

        # Set tooltip with variable name and data range
        self._update_tooltip()

        # Initial draw
        self.rugplot()

    def _update_tooltip(self, highlighted_value=None) -> None:
        """Set a Qt tooltip showing the variable name, data range, and current value."""
        if len(self.data) == 0:
            self.setToolTip(self.column_name)
            return

        data_min = gui_utils.set_decimal_places(self.data.min())
        data_max = gui_utils.set_decimal_places(self.data.max())
        tip = f"{self.column_name}\nMin: {data_min}\nMax: {data_max}"
        if highlighted_value is not None and pd.notna(highlighted_value):
            tip += f"\nCurrent value: {gui_utils.set_decimal_places(highlighted_value)}"
        self.setToolTip(tip)

    def _initialize_static_elements(self, orange_ticks: bool = True) -> None:
        """
        Initialize static plot elements (called once, not on every update).

        Creates orange ticks, axis line, sets limits and labels.
        These elements don't change when navigating between meshes.
        """
        if self._is_destroyed or len(self.data) == 0:
            return

        # Calculate axis limits with padding
        data_min = self.data.min()
        data_max = self.data.max()
        data_range = data_max - data_min

        if data_range == 0:
            padding = abs(data_min) * Config.RUGPLOT_PADDING_FACTOR if data_min != 0 else 1.0
        else:
            padding = Config.RUGPLOT_PADDING_FACTOR * data_range

        self.ax.set_xlim(data_min - padding, data_max + padding)
        self.ax.set_ylim(Config.RUGPLOT_Y_MIN, Config.RUGPLOT_Y_MAX)
        self.ax.set_yticks([])
        self.ax.set_xticks([])

        # Draw orange ticks (all data points) - cached for reuse
        if orange_ticks:
            self._orange_ticks_artist = self.ax.eventplot(
                self.data, orientation='horizontal',
                colors=Config.RUGPLOT_TICK_COLOR,
                linewidths=Config.RUGPLOT_TICK_WIDTH
            )
            self._orange_ticks_visible = True
        else:
            self._orange_ticks_visible = False

        # Draw horizontal axis line - cached for reuse
        self._axis_line_artist = self.ax.axhline(
            y=Config.RUGPLOT_AXIS_Y,
            color=Config.RUGPLOT_TICK_COLOR,
            linewidth=Config.RUGPLOT_AXIS_LINE_WIDTH,
            xmin=Config.RUGPLOT_XAXIS_MIN,
            xmax=Config.RUGPLOT_XAXIS_MAX
        )

        # Update min/max labels (static, only depends on data)
        min_label_str = gui_utils.set_decimal_places(data_min)
        max_label_str = gui_utils.set_decimal_places(data_max)
        self.min_label.set_text(min_label_str)
        self.max_label.set_text(max_label_str)

        self._static_initialized = True

    def _set_orange_ticks_visible(self, visible: bool) -> None:
        """Toggle visibility of orange ticks without recreating them."""
        if self._orange_ticks_artist is None:
            return

        # eventplot returns a list of EventCollection objects
        for collection in self._orange_ticks_artist:
            collection.set_visible(visible)
        self._orange_ticks_visible = visible

    def set_orange_ticks_visible(self, visible: bool) -> None:
        """
        Fix 8: Public method to toggle orange ticks visibility.

        This method properly encapsulates the orange ticks toggling logic
        so external code doesn't need to access private members.

        Args:
            visible: Whether orange ticks should be visible.
        """
        if self._is_destroyed:
            return

        if visible == self._orange_ticks_visible:
            return  # No change needed

        # Create orange ticks if they don't exist yet and we need to show them
        if visible and self._orange_ticks_artist is None:
            self._orange_ticks_artist = self.ax.eventplot(
                self.data, orientation='horizontal',
                colors=Config.RUGPLOT_TICK_COLOR,
                linewidths=Config.RUGPLOT_TICK_WIDTH
            )

        self._set_orange_ticks_visible(visible)

    def update_highlight(self, highlight_index: int | None = None, draw: bool = True) -> None:
        """
        Update only the highlight position (lightweight update).

        This is the fast path - only removes old highlight and adds new one.
        Call this instead of rugplot() when navigating between meshes.

        Args:
            highlight_index: Index in the original data to highlight (blue tick).
            draw: Whether to trigger a canvas redraw. Set False for batched updates.
        """
        if self._is_destroyed or len(self.data) == 0:
            return

        # Fix 5: Remove previous highlight artist with per-item exception safety
        if self._highlight_artist is not None:
            for collection in self._highlight_artist:
                try:
                    collection.remove()
                except Exception:
                    pass  # Artist may already be removed or axes destroyed
            self._highlight_artist = None

        # Remove previous highlight label if it exists
        if self.highlight_label is not None:
            try:
                self.highlight_label.remove()
            except Exception:
                pass  # Label may already be removed
            self.highlight_label = None

        # Draw new highlight if specified
        # Fix 2: Reject negative indices (would wrap around unexpectedly)
        highlighted_value = None
        if highlight_index is not None and highlight_index >= 0:
            try:
                # Fix 1: Use loc if index exists in data's index, else fall back to iloc
                if highlight_index in self._original_data.index:
                    highlighted_value = self._original_data.loc[highlight_index]
                elif highlight_index < len(self._original_data):
                    highlighted_value = self._original_data.iloc[highlight_index]

                if highlighted_value is not None and pd.notna(highlighted_value):
                    self._highlight_artist = self.ax.eventplot(
                        [highlighted_value], orientation='horizontal',
                        colors=Config.RUGPLOT_HIGHLIGHT_COLOR,
                        linewidths=Config.RUGPLOT_HIGHLIGHT_TICK_WIDTH
                    )
                    highlighted_value_str = gui_utils.set_decimal_places(highlighted_value)
                    self.highlight_label = self.ax.text(
                        highlighted_value, Config.RUGPLOT_HIGHLIGHT_LABEL_Y,
                        highlighted_value_str,
                        color=Config.RUGPLOT_HIGHLIGHT_COLOR,
                        ha='center', va='bottom'
                    )
            except (IndexError, KeyError, TypeError):
                highlighted_value = None

        self._update_tooltip(highlighted_value)

        if draw and not self._is_destroyed:
            self._safe_draw_idle()

    def _safe_draw_idle(self) -> None:
        """
        Safely schedule a deferred draw, protecting against C++ object deletion.

        This prevents RuntimeError when Qt has already deleted the underlying
        C++ widget but matplotlib's idle callback still fires.
        """
        if self._is_destroyed:
            return

        try:
            # Check if the underlying Qt C++ object still exists
            # Accessing any Qt property will raise RuntimeError if deleted
            _ = self.width()
            super().draw_idle()
        except RuntimeError:
            # C++ object already deleted, mark as destroyed
            self._is_destroyed = True
        except Exception:
            pass  # Silently ignore other drawing errors

    def draw_idle(self) -> None:
        """Override draw_idle to add destruction check."""
        self._safe_draw_idle()

    def rugplot(self, highlight_index: int | None = None, orange_ticks: bool = True) -> None:
        """
        Draw the rug plot with optional highlighted index.

        Args:
            highlight_index: Index in the original data to highlight (blue tick).
            orange_ticks: Whether to show all data points as orange ticks.
        """
        if self._is_destroyed:
            return

        if len(self.data) == 0:
            return

        # Initialize static elements on first call
        if not self._static_initialized:
            self._initialize_static_elements(orange_ticks)

        # Handle orange ticks visibility toggle
        if orange_ticks != self._orange_ticks_visible:
            if orange_ticks and self._orange_ticks_artist is None:
                # Need to create orange ticks (were never created)
                self._orange_ticks_artist = self.ax.eventplot(
                    self.data, orientation='horizontal',
                    colors=Config.RUGPLOT_TICK_COLOR,
                    linewidths=Config.RUGPLOT_TICK_WIDTH
                )
            self._set_orange_ticks_visible(orange_ticks)

        # Update highlight (the only thing that changes between meshes)
        self.update_highlight(highlight_index, draw=True)

    def resizeEvent(self, event) -> None:
        """Handle resize events with destruction check."""
        if self._is_destroyed:
            return
        try:
            super().resizeEvent(event)
        except RuntimeError:
            # C++ object deleted during resize
            self._is_destroyed = True

    def closeEvent(self, event) -> None:
        """Clean up matplotlib resources before Qt destroys the widget."""
        # Fix 6: Double-close protection
        if self._is_destroyed:
            event.accept()
            return

        self._is_destroyed = True

        # Stop any pending idle draw timer to prevent callback after deletion
        try:
            if hasattr(self, '_idle_draw_cntx') and self._idle_draw_cntx is not None:
                self._idle_draw_cntx.stop()
            if hasattr(self, '_timer') and self._timer is not None:
                self._timer.stop()
        except Exception:
            pass

        try:
            plt.close(self.figure)
        except Exception:
            pass  # Figure may already be closed
        super().closeEvent(event)
