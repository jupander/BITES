"""
Optuna Plot Worker - Background thread for generating Optuna visualization plots.

This module provides a QThread-based worker for generating Optuna plots without
blocking the main UI thread, with progress reporting and cancellation support.
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class PlotCost(Enum):
    """Relative computational cost of each plot type."""
    FAST = 1      # < 1 second typically
    MEDIUM = 2    # 1-5 seconds
    SLOW = 5      # 5-30 seconds
    VERY_SLOW = 10  # 30+ seconds, may take minutes


@dataclass
class PlotInfo:
    """Information about an Optuna plot type."""
    id: str
    display_name: str
    description: str
    cost: PlotCost
    min_trials: int = 1  # Minimum completed trials required

    @property
    def cost_indicator(self) -> str:
        """Return a visual indicator of computational cost."""
        if self.cost == PlotCost.FAST:
            return "Fast"
        elif self.cost == PlotCost.MEDIUM:
            return "Medium"
        elif self.cost == PlotCost.SLOW:
            return "Slow"
        else:
            return "Very slow - may take minutes"

    @property
    def cost_icon(self) -> str:
        """Return an icon representing cost."""
        if self.cost == PlotCost.FAST:
            return "\u26A1"  # Lightning bolt
        elif self.cost == PlotCost.MEDIUM:
            return "\u23F1"  # Stopwatch
        elif self.cost == PlotCost.SLOW:
            return "\u231B"  # Hourglass
        else:
            return "\u26A0"  # Warning triangle


# Define available plots with their metadata
AVAILABLE_PLOTS: Dict[str, PlotInfo] = {
    "optimization_history": PlotInfo(
        id="optimization_history",
        display_name="Optimization History",
        description="Shows how the objective value improved over trials",
        cost=PlotCost.FAST,
        min_trials=1,
    ),
    "slice": PlotInfo(
        id="slice",
        display_name="Slice Plot",
        description="Shows the relationship between each parameter and the objective",
        cost=PlotCost.MEDIUM,
        min_trials=2,
    ),
    "parallel_coordinate": PlotInfo(
        id="parallel_coordinate",
        display_name="Parallel Coordinate",
        description="Shows parameter combinations across all trials",
        cost=PlotCost.MEDIUM,
        min_trials=2,
    ),
    "param_importances": PlotInfo(
        id="param_importances",
        display_name="Parameter Importances",
        description="Calculates which parameters most affect the objective (uses fANOVA)",
        cost=PlotCost.VERY_SLOW,
        min_trials=3,
    ),
    "contour": PlotInfo(
        id="contour",
        display_name="Contour Plot",
        description="Shows 2D relationships between parameter pairs and objective",
        cost=PlotCost.SLOW,
        min_trials=4,
    ),
}

# Default plot order (fast to slow)
DEFAULT_PLOT_ORDER = [
    "optimization_history",
    "slice",
    "parallel_coordinate",
    "contour",
    "param_importances",
]


@dataclass
class PlotResult:
    """Result of generating a single plot."""
    plot_id: str
    success: bool
    file_path: Optional[Path] = None
    error_message: Optional[str] = None


class OptunaPlotWorker(QThread):
    """
    Background worker for generating Optuna visualization plots.

    Signals:
        started_plot: Emitted when starting to generate a plot (plot_id, current_index, total)
        completed_plot: Emitted when a plot is done (PlotResult)
        all_completed: Emitted when all plots are done (list of PlotResults)
        error: Emitted on fatal error (error_message)
    """

    # Signals
    started_plot = Signal(str, int, int)  # plot_id, current (1-based), total
    completed_plot = Signal(object)  # PlotResult
    all_completed = Signal(list)  # List[PlotResult]
    error = Signal(str)  # Fatal error message

    def __init__(
        self,
        output_folder: Path,
        selected_plots: List[str],
        open_in_browser: bool = True,
        parent=None
    ):
        """
        Initialize the plot worker.

        Args:
            output_folder: Path to the optimization output folder
            selected_plots: List of plot IDs to generate
            open_in_browser: Whether to open plots in browser after saving
            parent: Parent QObject
        """
        super().__init__(parent)
        self.output_folder = output_folder
        self.selected_plots = selected_plots
        self.open_in_browser = open_in_browser
        self._cancelled = threading.Event()
        self._study = None

    def cancel(self) -> None:
        """Request cancellation of the worker (thread-safe)."""
        logger.info("Plot generation cancellation requested")
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested (thread-safe)."""
        return self._cancelled.is_set()

    def run(self) -> None:
        """Main worker thread execution."""
        results: List[PlotResult] = []

        try:
            # Load the study first
            logger.info(f"Loading Optuna study from {self.output_folder}")
            self._study = self._load_study()

            if self.is_cancelled:
                logger.info("Cancelled before generating plots")
                return

            # Create output directory
            vis_dir = self.output_folder / "optuna_visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

            total = len(self.selected_plots)

            for i, plot_id in enumerate(self.selected_plots, 1):
                if self.is_cancelled:
                    logger.info(f"Cancelled at plot {i}/{total}")
                    break

                self.started_plot.emit(plot_id, i, total)

                result = self._generate_single_plot(plot_id, vis_dir)
                results.append(result)
                self.completed_plot.emit(result)

            self.all_completed.emit(results)

        except Exception as e:
            logger.error(f"Fatal error in plot worker: {e}", exc_info=True)
            self.error.emit(str(e))

    def _load_study(self) -> None:
        """Load the Optuna study from the output folder."""
        try:
            import optuna
            from bites.config.config_loader import Config
            from bites.database.pg_db_manager import ensure_database_for_target
        except ImportError as e:
            raise ImportError(
                f"Failed to import required modules: {e}. "
                "This may indicate a packaging issue with the executable."
            ) from e

        # Find the most recent config file
        config_files = list(self.output_folder.glob("config_*.yaml"))
        if not config_files:
            raise FileNotFoundError(f"No config file found in {self.output_folder}")

        config_file = sorted(config_files)[-1]
        logger.info(f"Using config file: {config_file.name}")

        config = Config.from_yaml(str(config_file))

        # Get storage URL
        if config.storage_mode == "shared":
            storage_url = ensure_database_for_target(config)
        else:
            raise ValueError(f"Unsupported storage mode: {config.storage_mode}")

        return optuna.load_study(
            study_name=config.study_name,
            storage=storage_url
        )

    def _generate_single_plot(self, plot_id: str, vis_dir: Path) -> PlotResult:
        """
        Generate a single plot.

        Args:
            plot_id: The plot identifier
            vis_dir: Directory to save the plot

        Returns:
            PlotResult with success/failure info
        """
        try:
            import optuna.visualization as vis
        except ImportError as e:
            return PlotResult(
                plot_id=plot_id,
                success=False,
                error_message=f"Failed to import optuna.visualization: {e}. "
                "This may indicate a packaging issue with the executable."
            )

        plot_info = AVAILABLE_PLOTS.get(plot_id)
        if not plot_info:
            return PlotResult(
                plot_id=plot_id,
                success=False,
                error_message=f"Unknown plot type: {plot_id}"
            )

        # Check minimum trials
        n_complete = len([t for t in self._study.trials
                         if t.state == self._study.trials[0].state.__class__.COMPLETE])
        if n_complete < plot_info.min_trials:
            return PlotResult(
                plot_id=plot_id,
                success=False,
                error_message=f"Requires at least {plot_info.min_trials} completed trials (found {n_complete})"
            )

        # Map plot_id to visualization function
        plot_functions = {
            "optimization_history": vis.plot_optimization_history,
            "slice": vis.plot_slice,
            "parallel_coordinate": vis.plot_parallel_coordinate,
            "param_importances": vis.plot_param_importances,
            "contour": vis.plot_contour,
        }

        plot_func = plot_functions.get(plot_id)
        if not plot_func:
            return PlotResult(
                plot_id=plot_id,
                success=False,
                error_message=f"No plot function for: {plot_id}"
            )

        try:
            logger.info(f"Generating {plot_id}...")
            fig = plot_func(self._study)

            if self.is_cancelled:
                return PlotResult(
                    plot_id=plot_id,
                    success=False,
                    error_message="Cancelled"
                )

            # Save the plot
            filename = f"{plot_id}.html"
            filepath = vis_dir / filename
            fig.write_html(str(filepath), validate=False)
            logger.info(f"Saved: {filepath}")

            # Note: Browser opening is handled by the dialog in the main thread
            # when it receives the completed_plot signal

            return PlotResult(
                plot_id=plot_id,
                success=True,
                file_path=filepath
            )

        except Exception as e:
            logger.error(f"Failed to generate {plot_id}: {e}", exc_info=True)
            return PlotResult(
                plot_id=plot_id,
                success=False,
                error_message=str(e)
            )


def get_plot_order() -> List[str]:
    """Get the default plot order (fast to slow)."""
    return DEFAULT_PLOT_ORDER.copy()


def get_available_plots() -> Dict[str, PlotInfo]:
    """Get dictionary of available plots."""
    return AVAILABLE_PLOTS.copy()
