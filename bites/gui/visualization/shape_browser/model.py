"""
Shape Browser Model - Business logic separated from GUI.

Contains:
- ShapeBrowserModel: State management, navigation, data loading
- FileService: File operations (saving meshes, screenshots)
"""
from __future__ import annotations

import datetime
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Signal

import bites.gui.visualization.shape_browser.utils as shape_browser_utils
import bites.utils.toothmaker_format as toothmaker_format
from bites.constants import (
    ShapeBrowserConfig as Config,
    SAVED_BY_SHAPE_BROWSER_FOLDER,
    ALL_SCREENSHOTS_SUBFOLDER,
)
from bites.paths import get_project_root

if TYPE_CHECKING:
    import pandas as pd

_logger = logging.getLogger(__name__)


@dataclass
class MeshDisplayInfo:
    """Information needed to display a mesh in the viewer."""
    aligned_file_path: str
    target_mesh_path: str
    loss: float | None
    chamfer_distance: float | None
    algorithm: str
    best_algorithm: str
    row_id: int
    mesh_identifier: str
    run_duration: float | None
    index_in_stack: int
    stack_size: int
    stack_key: str


@dataclass
class SlideshowState:
    """Slideshow playback state."""
    is_playing: bool = False
    is_reversed: bool = False
    display_time: float = Config.SLIDESHOW_DEFAULT_DISPLAY_TIME

    def toggle_playing(self) -> bool:
        """Toggle playing state, returns new state."""
        self.is_playing = not self.is_playing
        return self.is_playing

    def toggle_reversed(self) -> bool:
        """Toggle direction, returns new state."""
        self.is_reversed = not self.is_reversed
        return self.is_reversed

    def increase_speed(self) -> float:
        """Increase speed (decrease display time), returns new time."""
        if self.display_time <= Config.SLIDESHOW_MIN_DISPLAY_TIME:
            return self.display_time
        self.display_time = max(
            Config.SLIDESHOW_MIN_DISPLAY_TIME,
            self.display_time / Config.SLIDESHOW_SPEED_MULTIPLIER
        )
        return self.display_time

    def decrease_speed(self) -> float:
        """Decrease speed (increase display time), returns new time."""
        if self.display_time >= Config.SLIDESHOW_MAX_DISPLAY_TIME:
            return self.display_time
        self.display_time = min(
            Config.SLIDESHOW_MAX_DISPLAY_TIME,
            self.display_time * Config.SLIDESHOW_SPEED_MULTIPLIER
        )
        return self.display_time


class ShapeBrowserModel(QObject):
    """
    Model for shape browser state and navigation.

    Emits signals when state changes so views can update accordingly.
    All business logic (navigation, data loading, sorting) lives here.
    """

    # Signals for state changes
    mesh_changed = Signal(object)  # Emits MeshDisplayInfo
    stack_changed = Signal(str, int)  # stack_key, stack_size
    slideshow_state_changed = Signal(bool, bool, float)  # is_playing, is_reversed, display_time
    navigation_wrapped = Signal(str)  # "first" or "last"
    error_occurred = Signal(str)  # error message

    def __init__(
        self,
        folder_path: str | Path,
        preloaded_dataframe: pd.DataFrame | None = None,
        display_time: float = Config.SLIDESHOW_DEFAULT_DISPLAY_TIME
    ):
        super().__init__()
        self.folder_path = Path(folder_path)

        # State
        self._current_file_index = 0
        self._current_stack_key = 'loss'
        self._current_file_stack: list[dict] = []
        self._update_rugplots = True
        self.slideshow = SlideshowState(display_time=display_time)

        # Load alignment algorithms from config
        self.alignment_algorithms = shape_browser_utils.get_alignment_algorithm_names(
            self.folder_path
        )

        # Setup sorting options
        self.non_algo_columns = Config.NON_ALGORITHM_SORT_COLUMNS.copy()
        self.stack_keys = self.non_algo_columns + self.alignment_algorithms

        # Load dataframe
        self._load_dataframe(preloaded_dataframe)

        # Build file stacks for each sort key
        self.file_stacks: dict[str, list[dict]] = {}
        self._build_file_stacks()

        # Color setup
        self._setup_colors()

        # Number key mappings
        self._setup_numkey_mappings()

    @property
    def current_file_index(self) -> int:
        return self._current_file_index

    @property
    def current_stack_key(self) -> str:
        return self._current_stack_key

    @property
    def current_file_stack(self) -> list[dict]:
        return self._current_file_stack

    @property
    def stack_size(self) -> int:
        return len(self._current_file_stack)

    @property
    def update_rugplots(self) -> bool:
        return self._update_rugplots

    @update_rugplots.setter
    def update_rugplots(self, value: bool) -> None:
        self._update_rugplots = value

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def _load_dataframe(self, preloaded_dataframe: pd.DataFrame | None) -> None:
        """Load and prepare the dataframe."""
        if preloaded_dataframe is not None:
            self._dataframe = preloaded_dataframe.copy()
        else:
            print(f"Loading CSV file from '{self.folder_path}'...")
            self._dataframe = shape_browser_utils.get_dataframe_from_csv(self.folder_path)

        # Ensure 'id' column exists
        if 'id' not in self._dataframe.columns:
            print("Adding 'id' column to dataframe...")
            self._dataframe.insert(0, 'id', self._dataframe.index.to_series())

        # Filter out PRUNED and RUNNING rows
        if 'state' in self._dataframe.columns:
            original_count = len(self._dataframe)
            self._dataframe = self._dataframe[self._dataframe['state'] != 'PRUNED']
            filtered_count = len(self._dataframe)
            if filtered_count < original_count:
                print(f"Filtered PRUNED rows: {original_count} -> {filtered_count}")

            self._dataframe = self._dataframe[self._dataframe['state'] != 'RUNNING']
            filtered_count_running = len(self._dataframe)
            if filtered_count_running < filtered_count:
                print(f"Filtered RUNNING rows: {filtered_count} -> {filtered_count_running}")

    def _build_file_stacks(self) -> None:
        """Build sorted file stacks for each column."""
        for stack_key in self.stack_keys:
            try:
                self.file_stacks[stack_key] = shape_browser_utils.list_off_files_in_order(
                    column_to_sort_by=stack_key,
                    dataframe=self._dataframe,
                    alignment_algorithms=self.alignment_algorithms,
                    base_folder_path=self.folder_path
                )
            except KeyError:
                print(f"Warning: Column '{stack_key}' not found, skipping.")

    def _setup_colors(self) -> None:
        """Setup color mappings for algorithms."""
        self.color_name_to_value = Config.ALGORITHM_COLORS.copy()

        self.algo_to_color_name = {}
        color_order = Config.ALGORITHM_COLOR_ORDER
        for i, algo in enumerate(self.alignment_algorithms):
            color_name = color_order[i % len(color_order)]
            self.algo_to_color_name[algo] = color_name

    def _setup_numkey_mappings(self) -> None:
        """Setup number key to column mappings."""
        self.numkey_to_column = {}
        for numkey, column in enumerate(self.stack_keys[:10]):
            self.numkey_to_column[numkey] = column

        self.numkey_info_str = ' | '.join(
            f"{k}:{v}" for k, v in self.numkey_to_column.items()
        )

    def set_stack(self, stack_key: str) -> bool:
        """
        Switch to a different sort order.

        Args:
            stack_key: The column to sort by.

        Returns:
            True if successful, False if stack is empty or not found.
        """
        file_stack = self.file_stacks.get(stack_key, [])

        if not file_stack:
            self.error_occurred.emit(f"No files found for sort key '{stack_key}'")
            return False

        self._current_stack_key = stack_key
        self._current_file_stack = file_stack
        self._current_file_index = 0

        self.stack_changed.emit(stack_key, len(file_stack))
        return True

    def go_to_file(self, file_index: int) -> MeshDisplayInfo | None:
        """
        Navigate to a specific file index.

        Args:
            file_index: Index to navigate to (wraps around).

        Returns:
            MeshDisplayInfo for the new position, or None if stack is empty.
        """
        if not self._current_file_stack:
            return None

        stack_size = len(self._current_file_stack)

        # Wrap around
        if file_index >= stack_size:
            self._current_file_index = 0
            self.navigation_wrapped.emit("first")
        elif file_index < 0:
            self._current_file_index = stack_size - 1
            self.navigation_wrapped.emit("last")
        else:
            self._current_file_index = file_index

        return self._emit_current_mesh()

    def go_to_next(self) -> MeshDisplayInfo | None:
        """Navigate to next file."""
        return self.go_to_file(self._current_file_index + 1)

    def go_to_previous(self) -> MeshDisplayInfo | None:
        """Navigate to previous file."""
        return self.go_to_file(self._current_file_index - 1)

    def go_to_first(self) -> MeshDisplayInfo | None:
        """Navigate to first file."""
        return self.go_to_file(0)

    def go_to_last(self) -> MeshDisplayInfo | None:
        """Navigate to last file."""
        return self.go_to_file(len(self._current_file_stack) - 1)

    def advance_slideshow(self) -> MeshDisplayInfo | None:
        """Advance one step in slideshow direction."""
        if not self.slideshow.is_playing:
            return None
        delta = -1 if self.slideshow.is_reversed else 1
        return self.go_to_file(self._current_file_index + delta)

    def toggle_slideshow(self) -> bool:
        """Toggle slideshow, returns new playing state."""
        is_playing = self.slideshow.toggle_playing()
        self.slideshow_state_changed.emit(
            is_playing,
            self.slideshow.is_reversed,
            self.slideshow.display_time
        )
        return is_playing

    def toggle_direction(self) -> bool:
        """Toggle slideshow direction, returns True if reversed."""
        is_reversed = self.slideshow.toggle_reversed()
        self.slideshow_state_changed.emit(
            self.slideshow.is_playing,
            is_reversed,
            self.slideshow.display_time
        )
        return is_reversed

    def increase_speed(self) -> float:
        """Increase slideshow speed, returns new display time."""
        display_time = self.slideshow.increase_speed()
        self.slideshow_state_changed.emit(
            self.slideshow.is_playing,
            self.slideshow.is_reversed,
            display_time
        )
        return display_time

    def decrease_speed(self) -> float:
        """Decrease slideshow speed, returns new display time."""
        display_time = self.slideshow.decrease_speed()
        self.slideshow_state_changed.emit(
            self.slideshow.is_playing,
            self.slideshow.is_reversed,
            display_time
        )
        return display_time

    def toggle_rugplot_update(self) -> bool:
        """Toggle rugplot updating, returns new state."""
        self._update_rugplots = not self._update_rugplots
        return self._update_rugplots

    def get_current_mesh_info(self) -> MeshDisplayInfo | None:
        """Get info for current mesh without emitting signal."""
        if not self._current_file_stack:
            return None
        return self._create_mesh_info(self._current_file_stack[self._current_file_index])

    def get_color_for_algorithm(self, algorithm: str) -> tuple[list[float], str]:
        """
        Get color for an algorithm.

        Returns:
            Tuple of (color_value, color_name)
        """
        color_name = self.algo_to_color_name.get(algorithm, 'green')
        color_value = self.color_name_to_value.get(color_name, [0, 1, 0])
        return color_value, color_name

    def _emit_current_mesh(self) -> MeshDisplayInfo:
        """Create and emit mesh info for current position."""
        file_dict = self._current_file_stack[self._current_file_index]
        mesh_info = self._create_mesh_info(file_dict)
        self.mesh_changed.emit(mesh_info)
        return mesh_info

    def _create_mesh_info(self, file_dict: dict) -> MeshDisplayInfo:
        """Create MeshDisplayInfo from file dictionary."""
        return MeshDisplayInfo(
            aligned_file_path=file_dict['aligned_file_path'],
            target_mesh_path=file_dict['target_mesh_path'],
            loss=file_dict['loss'],
            chamfer_distance=file_dict['chamfer_distance'],
            algorithm=file_dict['algorithm'],
            best_algorithm=file_dict['best_algorithm'],
            row_id=file_dict['id'],
            mesh_identifier=file_dict['chosen_mesh_identifier'],
            run_duration=file_dict['run_duration'],
            index_in_stack=self._current_file_index,
            stack_size=len(self._current_file_stack),
            stack_key=self._current_stack_key
        )


class FileService:
    """
    Service for file operations related to shape browser.

    Handles saving meshes, parameters, and screenshots.
    Stateless - all state is passed in as parameters.
    """

    def __init__(self, folder_path: str | Path):
        self.folder_path = Path(folder_path)

    def save_mesh_with_screenshot(
        self,
        mesh_info: MeshDisplayInfo,
        screenshot_data: Any | None = None
    ) -> Path:
        """
        Save mesh files, parameters, and optional screenshot.

        Args:
            mesh_info: Information about the mesh to save.
            screenshot_data: Optional screenshot image data (numpy array).

        Returns:
            Path to the saved folder.
        """
        mesh_identifier = mesh_info.mesh_identifier
        print(f"\n{'-'*20} Saving current mesh {'-'*20}")

        # Create save directories
        results_saved_folder = self.folder_path / SAVED_BY_SHAPE_BROWSER_FOLDER
        results_subfolder = results_saved_folder / mesh_identifier
        results_screenshots_folder = results_saved_folder / ALL_SCREENSHOTS_SUBFOLDER

        for folder in [results_saved_folder, results_subfolder, results_screenshots_folder]:
            folder.mkdir(parents=True, exist_ok=True)

        # Save screenshot if provided
        screenshot_path = None
        if screenshot_data is not None:
            screenshot_path = self._save_screenshot(
                screenshot_data,
                mesh_identifier,
                results_subfolder,
                results_screenshots_folder
            )

        # Copy mesh and parameter files
        self._copy_mesh_files(mesh_info, results_subfolder)

        # Create project-level backup
        self._create_project_backup(
            mesh_identifier,
            results_subfolder,
            screenshot_path
        )

        print(f"Files saved to: {results_subfolder}")
        print(f"{'-'*20} Finished saving {'-'*20}\n")

        return results_subfolder

    def _save_screenshot(
        self,
        screenshot_data: Any,
        mesh_identifier: str,
        results_subfolder: Path,
        results_screenshots_folder: Path
    ) -> Path | None:
        """Save screenshot to both subfolder and screenshots folder."""
        from vispy.io import write_png

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        screenshot_filename = f"{mesh_identifier}_screenshot_{current_time}.png"
        screenshot_path = results_subfolder / screenshot_filename

        try:
            write_png(str(screenshot_path), screenshot_data)
            print(f"Screenshot saved: {screenshot_filename}")
            shutil.copy(screenshot_path, results_screenshots_folder)
            return screenshot_path
        except Exception as e:
            _logger.error(f"Error saving screenshot: {e}")
            return None

    def _copy_mesh_files(self, mesh_info: MeshDisplayInfo, dest_folder: Path) -> None:
        """Copy mesh and parameter files to destination."""
        aligned_mesh_path = Path(mesh_info.aligned_file_path)

        # Get parameter file path
        try:
            param_txt_filename = shape_browser_utils.get_param_txt_filename_from_mesh_identifier(
                mesh_info.mesh_identifier
            )
            parameter_file_path = self.folder_path / 'tooth_model_output' / f"{param_txt_filename}.txt"
        except ValueError:
            parameter_file_path = None

        # Copy files
        for file_path in [aligned_mesh_path, parameter_file_path]:
            if file_path and file_path.exists():
                try:
                    shutil.copy(file_path, dest_folder)
                except Exception as e:
                    _logger.error(f"Error copying {file_path}: {e}")

        # Create ToothMaker format file
        if parameter_file_path and parameter_file_path.exists():
            self._create_toothmaker_format(
                parameter_file_path,
                mesh_info.mesh_identifier,
                dest_folder
            )

    def _create_toothmaker_format(
        self,
        parameter_file_path: Path,
        mesh_identifier: str,
        dest_folder: Path
    ) -> None:
        """Create ToothMaker format parameter file."""
        try:
            format2_path = dest_folder / f"{mesh_identifier}_ToothMaker_format.txt"
            params, _ = toothmaker_format.parse_format1(
                str(parameter_file_path)
            )
            parts = mesh_identifier.split('__')
            num_iters = parts[-1] if len(parts) > 1 else "1000"
            toothmaker_format.write_format2(
                params, str(format2_path), num_iters
            )
        except Exception as e:
            _logger.warning(f"Error creating ToothMaker format: {e}")

    def _create_project_backup(
        self,
        mesh_identifier: str,
        results_subfolder: Path,
        screenshot_path: Path | None
    ) -> None:
        """Create backup at project root level."""
        try:
            project_saved_folder = get_project_root() / SAVED_BY_SHAPE_BROWSER_FOLDER
            project_subfolder = project_saved_folder / mesh_identifier
            project_screenshots_folder = project_saved_folder / ALL_SCREENSHOTS_SUBFOLDER

            for folder in [project_saved_folder, project_subfolder, project_screenshots_folder]:
                folder.mkdir(parents=True, exist_ok=True)

            shutil.copytree(results_subfolder, project_subfolder, dirs_exist_ok=True)

            if screenshot_path and screenshot_path.exists():
                shutil.copy(screenshot_path, project_screenshots_folder)
        except Exception as e:
            _logger.warning(f"Error creating project backup: {e}")
