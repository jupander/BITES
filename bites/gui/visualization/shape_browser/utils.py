"""
Utility functions for the shape browser.

Handles CSV loading, config parsing, and file path construction.
"""
from __future__ import annotations
import ast
import csv
import datetime
import json
import os
import re
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

import bites.gui.gui_utils as gui_utils
from bites.constants import algorithm_cuts_target, get_cut_mesh_filename

if TYPE_CHECKING:
    from typing import Any


class ShapeBrowserUtilsError(Exception):
    """Base exception for shape browser utility errors."""
    pass


class CSVLoadError(ShapeBrowserUtilsError):
    """Raised when CSV loading fails."""
    pass


class ConfigLoadError(ShapeBrowserUtilsError):
    """Raised when config loading fails."""
    pass


class FileNotFoundInFolderError(ShapeBrowserUtilsError):
    """Raised when expected files are not found in a folder."""
    pass


# Fix 18: Regex pattern for robust timestamp extraction
_TIMESTAMP_PATTERN = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')


def _parse_dict_string(value: str, row_id: Any = None) -> dict | None:
    """
    Parse a dict string from CSV, supporting both JSON and Python literal formats.

    New CSVs use JSON format (double quotes): {"key": "value"}
    Old CSVs use Python literal format (single quotes): {'key': 'value'}

    Args:
        value: String representation of a dict.
        row_id: Optional row identifier for warning messages.

    Returns:
        Parsed dict, or None if parsing fails or result is invalid.
    """
    if not value or not isinstance(value, str):
        return None

    # Try JSON first (new format)
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to ast.literal_eval (old Python literal format)
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    # Both methods failed
    row_info = f" for row {row_id}" if row_id is not None else ""
    print(f"Warning: Could not parse dict string{row_info}: {value[:100]}...")
    return None


def _extract_timestamp_from_filename(filename: str) -> datetime.datetime:
    """
    Extract timestamp from a filename using regex pattern matching.

    Fix 18: Uses regex instead of fragile character offset arithmetic.
    Matches YYYY-MM-DD_HH-MM-SS pattern anywhere in the filename.

    Args:
        filename: The filename to extract timestamp from.

    Returns:
        Parsed datetime, or datetime.min if no valid timestamp found.
    """
    match = _TIMESTAMP_PATTERN.search(filename)
    if match:
        try:
            return datetime.datetime.strptime(match.group(1), '%Y-%m-%d_%H-%M-%S')
        except ValueError:
            pass  # Invalid date values (e.g., month 13)
    # Fallback: return epoch for files without proper timestamp
    return datetime.datetime.min


def _load_single_csv(csv_file_path: Path) -> pd.DataFrame | None:
    """
    Load a single CSV file into a DataFrame.

    Args:
        csv_file_path: Path to the CSV file.

    Returns:
        DataFrame containing the CSV data, or None if loading fails.
    """
    try:
        dataframe = pd.read_csv(
            csv_file_path,
            delimiter=";",
            quoting=csv.QUOTE_NONNUMERIC,
            escapechar='\\',
            on_bad_lines='warn'
        )
        if dataframe.empty:
            print(f"Warning: CSV file is empty: {csv_file_path.name}")
            return None
        return dataframe
    except pd.errors.EmptyDataError:
        print(f"Warning: CSV file is empty: {csv_file_path.name}")
        return None
    except Exception as e:
        print(f"Warning: Failed to load CSV file '{csv_file_path.name}': {e}")
        return None


def get_dataframe_from_csv(folder_path: str | Path) -> pd.DataFrame:
    """
    Load all CSV files from the folder and merge them into a single DataFrame.

    Finds all CSV files in the folder (excluding ongoing_*.csv backups) and
    concatenates them. This allows viewing meshes from all optimization runs
    that contributed to the same study.

    Duplicates are removed based on 'chosen_mesh_identifier' to avoid showing
    the same trial multiple times if it appears in multiple CSVs.

    Args:
        folder_path: Path to the folder containing the CSV files.

    Returns:
        DataFrame containing merged data from all CSV files.

    Raises:
        CSVLoadError: If no CSV found or loading fails.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise CSVLoadError(f"Folder does not exist: {folder_path}")

    if not folder_path.is_dir():
        raise CSVLoadError(f"Path is not a directory: {folder_path}")

    # Find CSV files (exclude ongoing_*.csv backups)
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith('.csv') and not f.startswith('ongoing_')
    ]

    if not csv_files:
        raise CSVLoadError(f"No CSV file found in folder: {folder_path}")

    # Sort by timestamp (oldest first, so newer data takes precedence in dedup)
    csv_files_with_times = [
        (f, _extract_timestamp_from_filename(f)) for f in csv_files
    ]
    csv_files_with_times.sort(key=lambda x: x[1])  # Oldest first

    # Load all CSV files
    dataframes = []
    loaded_files = []
    for csv_file_name, _ in csv_files_with_times:
        csv_file_path = folder_path / csv_file_name
        df = _load_single_csv(csv_file_path)
        if df is not None:
            dataframes.append(df)
            loaded_files.append(csv_file_name)

    if not dataframes:
        raise CSVLoadError(f"No valid CSV data found in folder: {folder_path}")

    # Report what we loaded
    if len(loaded_files) == 1:
        print(f"Loaded CSV file '{loaded_files[0]}' from folder '{folder_path}'")
    else:
        print(f"Merging {len(loaded_files)} CSV files from folder '{folder_path}':")
        for fname in loaded_files:
            print(f"  - {fname}")

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Remove duplicates based on chosen_mesh_identifier (keep last = newest)
    if 'chosen_mesh_identifier' in combined_df.columns:
        original_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=['chosen_mesh_identifier'],
            keep='last'  # Keep the entry from the newest CSV
        )
        duplicates_removed = original_count - len(combined_df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate entries")

    print(f"Total trials loaded: {len(combined_df)}")

    return combined_df


def get_alignment_algorithm_names(folder_path: str | Path) -> list[str]:
    """
    Get the names of alignment algorithms from the config file in the folder.

    Args:
        folder_path: Path to the output folder containing the config file.

    Returns:
        List of alignment algorithm names.

    Raises:
        ConfigLoadError: If config file not found or doesn't contain algorithms.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise ConfigLoadError(f"Folder does not exist: {folder_path}")

    # Find YAML config files
    config_files = [f for f in os.listdir(folder_path) if f.endswith('.yaml')]

    if not config_files:
        raise ConfigLoadError(
            f"No config file found in folder '{folder_path}' "
            "(extension must be .yaml, not .yml)"
        )

    # Find newest config by timestamp in filename
    config_files_with_times = [
        (f, _extract_timestamp_from_filename(f)) for f in config_files
    ]
    config_files_with_times.sort(key=lambda x: x[1], reverse=True)

    config_file_name = config_files_with_times[0][0]
    config_file_path = folder_path / config_file_name

    try:
        config = gui_utils.load_yaml_config(str(config_file_path))
    except FileNotFoundError:
        raise ConfigLoadError(f"Config file not found: {config_file_path}")
    except Exception as e:
        raise ConfigLoadError(f"Failed to read config file '{config_file_name}': {e}")

    # Try new config structure first, then fall back to old structure
    alignment_algorithms = config.get('alignment_algorithms')

    if not alignment_algorithms:
        # Try old config structure for backwards compatibility
        optimization_config = config.get('optimization_config', {})
        alignment_algorithms = optimization_config.get('alignment_algorithms')

        if alignment_algorithms:
            print("Note: Using legacy config key 'optimization_config.alignment_algorithms'")

    if not alignment_algorithms:
        raise ConfigLoadError(
            f"No alignment algorithms found in config file '{config_file_name}'"
        )

    return alignment_algorithms


def get_param_txt_filename_from_mesh_identifier(mesh_identifier: str) -> str:
    """
    Get the parameter text filename from the mesh identifier.

    Args:
        mesh_identifier: Mesh identifier string, e.g., 'run240823-145700__7000'
                        Format: {run_identifier}__{iteration_number}

    Returns:
        Parameter text filename (without extension), e.g., '1000___run240623-185533494_'

    Raises:
        ValueError: If mesh_identifier format is invalid.
    """
    if '__' not in mesh_identifier:
        raise ValueError(
            f"Invalid mesh identifier format: '{mesh_identifier}'. "
            "Expected format: 'run_id__iteration_num'"
        )

    parts = mesh_identifier.split('__')
    if len(parts) < 2:
        raise ValueError(
            f"Invalid mesh identifier format: '{mesh_identifier}'. "
            "Expected format: 'run_id__iteration_num'"
        )

    tooth_model_run_identifier = parts[0]
    # Note: Only the first param file (iter = 1000) is saved
    return f"1000___{tooth_model_run_identifier}_"


def get_target_mesh_path(
    base_folder_path: str | Path,
    algorithm_name: str,
    mesh_identifier: str,
    cut_heights: dict[str, float] | None = None
) -> str:
    """
    Construct the target mesh path based on algorithm and mesh identifier.

    Args:
        base_folder_path: Base path to locate folders.
        algorithm_name: Name of the alignment algorithm.
        mesh_identifier: Identifier for the mesh file.
        cut_heights: Optional dict of cut heights by algorithm, e.g., {"STatac": 6.0}

    Returns:
        Full path to the target mesh file.

    Raises:
        FileNotFoundInFolderError: If target mesh file cannot be found.
    """
    base_folder_path = Path(base_folder_path)

    def get_original_mesh_path() -> str:
        """Get path to the original (uncut) target mesh."""
        folder_path = base_folder_path / 'original_target_mesh'

        if not folder_path.exists():
            raise FileNotFoundInFolderError(
                f"Original target mesh folder not found: {folder_path}"
            )

        files = list(folder_path.iterdir())
        if not files:
            raise FileNotFoundInFolderError(
                f"No files in original target mesh folder: {folder_path}"
            )

        return str(files[0])

    def get_cut_mesh_path() -> str:
        """Get path to the cut target mesh using cut height from CSV."""
        # cut_heights being None means missing data - this is a problem
        if cut_heights is None:
            print(f"Warning: cut_heights is None for '{mesh_identifier}' - using original target mesh")
            return get_original_mesh_path()

        # Individual algorithm value being None means no cut for that algorithm - this is valid
        cut_height = cut_heights.get(algorithm_name)
        if cut_height is None:
            return get_original_mesh_path()

        filename = get_cut_mesh_filename(cut_height)
        folder_path = base_folder_path / "cut_target_meshes"
        target_mesh_path = folder_path / filename

        if not target_mesh_path.exists():
            print(f"Warning: Cut mesh '{target_mesh_path}' not found. Using original.")
            return get_original_mesh_path()

        return str(target_mesh_path)

    # Use cut mesh for algorithms that cut the target, otherwise use original
    if algorithm_cuts_target(algorithm_name):
        target_mesh_path = get_cut_mesh_path()
    else:
        target_mesh_path = get_original_mesh_path()

    if not Path(target_mesh_path).exists():
        raise FileNotFoundInFolderError(
            f"Target mesh file does not exist: {target_mesh_path}"
        )

    return target_mesh_path


def list_off_files_in_order(
    column_to_sort_by: str,
    dataframe: pd.DataFrame,
    alignment_algorithms: list[str],
    base_folder_path: str | Path
) -> list[dict[str, Any]]:
    """
    List aligned mesh files (and corresponding target meshes) sorted by a column.

    Args:
        column_to_sort_by: Column name to sort by. Can be:
            - Algorithm name (e.g., 'STatac-CPD'): Sort by that algorithm's chamfer distance
            - 'loss': Sort by best loss
            - 'id', 'tm_run_duration': Sort by those columns
        dataframe: DataFrame containing the mesh data.
        alignment_algorithms: List of algorithm names.
        base_folder_path: Base folder path for locating files.

    Returns:
        List of dictionaries containing file info, sorted by the specified column.
        Each dict has keys: 'aligned_file_path', 'loss', 'chamfer_distance',
        'algorithm', 'target_mesh_path', 'id', 'best_algorithm',
        'index_in_sorted_stack', 'chosen_mesh_identifier', 'run_duration'

    Raises:
        KeyError: If the sort column doesn't exist in the dataframe.
    """
    start_time = datetime.datetime.now()
    base_folder_path = Path(base_folder_path)

    if column_to_sort_by not in dataframe.columns:
        raise KeyError(f"Column '{column_to_sort_by}' not found in dataframe")

    # Warn once if cut heights column is missing from CSV entirely
    if 'used_cut_heights' not in dataframe.columns:
        print("Warning: 'used_cut_heights' column missing from CSV - using original target meshes")

    # Sort the dataframe
    dataframe = dataframe.sort_values(column_to_sort_by, ascending=True)

    off_file_dicts: list[dict[str, Any]] = []

    # Fix 16: Use itertuples() instead of iterrows() for better performance
    # itertuples() is significantly faster for large DataFrames
    for row in dataframe.itertuples():
        try:
            file_dict = _extract_file_info_from_tuple(
                row=row,
                column_to_sort_by=column_to_sort_by,
                alignment_algorithms=alignment_algorithms,
                base_folder_path=base_folder_path
            )
            if file_dict:
                off_file_dicts.append(file_dict)
        except Exception as e:
            print(f"Warning: Failed to process row {row.Index}: {e}")
            traceback.print_exc()
            continue

    elapsed = (datetime.datetime.now() - start_time).total_seconds()

    if off_file_dicts:
        print(f"Loaded {len(off_file_dicts)} meshes sorted by '{column_to_sort_by}' ({elapsed:.2f}s)")
    else:
        print(f"Warning: No meshes found when sorting by '{column_to_sort_by}'")

    return off_file_dicts


def _extract_file_info_from_tuple(
    row: Any,
    column_to_sort_by: str,
    alignment_algorithms: list[str],
    base_folder_path: Path
) -> dict[str, Any] | None:
    """
    Fix 16: Extract file information from a namedtuple row (from itertuples()).

    This is faster than _extract_file_info which uses pd.Series from iterrows().

    Args:
        row: DataFrame row as namedtuple from itertuples().
        column_to_sort_by: Column being used for sorting.
        alignment_algorithms: List of algorithm names.
        base_folder_path: Base folder path.

    Returns:
        Dictionary with file info, or None if extraction fails.
    """
    index = row.Index

    # Determine algorithm and distances based on sort column
    if column_to_sort_by in alignment_algorithms:
        chamfer_distance = getattr(row, column_to_sort_by, None)
        loss = getattr(row, 'loss', chamfer_distance)
        algorithm_name = column_to_sort_by
    else:
        loss = getattr(row, 'loss', 0)
        chamfer_distance = loss
        algorithm_name = getattr(
            row, 'best_algorithm',
            alignment_algorithms[0] if alignment_algorithms else 'unknown'
        )

    # Get mesh identifier
    mesh_identifier = getattr(row, 'chosen_mesh_identifier', None)
    if not mesh_identifier:
        print(f"Warning: Row {index} missing 'chosen_mesh_identifier'")
        return None

    # Get other fields with defaults
    row_id = getattr(row, 'id', index)
    best_algorithm = getattr(row, 'best_algorithm', algorithm_name)
    run_duration = getattr(row, 'tm_run_duration', 0)

    # Parse cut heights from CSV (stored as JSON string, e.g., '{"STatac": 6.0}')
    # Supports both JSON (new format) and Python literal (old format) for backwards compatibility.
    # The value must be a dict - None for the entire value is invalid.
    # Individual algorithm values within the dict can be None (meaning no cut for that algorithm).
    cut_heights = None
    if hasattr(row, 'used_cut_heights'):
        cut_heights_raw = row.used_cut_heights
        if pd.isna(cut_heights_raw):
            # Cell is empty/NaN - missing data, warn
            print(f"Warning: 'used_cut_heights' is empty for row {index} - using original target mesh")
        else:
            parsed = _parse_dict_string(cut_heights_raw, row_id=index)
            if parsed is not None:
                # Validate dict has string keys and numeric/None values
                if all(
                    isinstance(k, str) and (v is None or isinstance(v, (int, float)))
                    for k, v in parsed.items()
                ):
                    cut_heights = parsed
                else:
                    print(f"Warning: cut_heights for row {index} has invalid values: {parsed}")

    # Construct file paths
    aligned_mesh_folder = f"{algorithm_name}_aligned_meshes"
    aligned_off_file_path = str(
        base_folder_path / aligned_mesh_folder / f"{algorithm_name}_aligned_{mesh_identifier}.ply"
    )

    # Fix 17: Validate aligned file exists before returning
    if not Path(aligned_off_file_path).exists():
        print(f"Warning: Aligned mesh not found: {aligned_off_file_path}")
        return None

    try:
        target_mesh_path = get_target_mesh_path(
            base_folder_path, algorithm_name, mesh_identifier, cut_heights
        )
    except FileNotFoundInFolderError as e:
        print(f"Warning: {e}")
        return None

    return {
        'aligned_file_path': aligned_off_file_path,
        'loss': loss,
        'chamfer_distance': chamfer_distance,
        'algorithm': algorithm_name,
        'target_mesh_path': target_mesh_path,
        'id': row_id,
        'best_algorithm': best_algorithm,
        'index_in_sorted_stack': index,
        'chosen_mesh_identifier': mesh_identifier,
        'run_duration': run_duration
    }


def validate_folder_for_shape_browser(folder_path: str | Path) -> list[str]:
    """
    Check whether a folder contains the files needed by the shape browser.

    Returns a list of problems found.  An empty list means the folder looks
    valid.  The checks are intentionally lightweight (no CSV parsing) so that
    they can run in the main process before spawning the subprocess.
    """
    folder_path = Path(folder_path)
    problems: list[str] = []

    if not folder_path.exists():
        problems.append(f"Folder does not exist: {folder_path}")
        return problems

    if not folder_path.is_dir():
        problems.append(f"Path is not a directory: {folder_path}")
        return problems

    # 1. At least one results CSV (excluding ongoing_*.csv backups)
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith('.csv') and not f.startswith('ongoing_')
    ]
    if not csv_files:
        problems.append("No results CSV file found.")

    # 2. At least one config YAML
    yaml_files = [f for f in os.listdir(folder_path) if f.endswith('.yaml')]
    if not yaml_files:
        problems.append("No config YAML file found.")

    # 3. original_target_mesh folder with at least one mesh file
    target_mesh_dir = folder_path / 'original_target_mesh'
    if not target_mesh_dir.is_dir():
        problems.append("Missing 'original_target_mesh' subfolder.")
    else:
        mesh_files = [
            f for f in os.listdir(target_mesh_dir)
            if f.endswith(('.ply', '.off', '.stl', '.obj', '.glb', '.gltf', '.3mf'))
        ]
        if not mesh_files:
            problems.append("No mesh files in 'original_target_mesh' subfolder.")

    # 4. At least one *_aligned_meshes folder with mesh files
    aligned_dirs = [
        d for d in os.listdir(folder_path)
        if (folder_path / d).is_dir() and d.endswith('_aligned_meshes')
    ]
    if not aligned_dirs:
        problems.append(
            "No aligned mesh folders found "
            "(expected folders ending with '_aligned_meshes')."
        )
    else:
        has_meshes = any(
            any(f.endswith(('.ply', '.off', '.stl', '.obj', '.glb', '.gltf', '.3mf'))
                for f in os.listdir(folder_path / d))
            for d in aligned_dirs
        )
        if not has_meshes:
            problems.append("Aligned mesh folders exist but contain no mesh files.")

    return problems


def _extract_file_info(
    row: pd.Series,
    index: Any,
    column_to_sort_by: str,
    alignment_algorithms: list[str],
    base_folder_path: Path
) -> dict[str, Any] | None:
    """
    Extract file information from a dataframe row.

    Note: This function is kept for backwards compatibility.
    Use _extract_file_info_from_tuple with itertuples() for better performance.

    Args:
        row: DataFrame row.
        index: Row index.
        column_to_sort_by: Column being used for sorting.
        alignment_algorithms: List of algorithm names.
        base_folder_path: Base folder path.

    Returns:
        Dictionary with file info, or None if extraction fails.
    """
    # Determine algorithm and distances based on sort column
    if column_to_sort_by in alignment_algorithms:
        chamfer_distance = row[column_to_sort_by]
        loss = row.get('loss', chamfer_distance)
        algorithm_name = column_to_sort_by
    else:
        loss = row.get('loss', 0)
        chamfer_distance = loss
        algorithm_name = row.get('best_algorithm', alignment_algorithms[0] if alignment_algorithms else 'unknown')

    # Get mesh identifier
    mesh_identifier = row.get('chosen_mesh_identifier')
    if not mesh_identifier:
        print(f"Warning: Row {index} missing 'chosen_mesh_identifier'")
        return None

    # Get other fields with defaults
    row_id = row.get('id', index)
    best_algorithm = row.get('best_algorithm', algorithm_name)
    run_duration = row.get('tm_run_duration', 0)

    # Parse cut heights from CSV (stored as JSON string, e.g., '{"STatac": 6.0}')
    # Supports both JSON (new format) and Python literal (old format) for backwards compatibility.
    # The value must be a dict - None for the entire value is invalid.
    # Individual algorithm values within the dict can be None (meaning no cut for that algorithm).
    cut_heights = None
    if 'used_cut_heights' in row.index:
        cut_heights_raw = row['used_cut_heights']
        if pd.isna(cut_heights_raw):
            # Cell is empty/NaN - missing data, warn
            print(f"Warning: 'used_cut_heights' is empty for row {index} - using original target mesh")
        else:
            parsed = _parse_dict_string(cut_heights_raw, row_id=index)
            if parsed is not None:
                # Validate dict has string keys and numeric/None values
                if all(
                    isinstance(k, str) and (v is None or isinstance(v, (int, float)))
                    for k, v in parsed.items()
                ):
                    cut_heights = parsed
                else:
                    print(f"Warning: cut_heights for row {index} has invalid values: {parsed}")

    # Construct file paths
    aligned_mesh_folder = f"{algorithm_name}_aligned_meshes"
    aligned_off_file_path = str(
        base_folder_path / aligned_mesh_folder / f"{algorithm_name}_aligned_{mesh_identifier}.ply"
    )

    # Fix 17: Validate aligned file exists before returning
    if not Path(aligned_off_file_path).exists():
        print(f"Warning: Aligned mesh not found: {aligned_off_file_path}")
        return None

    try:
        target_mesh_path = get_target_mesh_path(
            base_folder_path, algorithm_name, mesh_identifier, cut_heights
        )
    except FileNotFoundInFolderError as e:
        print(f"Warning: {e}")
        return None

    return {
        'aligned_file_path': aligned_off_file_path,
        'loss': loss,
        'chamfer_distance': chamfer_distance,
        'algorithm': algorithm_name,
        'target_mesh_path': target_mesh_path,
        'id': row_id,
        'best_algorithm': best_algorithm,
        'index_in_sorted_stack': index,
        'chosen_mesh_identifier': mesh_identifier,
        'run_duration': run_duration
    }
