"""
Output Manager - Output folder creation, mesh I/O, and temporary file cleanup.
"""

import re
import time
import shutil
import logging
import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import bites.mesh.read_coff as read_coff
import bites.mesh.trimesh_utils as trimesh_utils
from bites.config.config_io import save_yaml_config, config_to_serializable_dict
from bites.constants import (
    algorithm_cuts_target,
    get_cut_mesh_filename,
    format_mesh_identifier,
)


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.', _depth: int = 0, max_depth: int = 20) -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single-level dictionary with compound keys.

    Parameters:
    - d: The dictionary to flatten
    - parent_key: Prefix for keys (used in recursion)
    - sep: Separator between key levels
    - _depth: Current recursion depth (internal use)
    - max_depth: Maximum recursion depth to prevent stack overflow

    Returns:
    - Flattened dictionary
    """
    if _depth >= max_depth:
        # Return the value as-is to prevent infinite recursion
        return {parent_key: d} if parent_key else {'_nested': d}

    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep, _depth=_depth + 1, max_depth=max_depth))
        else:
            items[new_key] = v
    return items


def stop_the_timer(start: float) -> Tuple[str, float]:
    """Stops the timer and returns the elapsed time in a human-readable format."""
    end = time.time()
    elapsed_time_in_seconds = end - start
    end_time_in_24h_format = time.strftime("%H:%M:%S")
    if elapsed_time_in_seconds > 3600:
        elapsed_time = elapsed_time_in_seconds / 3600
        return f"Finished at {end_time_in_24h_format}\nTotal elapsed time: {elapsed_time:.1f} hours.", elapsed_time_in_seconds
    elif elapsed_time_in_seconds > 60:
        elapsed_time = elapsed_time_in_seconds / 60
        return f"Finished at {end_time_in_24h_format}\nTotal elapsed time: {elapsed_time:.1f} minutes.", elapsed_time_in_seconds
    else:
        return f"Finished at {end_time_in_24h_format}\nTotal elapsed time: {elapsed_time_in_seconds:.2f} seconds.", elapsed_time_in_seconds


def _create_aligned_mesh_folders(results_folder_path: Path, alignment_algorithms: list) -> dict:
    """
    Create subfolders for each alignment algorithm's aligned meshes.

    Parameters:
    - results_folder_path: Path to the output folder
    - alignment_algorithms: List of alignment algorithm names

    Returns:
    - dict mapping algorithm names to their folder paths (as Path objects)
    """
    folder_paths = {}
    for algorithm in alignment_algorithms:
        folder_path = results_folder_path / f'{algorithm}_aligned_meshes'
        folder_path.mkdir(parents=True, exist_ok=True)
        folder_paths[algorithm] = folder_path
    return folder_paths


def _create_cut_target_folder(results_folder_path: Path) -> Path:
    """
    Create the shared folder for cut target meshes.

    Parameters:
    - results_folder_path: Path to the output folder

    Returns:
    - Path to the cut target meshes folder
    """
    folder_path = results_folder_path / "cut_target_meshes"
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def _export_aligned_mesh(aligned_mesh, output_path: Path, algorithm: str, iteration: int, logger) -> bool:
    """
    Validate and export a single aligned mesh.

    Parameters:
    - aligned_mesh: The trimesh object to export
    - output_path: Full path for the output file
    - algorithm: Name of the alignment algorithm (for logging)
    - iteration: Iteration number (for logging)
    - logger: Logger instance

    Returns:
    - True if export succeeded, False otherwise
    """
    if not trimesh_utils.validate_trimesh(aligned_mesh, f"aligned mesh for {algorithm} at iter {iteration}"):
        logger.warning(f"_export_aligned_mesh: Mesh for {algorithm} at iter {iteration} is not valid, skipping.")
        return False

    try:
        aligned_mesh.export(output_path)
        return True
    except Exception as e:
        logger.error(f"_export_aligned_mesh: Could not write mesh to '{output_path}': {e}")
        return False


def _export_cut_target_mesh(target_mesh, cut_height: float, output_folder: Path, algorithm: str, logger) -> bool:
    """
    Export cut target mesh if it doesn't already exist.

    Parameters:
    - target_mesh: The cut target mesh to export
    - cut_height: The cut height value (used for filename)
    - output_folder: Folder path for cut target meshes
    - algorithm: Name of the alignment algorithm (for logging)
    - logger: Logger instance

    Returns:
    - True if export succeeded or file already exists, False on error
    """
    if cut_height is None:
        logger.debug(f"_export_cut_target_mesh: Cut height for {algorithm} is None, skipping.")
        return False

    filename = get_cut_mesh_filename(cut_height)
    output_path = output_folder / filename

    # Only write if file doesn't already exist
    if output_path.exists():
        return True

    try:
        target_mesh.export(output_path)
        return True
    except Exception as e:
        logger.error(f"_export_cut_target_mesh: Error writing cut mesh for {algorithm} at height {cut_height}: {e}")
        return False


def write_aligned_meshes_to_files(
    config,
    aligned_meshes_by_iter_dict: dict,
    target_meshes_by_iter_dict: dict,
    best_chamfer_distance_by_iter_dict: dict,
    iter_of_chosen_mesh: int,
    tooth_model_run_identifier: str,
    results_folder_path,
    used_cut_heights: dict
) -> dict:
    """
    Write the aligned meshes and target meshes to files in the output folder.

    Parameters:
    - config: Configuration object with logger
    - aligned_meshes_by_iter_dict: Dict mapping iteration -> algorithm -> aligned mesh
    - target_meshes_by_iter_dict: Dict mapping iteration -> algorithm -> target mesh
    - best_chamfer_distance_by_iter_dict: Dict mapping iteration -> best chamfer distance
    - iter_of_chosen_mesh: The iteration number of the chosen (best) mesh
    - tooth_model_run_identifier: Unique identifier for this tooth model run
    - results_folder_path: Path to the output folder
    - used_cut_heights: Dict mapping algorithm name -> cut height used for that algorithm
                        (should be from the chosen mesh's iteration)

    Returns:
    - dict mapping iteration number to mesh identifier string
    """
    logger = config.logger

    # Get the aligned meshes for the chosen iteration
    aligned_meshes = aligned_meshes_by_iter_dict.get(iter_of_chosen_mesh)
    if not aligned_meshes:
        logger.warning(f"write_aligned_meshes_to_files: No aligned meshes found for iter {iter_of_chosen_mesh}")
        return {}

    # Create output directories
    aligned_folders = _create_aligned_mesh_folders(results_folder_path, aligned_meshes.keys())
    cut_target_folder = _create_cut_target_folder(results_folder_path)

    # Generate mesh identifier
    mesh_identifier = format_mesh_identifier(tooth_model_run_identifier, iter_of_chosen_mesh)

    # Export each algorithm's aligned mesh and (if applicable) cut target mesh
    for algorithm, aligned_mesh in aligned_meshes.items():
        # Export aligned mesh
        aligned_output_path = aligned_folders[algorithm] / f"{algorithm}_aligned_{mesh_identifier}.ply"
        _export_aligned_mesh(aligned_mesh, aligned_output_path, algorithm, iter_of_chosen_mesh, logger)

        # Export cut target mesh for algorithms that cut the target
        if algorithm_cuts_target(algorithm):
            cut_height = used_cut_heights.get(algorithm)
            target_mesh = target_meshes_by_iter_dict.get(iter_of_chosen_mesh, {}).get(algorithm)
            if target_mesh is not None:
                _export_cut_target_mesh(target_mesh, cut_height, cut_target_folder, algorithm, logger)

    return {iter_of_chosen_mesh: mesh_identifier}


def gather_mesh_by_iter_dict(tm_run_identifier: str, tm_output_path: Union[str, Path], logger=None) -> dict:
    """List all the OFF files in the given folder that match the given tooth_model run identifier.
    Returns a dictionary with the number of iterations as keys and the corresponding meshes as values."""

    # Use a module logger if none provided
    if logger is None:
        logger = logging.getLogger(__name__)

    tm_output_path = Path(tm_output_path)
    meshes_dict = {}

    for file_path in tm_output_path.iterdir():
        if file_path.name.endswith(f'{tm_run_identifier}_.off'):
            match = re.search(r'(\d+)_+(run.+)_+\.off$', file_path.name)  # Extract the relevant part of the filename
            if match:
                iter_value = int(match.group(1))
            else:
                logger.warning(f'Filename "{file_path.name}" does not match the expected format: "(n_iter)(_)_{tm_run_identifier}_.off"')
                continue

            try:
                mesh = read_coff.read_coff_mesh(file_path, logger)  # load the mesh using a custom function that can handle (and clean) COFF files written by tooth_model
                meshes_dict[iter_value] = mesh
            except Exception as e:
                logger.error(f"Could not read mesh from file '{file_path}': {e}")
                continue

    logger.debug(f"gather_mesh_by_iter_dict: Found {len(meshes_dict)} meshes matching the identifier '{tm_run_identifier}' in the folder '{tm_output_path}'")

    return meshes_dict


def copy_original_target_mesh(target_shape_path: Path, results_folder_path: Path) -> None:
    """
    Ensures that the 'original_target_mesh' subfolder exists in the output folder,
    and copies the file to that subfolder if it doesn't already exist there.

    Parameters:
    - target_shape_path (str or Path): Path to the file to copy.
    - results_folder_path (str or Path): Path to the output folder.
    """
    target_mesh_subfolder = results_folder_path / 'original_target_mesh'
    target_mesh_subfolder.mkdir(parents=True, exist_ok=True)
    destination_path = target_mesh_subfolder / target_shape_path.name
    if not destination_path.exists():
        shutil.copy(target_shape_path, destination_path)


def create_output_folder(config, output_root: Path) -> Tuple[Path, str]:
    """
    Create the output folder for an optimization study.

    Returns:
    - Tuple of (results_folder_path, optimization_run_id)
    """
    logger = config.logger if hasattr(config, 'logger') else logging.getLogger(__name__)

    # Create the output folder for the study (if it doesn't exist)
    results_folder_path = output_root / config.study_name
    results_folder_path.mkdir(parents=True, exist_ok=True)

    # Define optimization run identifier (e.g. "study_name_2023-10-01_12-00-00")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    optimization_run_id = f"{config.study_name}_{current_time}"

    # Save the used config file to the output folder
    config_dict = config_to_serializable_dict(config)
    new_config_file_name = f"config_{optimization_run_id}.yaml"
    new_config_file_path = results_folder_path / new_config_file_name

    try:
        save_yaml_config(new_config_file_path, config_dict, template=False, logger=logger)
    except Exception as e:
        logger.error(f"Error saving config file: {e}")

    # Save a copy of the target mesh to the output folder (if it doesn't exist)
    copy_original_target_mesh(config.target_shape_path, results_folder_path)

    return results_folder_path, optimization_run_id


def remove_temporary_files(tm_run_identifier: str, tm_output_path: Union[str, Path], config) -> None:
    """
    Remove unneeded files created by the tooth model run, to save space.
    """
    logger = config.logger if hasattr(config, 'logger') else logging.getLogger(__name__)
    tm_output_path = Path(tm_output_path)

    # Spare just one version of the parameter txt files output by the tooth model for each run
    # (it creates one for each iter value, only differing in n_cells)
    try:
        remove_tm_output_txt_files(tm_run_identifier, tm_output_path, logger=logger)
    except Exception as e:
        raise RuntimeError(f"Error removing temporary .txt files: {e}")

    # Remove the original unaligned meshes generated by the tooth model (take a lot of space)
    if not config.save_original_tooth_model_meshes:
        try:
            remove_tm_output_meshes(tm_run_identifier, tm_output_path, logger=logger)
        except Exception as e:
            raise RuntimeError(f"Error removing temporary .off files: {e}")

    # Delete the .dad files generated by the tooth model (take a lot of space)
    try:
        remove_tm_output_dad_files(tm_run_identifier, tm_output_path, logger=logger)
    except Exception as e:
        raise RuntimeError(f"Error removing temporary .dad files: {e}")


def remove_tm_output_meshes(tm_run_identifier: str, tm_output_path: Union[str, Path], logger=None) -> None:
    """Remove all the OFF files in the given folder that match the given tooth model run identifier."""
    if logger is None:
        logger = logging.getLogger(__name__)
    tm_output_path = Path(tm_output_path)

    for file_path in tm_output_path.iterdir():
        if file_path.name.endswith(f'{tm_run_identifier}_.off'):
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Error removing file {file_path}: {e}")
                continue


def remove_tm_output_txt_files(tm_run_identifier: str, tm_output_path: Union[str, Path], logger=None) -> None:
    """Remove all the TXT files in the given folder that match the given tooth model run identifier, save for the one that has iter of 1000."""
    if logger is None:
        logger = logging.getLogger(__name__)
    tm_output_path = Path(tm_output_path)

    for file_path in tm_output_path.iterdir():
        if file_path.name.endswith(f'{tm_run_identifier}_.txt'):
            if '1000' in file_path.name:  # Keep one version for reference
                continue  # skip this file
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Error removing file {file_path}: {e}")
                continue


def remove_tm_output_dad_files(tm_run_identifier: str, tm_output_path: Union[str, Path], logger=None) -> None:
    """Remove all the DAD files in the given folder that match the given tooth model run identifier."""
    if logger is None:
        logger = logging.getLogger(__name__)
    tm_output_path = Path(tm_output_path)

    for file_path in tm_output_path.iterdir():
        if file_path.name.endswith(f'{tm_run_identifier}_.dad'):
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Error removing file {file_path}: {e}")
                continue


def calculate_sim_duration_by_iter(tm_run_identifier: str, tm_output_path: Union[str, Path], start_time: datetime.datetime, logger=None) -> dict:
    """
    List all the OFF files in the given folder that match the given tooth_model run identifier.
    Returns a dictionary with the number of iterations as keys and duration_from_start as values.
    (Based on the gather_mesh_by_iter_dict function)

    Parameters:
    - tm_run_identifier: A unique string used in the filename to identify the run
    - tm_output_path: Path to the directory containing the mesh files
    - start_time: A datetime object representing the reference start time
    - logger: Optional logger instance

    Returns:
    - duration_dict: A dictionary with iteration values as keys and the corresponding durations as values
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    tm_output_path = Path(tm_output_path)
    duration_dict = {}  # {iter_value: duration_from_start}

    for file_path in tm_output_path.iterdir():
        if file_path.name.endswith(f'{tm_run_identifier}_.off'):
            match = re.search(r'(\d+)_+(run.+)_+\.off$', file_path.name)
            if match:
                iter_value = int(match.group(1))
            else:
                logger.warning(f'Filename "{file_path.name}" does not match expected format: "(n_iter)(_)_{tm_run_identifier}_.off"')
                continue

            try:
                # Use st_mtime (modification time) for cross-platform consistency
                # st_ctime is creation time on Windows but inode change time on Unix
                file_timestamp = file_path.stat().st_mtime
                file_time = datetime.datetime.fromtimestamp(file_timestamp)
                duration = (file_time - start_time).total_seconds()
                duration_dict[iter_value] = duration
            except Exception as e:
                logger.warning(f"Error calculating simulation time for {file_path.name}: {e}")
                continue

    return duration_dict
