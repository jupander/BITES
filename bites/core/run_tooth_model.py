import re
import string
import random
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Union, Optional, Tuple

logger = logging.getLogger(__name__)

from bites.utils.subprocess_utils import run_subprocess_with_cleanup
from bites.constants import RUN_IDENTIFIER_PREFIX, RUN_IDENTIFIER_RANDOM_LETTERS
from bites.mesh.read_coff import read_coff_mesh

def get_parameter_names_array() -> list:
    """
    Returns the hardcoded array of names of the parameters used by the tooth model, in the order they are expected.
    """
    # NOTE: the earlier version of this function was using the following hardcoded array, including 'iter' and an extra '-'(standing for n_cells, present in some text formats)
    #return ['Egr', 'Mgr', 'Rep', 'Swi', 'Adh', 'Act', 'Inh', '-', 'Sec', '-', 'Da', 'Di', 'Ds', '-', 'Int', 'Set', 'Boy', 'Dff', 'Bgr', 'Abi', 'Pbi', 'Lbi', 'Bbi', 'Rad', 'Deg', 'Dgr', 'Ntr', 'Bwi', 'Ina', 'uMgr', '-', 'iter']
    return ['Egr', 'Mgr', 'Rep', 'Swi', 'Adh', 'Act', 'Inh', '-', 'Sec', '-', 'Da', 'Di', 'Ds', '-', 'Int', 'Set', 'Boy', 'Dff', 'Bgr', 'Abi', 'Pbi', 'Lbi', 'Bbi', 'Rad', 'Deg', 'Dgr', 'Ntr', 'Bwi', 'Ina', 'uMgr']


def construct_model_parameters_string(trial_parameters_dict: dict) -> str:
    """
    Returns a string of the parameter values in the order and format expected by the tooth model. (e.g. '0.1;0.2;0.3;...')
    """
    # Get the parameter names in the order expected by the tooth model
    parameter_names_array = get_parameter_names_array()

    # add the empty '-' parameter to the trial_parameters_dict if it is not already there
    if '-' not in trial_parameters_dict:
        trial_parameters_dict['-'] = 0.0
    
    # Create a list of parameter values in the order expected by tooth_model
    param_values_array = [trial_parameters_dict[name] for name in parameter_names_array if name in trial_parameters_dict]

    expected_count = len(parameter_names_array)
    if len(param_values_array) != expected_count:
        raise ValueError(f"Expected {expected_count} parameters, but got {len(param_values_array)}. Please check the parameter names and values.")

    # Convert to a string with semicolon separators
    param_values_string = ';'.join(map(str, param_values_array))

    return param_values_string
    

def construct_run_identifier_string() -> str:
    """
    Construct a unique run identifier string.

    Format: "run{YYMMDD-HHMMSSmmm}{random_letters}"
    Example: "run240823-145700123abc"
    """
    # Timestamp: exclude sub-milliseconds and thousands/hundreds of the year
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")[2:-3]

    # Random letters for uniqueness (26^3 = 17,576 combinations)
    random_letters = ''.join(random.choices(string.ascii_lowercase, k=RUN_IDENTIFIER_RANDOM_LETTERS))

    tm_run_identifier = f"{RUN_IDENTIFIER_PREFIX}{current_time}{random_letters}"

    return tm_run_identifier


def construct_command_string(model_executable_path: Path, trial_parameters_dict: dict, iter_interval: int, tm_output_path: Path) -> Tuple[list, str]:
    """
    Constructs the command list for running the tooth model executable.

    Args:
        model_executable_path: Path to the tooth model executable.
        trial_parameters_dict: Dict of trial parameters (must contain 'iter_max' or 'iter').
        iter_interval: Spacing between data points in the simulation.
        tm_output_path: Path to the folder for tooth model output files.

    Returns:
        Tuple of (command list, run identifier string).
    """
    param_values_string = construct_model_parameters_string(trial_parameters_dict)
    tm_run_identifier = construct_run_identifier_string()

    # Determine max iterations: 'iter_max' is set by the optimization pipeline,
    # 'iter' is a legacy fallback used by the benchmark script.
    if 'iter_max' in trial_parameters_dict:
        max_iterations = trial_parameters_dict['iter_max']
    elif 'iter' in trial_parameters_dict:
        max_iterations = trial_parameters_dict['iter']
        logger.info(f"Using legacy 'iter' key ({max_iterations}) - consider using 'iter_max' instead")
    else:
        raise KeyError("trial_parameters_dict must contain 'iter_max' or 'iter'")
    n_intervals = int(max_iterations) // int(iter_interval)

    write_off_files_tm = str(1)  # write OFF files (1) or not (0)

    command = [
        str(model_executable_path),
        str(param_values_string),
        str(tm_run_identifier),
        str(iter_interval),
        str(n_intervals),
        str(write_off_files_tm),
        str(tm_output_path),
    ]

    return command, tm_run_identifier


def _get_last_iteration_from_output(tm_output_path: Path, tm_run_identifier: str) -> Optional[int]:
    """
    Scan the output directory for mesh files and return the highest iteration number found.

    Parameters:
    - tm_output_path: Path to the tooth model output directory
    - tm_run_identifier: The unique run identifier string

    Returns:
    - The highest iteration number found, or None if no mesh files exist
    """
    last_iter = None
    for file_path in tm_output_path.iterdir():
        if file_path.name.endswith(f'{tm_run_identifier}_.off'):
            match = re.search(r'(\d+)_+run', file_path.name)
            if match:
                iter_val = int(match.group(1))
                if last_iter is None or iter_val > last_iter:
                    last_iter = iter_val
    return last_iter


def _get_newest_mesh_file(tm_output_path: Path, tm_run_identifier: str) -> Optional[Tuple[Path, int]]:
    """
    Find the most recently created mesh file for the given run identifier.

    Parameters:
    - tm_output_path: Path to the tooth model output directory
    - tm_run_identifier: The unique run identifier string

    Returns:
    - Tuple of (file_path, iteration_number) for the newest file, or None if no files found
    """
    newest_file = None
    newest_time = None
    newest_iter = None

    for file_path in tm_output_path.iterdir():
        if file_path.name.endswith(f'{tm_run_identifier}_.off'):
            try:
                file_time = file_path.stat().st_mtime
                if newest_time is None or file_time > newest_time:
                    newest_time = file_time
                    newest_file = file_path
                    match = re.search(r'(\d+)_+run', file_path.name)
                    if match:
                        newest_iter = int(match.group(1))
            except OSError:
                continue

    if newest_file is not None:
        return (newest_file, newest_iter)
    return None


def _delete_if_corrupted_mesh(tm_output_path: Path, tm_run_identifier: str, logger) -> Optional[int]:
    """
    Check the most recently created mesh file and delete it only if it's corrupted.

    When the tooth model process is killed mid-write, the last mesh file may be
    truncated or corrupted. This function validates the newest file and removes
    it only if it cannot be read properly.

    Parameters:
    - tm_output_path: Path to the tooth model output directory
    - tm_run_identifier: The unique run identifier string
    - logger: Logger instance

    Returns:
    - The iteration number of the deleted file, or None if no file was deleted
    """
    result = _get_newest_mesh_file(tm_output_path, tm_run_identifier)
    if result is None:
        return None

    newest_file, newest_iter = result

    # Try to read the mesh - if it fails, the file is corrupted
    try:
        read_coff_mesh(newest_file, logger=logger)
        # Mesh is valid, don't delete
        logger.debug(f"Newest mesh file {newest_file.name} (iter {newest_iter}) is valid, keeping it.")
        return None
    except (ValueError, OSError) as read_error:
        # Mesh is corrupted, delete it
        logger.debug(f"Mesh file {newest_file.name} is corrupted: {read_error}")
        try:
            newest_file.unlink()
            logger.info(f"Deleted corrupted mesh file: {newest_file.name} (iter {newest_iter})")
            return newest_iter
        except OSError as e:
            logger.warning(f"Could not delete corrupted mesh file {newest_file.name}: {e}")
            return None


def run_tooth_model_as_subprocess(model_executable_path: Path, trial_parameters_dict: dict, tm_timeout: int, iter_interval: int, tm_output_path: Union[str, Path], logger, trial_number: Optional[int] = None) -> Tuple[bool, int, datetime.datetime, str]:
    """
    Prepares the command and runs the tooth model as a subprocess.
    
    Parameters:
    - config: Config object.
    - trial_parameters_dict: dict, the parameter values to be passed to the tooth model. If 'iter' is present as a key, it will override the config

    Returns:
    - process: subprocess.CompletedProcess, the completed process. None if an error occurred.
    - elapsed_time: float, the time it took to run the subprocess in seconds.
    - tm_run_identifier: str, the name of the output file (also returned in case of an error, to help finding the output files) e.g. 'run250520-141625928fiu'
    """

    trial_prefix = f"[Trial {trial_number}] " if trial_number is not None else ""

    # check that the model executable path is valid
    if not model_executable_path.exists():
        logger.error(f"Model executable path does not exist: {model_executable_path}")
        raise FileNotFoundError(f"Model executable path does not exist: {model_executable_path}")

    # ------------------------------------------------------------------
    # Prepare the command 

    command, tm_run_identifier = construct_command_string(model_executable_path, trial_parameters_dict, iter_interval, tm_output_path) # construct the command string for running the tooth model

    # Create the output directory if it does not exist
    tm_output_path = Path(tm_output_path)
    tm_output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run the subprocess 

    # time the execution
    start = datetime.datetime.now()
    
    # Run tooth_model as a subprocess
    # Uses Job Objects on Windows to ensure it's killed if the parent process dies
    try:
        process = run_subprocess_with_cleanup(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=tm_timeout)
        #if process: # "subprocess.run() will either return a CompletedProcess or raise an exception. It will never return None."
        #if process: # despite the above, I still somehow got "ERROR: Error running tooth model: cannot access local variable 'process' where it is not associated with a value"
        if process.returncode != 0:
            logger.warning(f"{trial_prefix}Tooth model subprocess exited with code {process.returncode}")
            stderr_output = process.stderr.decode('utf-8', errors='replace').strip() # "subprocess.run(..., stderr=subprocess.PIPE) guarantees that process.stderr will be a bytes object, never None."
            if stderr_output:
                logger.warning(f"Subprocess stderr: {stderr_output}")

    except subprocess.TimeoutExpired:
        # Check and delete corrupted mesh file if the process was killed mid-write
        deleted_iter = _delete_if_corrupted_mesh(tm_output_path, tm_run_identifier, logger)

        # Get the last completed iteration from output files
        last_iter = _get_last_iteration_from_output(tm_output_path, tm_run_identifier)

        # Get max iterations for the log message
        max_iterations = trial_parameters_dict.get('iter_max') or trial_parameters_dict.get('iter')

        # Build informative log message
        if last_iter is not None and max_iterations is not None:
            #logger.info(f"{datetime.datetime.now():%H:%M} Tooth model timeout after {tm_timeout/60:.1f} min, reached {last_iter}/{max_iterations} iterations.")
            logger.info(f"{trial_prefix}Tooth model timeout after {tm_timeout/60:.1f} min, reached {last_iter}/{max_iterations} iterations.")
        elif last_iter is not None:
            #logger.info(f"{datetime.datetime.now():%H:%M} Tooth model timeout after {tm_timeout/60:.1f} min, last completed iteration: {last_iter}.")
            logger.info(f"{trial_prefix}Tooth model timeout after {tm_timeout/60:.1f} min, last completed iteration: {last_iter}.")
        else:
            #logger.info(f"{datetime.datetime.now():%H:%M} Tooth model timeout after {tm_timeout/60:.1f} min, no mesh files found.")
            logger.info(f"{trial_prefix}Tooth model timeout after {tm_timeout/60:.1f} min, no output mesh files found.")

        if deleted_iter is not None:
            logger.debug(f"Removed corrupted mesh at iteration {deleted_iter}.")

        process = None
        
    except KeyboardInterrupt:
        logger.info(f"{trial_prefix}Tooth model was interrupted by the user.")
        process = None
        
    except Exception as e:
        logger.warning(f"{trial_prefix}Unexpected error when running the tooth model subprocess: {str(e)}")
        process = None

    # stop the timer
    end = datetime.datetime.now()
    elapsed = end - start
    tm_run_duration = round(elapsed.total_seconds()) # round the value to seconds

    if process:
        tm_run_complete = True
    else:
        tm_run_complete = False

    return tm_run_complete, tm_run_duration, start, tm_run_identifier
