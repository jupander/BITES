"""
Shared constants and utility functions for BITES.

This module centralizes:
- Naming formats (identifiers, filenames, database names)
- Default paths and folder names
- Limits and thresholds
- Technical setting defaults
- Algorithm capability detection

Note: Base folder names (results, configs, etc.) are defined in bites.paths.
This module imports from paths.py to build derived paths, avoiding duplication.
"""

from bites.paths import DEFAULT_CONFIGS_FOLDER, DEFAULT_RESULTS_FOLDER

# =============================================================================
# NAMING FORMATS
# =============================================================================

# Format for mesh identifiers: "{run_identifier}__{iteration_num}"
# Example: "run240823-145700abc__7000"
MESH_IDENTIFIER_FORMAT = "{run_id}__{iter_num}"

# Format for run identifiers: "run{YYMMDD-HHMMSSmmm}{random_letters}"
# Example: "run240823-145700abc"
RUN_IDENTIFIER_PREFIX = "run"
RUN_IDENTIFIER_RANDOM_LETTERS = 3

# Format for database names: "db_{study_name}"
DATABASE_NAME_PREFIX = "db_"


# =============================================================================
# DEFAULT PATHS AND FOLDER NAMES
# =============================================================================

# Template file for config injection
TEMPLATE_FILENAME = "template.yaml"

# Default example config file path (relative to project root)
# Used as fallback when no config path is stored in registry
DEFAULT_EXAMPLE_CONFIG_PATH = f"{DEFAULT_CONFIGS_FOLDER}/optimization_config.yaml"

# Default example output folder path (relative to project root)
# Used as fallback when no output folder is stored in registry
DEFAULT_EXAMPLE_RESULTS_FOLDER = f"{DEFAULT_RESULTS_FOLDER}/example_seal"


# =============================================================================
# TECHNICAL SETTING DEFAULTS
# These are used when technical settings are not specified in config
# =============================================================================

# Default tooth model executable (relative to bin folder)
DEFAULT_MODEL_EXECUTABLE = "tooth_model.exe"

# Default subfolder name for tooth model output within the output folder
DEFAULT_TM_SUBFOLDER_NAME = "tooth_model_output"

# Default folder for centralized backup of optimization results (CSV and database files).
# This folder in the project root collects copies of results that are otherwise only
# found in their individual output folders, making it easier to access all results
# in one place. Leave empty to disable this backup feature.
DEFAULT_RESULTS_BACKUP_FOLDER = "results_backup"

# Folder name for shapes saved by the shape browser (used in output folder and project root)
SAVED_BY_SHAPE_BROWSER_FOLDER = "Saved_by_Shape_Browser"

# Subfolder name for all screenshots within Saved_by_Shape_Browser
ALL_SCREENSHOTS_SUBFOLDER = "All_screenshots"

# Default iteration interval for mesh saving
DEFAULT_ITER_INTERVAL = 1000


# =============================================================================
# Config Editor GUI CONSTANTS
# =============================================================================

# Maximum number of iterations that can be set in the UI
MAX_POSSIBLE_ITERATIONS = 50000

# Parameters after which to add a divider line (search range and fixed param value windows)
DIVIDER_AFTER_PARAMS_TOOTHMODEL = ['Adh', 'Sec', 'Ds', 'Bgr', 'Bbi', 'Ntr', 'uMgr']
DIVIDER_AFTER_PARAMS_ALPHABETICAL = ['Adh', 'Bwi', 'Ds', 'Int', 'Pbi', 'Rep', 'uMgr']

DEFAULT_CONFIG_FILE_NAME = 'example_config.yaml'

IMPLEMENTED_ALIGNMENT_ALGORITHMS = ['STatac-AB-CPD', 'STatac-C-CPD', 'STatac-AB', 'STatac-C', 'ST']


# =============================================================================
# MESH FILE CONSTANTS
# =============================================================================

# Precision for rounding cut heights (number of decimal places)
CUT_HEIGHT_PRECISION = 1

# Filename pattern for cut target meshes
CUT_MESH_FILENAME_TEMPLATE = "target_cut_at_height_{height}.ply"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_mesh_identifier(run_id: str, iter_num: int) -> str:
    """
    Generate a mesh identifier from run ID and iteration number.

    Example: "run240823-145700abc__7000"

    Parameters:
        run_id: The run identifier string
        iter_num: The iteration number

    Returns:
        Formatted mesh identifier string
    """
    return MESH_IDENTIFIER_FORMAT.format(run_id=run_id, iter_num=iter_num)


def format_database_name(study_name: str) -> str:
    """
    Generate a database name from study name.

    Example: "db_my_study"

    Parameters:
        study_name: The study name

    Returns:
        Formatted database name (still needs sanitization)
    """
    return f"{DATABASE_NAME_PREFIX}{study_name}"


def get_cut_mesh_filename(height: float) -> str:
    """
    Generate the filename for a cut target mesh at the given height.

    Parameters:
        height: The cut height value

    Returns:
        Filename string like "target_cut_at_height_6.0.ply"
    """
    rounded = round(float(height), CUT_HEIGHT_PRECISION)
    return CUT_MESH_FILENAME_TEMPLATE.format(height=rounded)


def algorithm_cuts_target(algorithm_name: str) -> bool:
    """
    Check if an alignment algorithm modifies (cuts) the target mesh.

    Currently, all STatac variants cut the target mesh to improve alignment.

    Parameters:
        algorithm_name: Name of the alignment algorithm

    Returns:
        True if the algorithm cuts the target mesh, False otherwise
    """
    if not isinstance(algorithm_name, str):
        return False
    return 'statac' in algorithm_name.lower()


# =============================================================================
# SHAPE BROWSER / VISUALIZATION CONSTANTS
# =============================================================================

class ShapeBrowserConfig:
    """Configuration constants for the shape browser visualization."""

    # -- Slideshow timing (in seconds) --
    SLIDESHOW_MIN_DISPLAY_TIME = 0.025  # Fastest slideshow speed
    SLIDESHOW_MAX_DISPLAY_TIME = 3.2    # Slowest slideshow speed
    SLIDESHOW_DEFAULT_DISPLAY_TIME = 2.0
    SLIDESHOW_SPEED_MULTIPLIER = 2.0    # Factor for speed changes

    # -- Window dimensions --
    RUGPLOT_WINDOW_WIDTH = 1000
    RUGPLOT_WINDOW_HEIGHT = 800
    RUGPLOT_WINDOW_X = 850
    RUGPLOT_WINDOW_Y = 20
    CONSOLE_WIDGET_HEIGHT = 200

    # -- Fonts --
    DEFAULT_FONT_FAMILY = "Verdana"
    DEFAULT_FONT_SIZE = 12

    # -- Rugplot layout --
    MAX_RUGPLOTS_PER_COLUMN = 10

    # -- Rugplot axis positioning (values determined by trial and error) --
    RUGPLOT_XAXIS_MIN = 0.045   # Left margin for axis line
    RUGPLOT_XAXIS_MAX = 0.952   # Right margin for axis line
    RUGPLOT_Y_MIN = -0.5        # Bottom of plot area
    RUGPLOT_Y_MAX = 3.0         # Top of plot area
    RUGPLOT_AXIS_Y = 1.0        # Y position of the horizontal axis line
    RUGPLOT_LABEL_Y = 0.7       # Y position for min/max/name labels
    RUGPLOT_HIGHLIGHT_LABEL_Y = -0.3  # Y position for highlighted value label
    RUGPLOT_PADDING_FACTOR = 0.05     # Padding as fraction of data range

    # -- Rugplot appearance --
    RUGPLOT_TICK_WIDTH = 0.3
    RUGPLOT_HIGHLIGHT_TICK_WIDTH = 1.5
    RUGPLOT_AXIS_LINE_WIDTH = 1.0
    RUGPLOT_TICK_COLOR = 'orange'
    RUGPLOT_HIGHLIGHT_COLOR = 'blue'

    # -- Algorithm visualization colors --
    # Colors for target meshes of different alignment algorithms (numkeys 2-9)
    ALGORITHM_COLORS = {
        'green': [0, 1, 0],
        'cyan': [0, 1, 1],
        'pink': [1, 0, 1],
        'orange': [1, 0.5, 0],
        'red': [1, 0, 0],
        'blue': [0, 0, 1],
        'purple': [0.5, 0, 0.5],
        'yellow': [1, 1, 0],
    }
    # Order in which colors are assigned to algorithms
    ALGORITHM_COLOR_ORDER = ['green', 'cyan', 'pink', 'orange', 'red', 'blue', 'purple', 'yellow']

    # -- Columns excluded from rugplots by default --
    # These are filtered out because they're not useful for visualization
    DEFAULT_EXCLUDED_COLUMNS = [
        "trial_number",
        "datetime_start",
        "state",
        "tm_run_identifier",
        "chosen_mesh_identifier",
        "best_alignment_algorithm_per_iter",
        "best_chamfer_distance_per_iter",
        "chamfer_distances_all",
        "evaluate_loss_duration",
        "tm_run_incomplete",
        #"iter_of_chosen_mesh",
        "params_to_optimize",
        "alignment_algorithms",
        "run_config",
    ]

    # -- Non-algorithm columns used for sorting --
    #NON_ALGORITHM_SORT_COLUMNS = ['id', 'loss', 'tm_run_duration']
    NON_ALGORITHM_SORT_COLUMNS = ['id', 'loss', 'chosen_mesh_duration']

    # -- Thread/timer settings --
    CONSOLE_THREAD_SLEEP_INTERVAL = 0.1  # seconds
    CONSOLE_THREAD_STOP_TIMEOUT = 3000   # milliseconds
    CONSOLE_THREAD_TERMINATE_TIMEOUT = 1000  # milliseconds
