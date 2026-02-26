"""
Default configuration values for creating new config files.

Contains the default config dictionary used when a user creates a new
optimization config file from scratch.
"""

import logging

import psutil

import bites.config.config_io as config_io
from bites.constants import DEFAULT_ITER_INTERVAL

logger = logging.getLogger(__name__)


# Default config file contents.
# All path/folder override fields use empty string to mean "use default".
# This makes configs portable across systems - paths are resolved at runtime.
# Override fields are named with '_override' suffix to clarify their purpose.
DEFAULT_CONFIG = {
    "alignment_algorithms": ["STatac-C-CPD"],
    "assets_folder_override": "",  # empty = use default from paths module
    "bias_param_config": {
        "Abi_Pbi_factor": 1.0,
        "Abi_value_for_Pbi_too": False,
        "Bbi_Lbi_factor": 1.0,
        "Bbi_value_for_Lbi_too": False,
    },
    "configs_folder_override": "",  # empty = use default from paths module
    "iter_interval": DEFAULT_ITER_INTERVAL,
    "max_iterations": 15000,
    "max_trials": 1000,
    "min_cut_height": 12,
    "min_iterations": 5000,
    "model_executable_path_override": "",  # empty = use default (bin/tooth_model.exe)
    "multivariate": True,
    "open_results_folder_after_run": False,
    "parallel_search_enabled": True,
    # Use all but one CPU core, fallback to 1 if cpu_count() fails
    "parallel_trials": max((psutil.cpu_count(logical=True) or 1) - 1, 1),
    "param_names_all": [
        "Egr", "Mgr", "Rep", "Swi", "Adh", "Act", "Inh", "Sec",
        "Da", "Di", "Ds", "Int", "Set", "Boy", "Dff", "Bgr",
        "Abi", "Pbi", "Lbi", "Bbi", "Rad", "Deg", "Dgr", "Ntr",
        "Bwi", "Ina", "uMgr"
    ],
    "params_fixed_values": {
        "-": 0.0,
        "Abi": 12.0,
        "Act": 0.22,
        "Adh": 0.001,
        "Bbi": 1.0,
        "Bgr": 1.0,
        "Boy": 0.1,
        "Bwi": 0.8,
        "Da": 0.2,
        "Deg": 0.1,
        "Dff": 0.0004,
        "Dgr": 10000.0,
        "Di": 0.2,
        "Ds": 0.2,
        "Egr": 0.025,
        "Ina": 0.0,
        "Inh": 1.5,
        "Int": 0.19,
        "Lbi": 1.0,
        "Mgr": 200.0,
        "Ntr": 1.0e-05,
        "Pbi": 18.0,
        "Rad": 2.0,
        "Rep": 1.0,
        "Sec": 0.03,
        "Set": 0.95,
        "Swi": 0.0,
        "iter": 4000.0,
        "uMgr": 0.0,
    },
    "params_optimization_ranges": {
        "Abi": [0.0, 20.0],
        "Act": [0.01, 3.0],
        "Adh": [0.0, 0.02],
        "Bbi": [0.0, 3.0],
        "Bgr": [0.0, 2.0],
        "Boy": [0.0, 1.0],
        "Bwi": [0.0, 5.0],
        "Da": [0.0, 2.0],
        "Deg": [0.0, 0.3],
        "Dff": [0.0, 0.001],
        "Dgr": [0.0, 30000.0],
        "Di": [0.0, 2.0],
        "Ds": [0.0, 2.0],
        "Egr": [0.01, 0.05],
        "Ina": [0.0, 100.0],
        "Inh": [1.0, 1000.0],
        "Int": [0.0, 1.0],
        "Lbi": [0.0, 3.0],
        "Mgr": [0.0, 20000.0],
        "Ntr": [0.0, 0.0002],
        "Pbi": [0.0, 20.0],
        "Rad": [2.0, 4.0],
        "Rep": [0.0, 10.0],
        "Sec": [0.0, 0.3],
        "Set": [0.5, 1.0],
        "Swi": [0.0, 2.0],
        "uMgr": [0.0, 500.0],
    },
    "params_to_optimize": [
        "Egr", "Mgr", "Rep", "Swi", "Adh", "Act", "Inh", "Sec",
        "Da", "Di", "Ds", "Int", "Set", "Boy", "Dff", "Bgr",
        "Abi", "Pbi", "Bbi", "Lbi", "Rad", "Deg", "Dgr", "Ntr", "Bwi",
        "Ina", "uMgr"
    ],
    "postgresql_config": {
        "db_name": "postgres",  # default database name, used to connect to the server
        "host": "localhost",
        "password": "PGPASS_ENABLED",
        "port": 5432,
        "user": "postgres",
    },
    "results_backup_folder_override": "",  # empty = use DEFAULT_RESULTS_BACKUP_FOLDER
    "results_folder_override": "",  # empty = use default "results" folder
    "save_original_tooth_model_meshes": False,
    "storage_mode": "shared",
    "study_name": "example_seal",
    "study_timeout": 1.0,  # 0 = no timeout
    "target_shape_path": "target_shapes/example_ringed_seal.off",
    "technical_settings": True,
    "tm_subfolder_name_override": "",  # empty = use DEFAULT_TM_SUBFOLDER_NAME
    "tm_timeout": 300,
}


def create_config_file_with_default_values(filepath) -> dict | None:
    """
    Create a new config file with default values.

    Args:
        filepath: Path where the config file should be written.

    Returns:
        The default config dictionary if successful, None on failure.
    """
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)

    try:
        config_io.save_yaml_config(filepath, config, template=False)
    except Exception as e:
        logger.error(f"Error creating default config file: {e}")
        return None

    return config
