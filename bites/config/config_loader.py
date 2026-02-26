import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from bites.constants import (
    DEFAULT_MODEL_EXECUTABLE,
    DEFAULT_TM_SUBFOLDER_NAME,
    DEFAULT_ITER_INTERVAL,
    DEFAULT_RESULTS_BACKUP_FOLDER,
)
from bites.paths import get_bin_folder


@dataclass
class BiasParamConfig:
    Abi_value_for_Pbi_too: bool
    Abi_Pbi_factor: float
    Bbi_value_for_Lbi_too: bool
    Bbi_Lbi_factor: float

@dataclass
class PostgresConfig:
    user: str
    password: str
    host: str
    port: int
    db_name: str

@dataclass
class Config:
    # Target shape:
    target_shape_path: Path
    min_cut_height: int
    # Study:
    study_name: str
    max_trials: int
    study_timeout: float  # in hours now instead of seconds
    # Simulation:
    min_iterations: int
    max_iterations: int
    iter_interval: int
    tm_timeout: int
    # Optimization:
    multivariate: bool
    # Parallelism:
    parallel_trials: int
    storage_mode: str
    # Loss function:
    alignment_algorithms: list[str]
    # Params:
    params_to_optimize: list[str]
    params_optimization_ranges: dict[str, list[float]]
    params_fixed_values: dict[str, float]
    bias_param_config: BiasParamConfig
    # PostgreSQL:
    postgresql_config: Optional[PostgresConfig]
    # Technical settings:
    # These are resolved at load time (empty in YAML -> default value here):
    model_executable_path: Path  # resolved to bin/tooth_model.exe if empty
    tm_subfolder_name: str  # resolved to DEFAULT_TM_SUBFOLDER_NAME if empty
    results_backup_folder: str  # resolved to DEFAULT_RESULTS_BACKUP_FOLDER if empty
    save_original_tooth_model_meshes: bool
    open_results_folder_after_run: bool
    # Path overrides (stored as-is, resolved via paths module when needed):
    results_folder: str  # empty = use default from paths.get_results_folder()
    configs_folder: str  # empty = use default from paths.get_configs_folder()
    assets_folder: str  # empty = use default from paths.get_assets_folder()
    # Other:
    param_names_all: list[str]

    @staticmethod
    def from_yaml(path: Path) -> 'Config':
        """
        Load configuration from a YAML file.

        Technical settings are optional overrides (YAML keys end with '_override').
        Empty strings mean "use default":
        - model_executable_path_override: resolved to actual Path (bin/tooth_model.exe)
        - tm_subfolder_name_override: resolved to DEFAULT_TM_SUBFOLDER_NAME
        - results_backup_folder_override: resolved to DEFAULT_RESULTS_BACKUP_FOLDER
        - results_folder_override: empty = use default from paths module
        - configs_folder_override: empty = use default from paths module
        - assets_folder_override: empty = use default from paths module
        """
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        # Resolve model_executable_path (this one is resolved to actual Path)
        model_executable_path = raw.get('model_executable_path_override')
        if not model_executable_path:
            model_executable_path = get_bin_folder() / DEFAULT_MODEL_EXECUTABLE
        else:
            model_executable_path = Path(model_executable_path)

        # Resolve other technical settings with defaults
        tm_subfolder_name = raw.get('tm_subfolder_name_override') or DEFAULT_TM_SUBFOLDER_NAME
        results_backup_folder = raw.get('results_backup_folder_override') or DEFAULT_RESULTS_BACKUP_FOLDER
        iter_interval = raw.get('iter_interval') or DEFAULT_ITER_INTERVAL

        # Path overrides: store as-is (empty string = use default, handled by paths module)
        results_folder = raw.get('results_folder_override', '')
        configs_folder = raw.get('configs_folder_override', '')
        assets_folder = raw.get('assets_folder_override', '')

        # Respect parallel_search_enabled flag: if False, force parallel_trials to 1
        parallel_trials = raw['parallel_trials']
        if not raw.get('parallel_search_enabled', True):
            parallel_trials = 1

        return Config(
            # Target shape:
            target_shape_path=Path(raw['target_shape_path']),
            min_cut_height=raw['min_cut_height'],
            # Study:
            study_name=raw['study_name'],
            max_trials=raw['max_trials'],
            study_timeout=raw['study_timeout'],
            # Simulation:
            min_iterations=raw['min_iterations'],
            max_iterations=raw['max_iterations'],
            iter_interval=iter_interval,
            tm_timeout=raw['tm_timeout'],
            # Optimization:
            multivariate=raw['multivariate'],
            # Parallelism:
            parallel_trials=parallel_trials,
            storage_mode=raw['storage_mode'],
            # Loss function:
            alignment_algorithms=raw['alignment_algorithms'],
            # Params:
            params_to_optimize=raw['params_to_optimize'],
            params_optimization_ranges=raw['params_optimization_ranges'],
            params_fixed_values=raw['params_fixed_values'],
            bias_param_config=BiasParamConfig(**raw['bias_param_config']),
            # PostgreSQL (with backwards compatibility for old key name):
            postgresql_config=PostgresConfig(**raw['postgresql_config']) if 'postgresql_config' in raw else PostgresConfig(**raw['postgres_config']),
            # Technical settings:
            model_executable_path=model_executable_path,
            tm_subfolder_name=tm_subfolder_name,
            save_original_tooth_model_meshes=raw.get('save_original_tooth_model_meshes', False),
            open_results_folder_after_run=raw.get('open_results_folder_after_run', False),
            results_backup_folder=results_backup_folder,
            results_folder=results_folder,
            configs_folder=configs_folder,
            assets_folder=assets_folder,
            param_names_all=raw['param_names_all'],
        )
