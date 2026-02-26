"""
Tests for bites.config.config_loader module.

Covers:
- BiasParamConfig dataclass construction
- PostgresConfig dataclass construction
- Config dataclass construction and field types
- Config.from_yaml() loading from YAML files
- Default resolution for technical settings
- Backwards-compatible 'postgres_config' key
- parallel_search_enabled flag behavior
"""

from pathlib import Path

import pytest
import yaml

from bites.config.config_loader import BiasParamConfig, Config, PostgresConfig


# ---------------------------------------------------------------------------
# BiasParamConfig
# ---------------------------------------------------------------------------

class TestBiasParamConfig:
    """Tests for BiasParamConfig dataclass."""

    def test_creation(self):
        """Can create BiasParamConfig with valid values."""
        bpc = BiasParamConfig(
            Abi_value_for_Pbi_too=True,
            Abi_Pbi_factor=1.5,
            Bbi_value_for_Lbi_too=False,
            Bbi_Lbi_factor=2.0,
        )
        assert bpc.Abi_value_for_Pbi_too is True
        assert bpc.Abi_Pbi_factor == 1.5
        assert bpc.Bbi_value_for_Lbi_too is False
        assert bpc.Bbi_Lbi_factor == 2.0

    def test_has_four_fields(self):
        """BiasParamConfig has exactly 4 fields."""
        import dataclasses
        fields = dataclasses.fields(BiasParamConfig)
        assert len(fields) == 4


# ---------------------------------------------------------------------------
# PostgresConfig
# ---------------------------------------------------------------------------

class TestPostgresConfig:
    """Tests for PostgresConfig dataclass."""

    def test_creation(self):
        """Can create PostgresConfig with valid values."""
        pg = PostgresConfig(
            user="postgres",
            password="secret",
            host="localhost",
            port=5432,
            db_name="mydb",
        )
        assert pg.user == "postgres"
        assert pg.host == "localhost"
        assert pg.port == 5432

    def test_has_five_fields(self):
        """PostgresConfig has exactly 5 fields."""
        import dataclasses
        fields = dataclasses.fields(PostgresConfig)
        assert len(fields) == 5


# ---------------------------------------------------------------------------
# Config.from_yaml
# ---------------------------------------------------------------------------

class TestConfigFromYaml:
    """Tests for Config.from_yaml() loading."""

    def test_loads_successfully(self, sample_config_path):
        """Config.from_yaml loads the sample config without errors."""
        config = Config.from_yaml(sample_config_path)
        assert isinstance(config, Config)

    def test_study_name(self, sample_config_path):
        """study_name is loaded as a string."""
        config = Config.from_yaml(sample_config_path)
        assert config.study_name == "test_study"

    def test_target_shape_path_is_path(self, sample_config_path):
        """target_shape_path is a Path object."""
        config = Config.from_yaml(sample_config_path)
        assert isinstance(config.target_shape_path, Path)

    def test_model_executable_path_is_path(self, sample_config_path):
        """model_executable_path is a Path object."""
        config = Config.from_yaml(sample_config_path)
        assert isinstance(config.model_executable_path, Path)

    def test_alignment_algorithms_is_list(self, sample_config_path):
        """alignment_algorithms is loaded as a list."""
        config = Config.from_yaml(sample_config_path)
        assert isinstance(config.alignment_algorithms, list)
        assert "STatac-C-CPD" in config.alignment_algorithms

    def test_bias_param_config_type(self, sample_config_path):
        """bias_param_config is a BiasParamConfig instance."""
        config = Config.from_yaml(sample_config_path)
        assert isinstance(config.bias_param_config, BiasParamConfig)

    def test_postgresql_config_type(self, sample_config_path):
        """postgresql_config is a PostgresConfig instance."""
        config = Config.from_yaml(sample_config_path)
        assert isinstance(config.postgresql_config, PostgresConfig)

    def test_numeric_fields(self, sample_config_path):
        """Numeric fields have expected values."""
        config = Config.from_yaml(sample_config_path)
        assert config.max_trials == 100
        assert config.min_iterations == 5000
        assert config.max_iterations == 15000
        assert config.tm_timeout == 300

    def test_boolean_fields(self, sample_config_path):
        """Boolean fields have expected values."""
        config = Config.from_yaml(sample_config_path)
        assert config.multivariate is True
        assert config.save_original_tooth_model_meshes is False
        assert config.open_results_folder_after_run is False

    def test_param_names_all_is_list(self, sample_config_path):
        """param_names_all is a list of strings."""
        config = Config.from_yaml(sample_config_path)
        assert isinstance(config.param_names_all, list)
        assert all(isinstance(n, str) for n in config.param_names_all)
        assert "Egr" in config.param_names_all


# ---------------------------------------------------------------------------
# Default resolution
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """Tests for default value resolution in Config.from_yaml()."""

    def test_default_model_executable_path(self, sample_config_path):
        """Empty model_executable_path_override resolves to bin/tooth_model.exe."""
        config = Config.from_yaml(sample_config_path)
        assert config.model_executable_path.name == "tooth_model.exe"
        assert "bin" in str(config.model_executable_path)

    def test_default_tm_subfolder_name(self, sample_config_path):
        """Empty tm_subfolder_name_override resolves to default."""
        from bites.constants import DEFAULT_TM_SUBFOLDER_NAME
        config = Config.from_yaml(sample_config_path)
        assert config.tm_subfolder_name == DEFAULT_TM_SUBFOLDER_NAME

    def test_default_results_backup_folder(self, sample_config_path):
        """Empty results_backup_folder_override resolves to default."""
        from bites.constants import DEFAULT_RESULTS_BACKUP_FOLDER
        config = Config.from_yaml(sample_config_path)
        assert config.results_backup_folder == DEFAULT_RESULTS_BACKUP_FOLDER

    def test_default_iter_interval(self, sample_config_dict, tmp_path):
        """When iter_interval is 0/missing, uses DEFAULT_ITER_INTERVAL."""
        from bites.constants import DEFAULT_ITER_INTERVAL

        sample_config_dict["iter_interval"] = 0  # falsy -> use default
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = Config.from_yaml(config_path)
        assert config.iter_interval == DEFAULT_ITER_INTERVAL


# ---------------------------------------------------------------------------
# parallel_search_enabled
# ---------------------------------------------------------------------------

class TestParallelSearchEnabled:
    """Tests for parallel_search_enabled flag behavior."""

    def test_disabled_forces_parallel_trials_to_1(self, sample_config_dict, tmp_path):
        """When parallel_search_enabled=False, parallel_trials becomes 1."""
        sample_config_dict["parallel_search_enabled"] = False
        sample_config_dict["parallel_trials"] = 8

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = Config.from_yaml(config_path)
        assert config.parallel_trials == 1

    def test_enabled_preserves_parallel_trials(self, sample_config_dict, tmp_path):
        """When parallel_search_enabled=True, parallel_trials stays as configured."""
        sample_config_dict["parallel_search_enabled"] = True
        sample_config_dict["parallel_trials"] = 8

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = Config.from_yaml(config_path)
        assert config.parallel_trials == 8


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------

class TestBackwardsCompatibility:
    """Tests for backwards-compatible config key names."""

    def test_postgres_config_old_key(self, sample_config_dict, tmp_path):
        """'postgres_config' key works as alias for 'postgresql_config'."""
        # Rename the key to the old name
        sample_config_dict["postgres_config"] = sample_config_dict.pop("postgresql_config")

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = Config.from_yaml(config_path)
        assert isinstance(config.postgresql_config, PostgresConfig)
        assert config.postgresql_config.host == "localhost"
