"""
Tests for bites.core.run_optimization module.

Covers:
- construct_storage_path: PostgreSQL vs SQLite fallback logic
- construct_trial_parameters_dict: parameter sampling and bias linking
- save_trial_callback: incremental CSV writing
- memory_monitor_callback and rounded_params_logging_callback: callback factories

All database and Optuna internals are mocked.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from bites.core.run_optimization import (
    construct_storage_path,
    construct_trial_parameters_dict,
    save_trial_callback,
    memory_monitor_callback,
    rounded_params_logging_callback,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Return a MagicMock config with standard fields."""
    config = MagicMock()
    config.storage_mode = "shared"
    config.logger = logging.getLogger("test_opt")
    config.study_name = "test_study"
    config.max_iterations = 7000
    config.tm_timeout = 300
    config.alignment_algorithms = ["ICP"]
    config.params_to_optimize = ["Egr", "Act"]
    config.params_optimization_ranges = {
        "Egr": [0.01, 0.05],
        "Act": [0.01, 3.0],
    }
    config.params_fixed_values = {
        "-": 0.0, "Egr": 0.025, "Mgr": 200.0, "Rep": 1.0, "Swi": 0.0,
        "Adh": 0.001, "Act": 0.22, "Inh": 1.5, "Sec": 0.03, "Da": 0.2,
        "Di": 0.2, "Ds": 0.2, "Int": 0.19, "Set": 0.95, "Boy": 0.1,
        "Dff": 0.0004, "Bgr": 1.0, "Abi": 12.0, "Pbi": 18.0, "Lbi": 1.0,
        "Bbi": 1.0, "Rad": 2.0, "Deg": 0.1, "Dgr": 10000.0, "Ntr": 1e-05,
        "Bwi": 0.8, "Ina": 0.0, "uMgr": 0.0,
    }
    config.param_names_all = [
        "Egr", "Mgr", "Rep", "Swi", "Adh", "Act", "Inh", "Sec",
        "Da", "Di", "Ds", "Int", "Set", "Boy", "Dff", "Bgr",
        "Abi", "Pbi", "Lbi", "Bbi", "Rad", "Deg", "Dgr", "Ntr",
        "Bwi", "Ina", "uMgr",
    ]
    config.bias_param_config = MagicMock()
    config.bias_param_config.Abi_value_for_Pbi_too = False
    config.bias_param_config.Abi_Pbi_factor = 1.0
    config.bias_param_config.Bbi_value_for_Lbi_too = False
    config.bias_param_config.Bbi_Lbi_factor = 1.0
    return config


@pytest.fixture
def mock_trial():
    """Return a MagicMock Optuna trial."""
    trial = MagicMock()
    trial.suggest_float.return_value = 0.03
    trial.suggest_int.side_effect = lambda name, low, high, step=1: low
    trial.number = 1
    trial.value = 0.5
    trial.params = {"Egr": 0.03, "Act": 1.5}
    return trial


# ============================================================================
# TestConstructStoragePath
# ============================================================================

class TestConstructStoragePath:
    """Tests for construct_storage_path."""

    @patch("bites.core.run_optimization.ensure_database_for_target")
    def test_postgres_success(self, mock_ensure, mock_config):
        """Shared mode with working PostgreSQL returns postgres URL."""
        mock_ensure.return_value = "postgresql://localhost/db"
        url, stype = construct_storage_path(mock_config, Path("/out"), "study")
        assert stype == "postgres"
        assert "postgresql" in url

    @patch("bites.core.run_optimization.ensure_database_for_target")
    def test_postgres_failure_falls_back_to_sqlite(self, mock_ensure, mock_config, tmp_path):
        """When PostgreSQL fails, should fall back to SQLite."""
        mock_ensure.side_effect = RuntimeError("connection refused")
        url, stype = construct_storage_path(mock_config, tmp_path, "test_study")
        assert stype == "sqlite"
        assert "sqlite" in url
        assert "test_study.db" in url

    @patch("bites.core.run_optimization.ensure_database_for_target")
    def test_postgres_failure_no_fallback_returns_none(self, mock_ensure, mock_config):
        """If PostgreSQL fails and no fallback path, return (None, None)."""
        mock_ensure.side_effect = RuntimeError("fail")
        url, stype = construct_storage_path(mock_config, None, None)
        assert url is None
        assert stype is None

    def test_invalid_storage_mode_raises(self, mock_config):
        """Non-shared storage mode should raise ValueError."""
        mock_config.storage_mode = "local"
        with pytest.raises(ValueError, match="Invalid storage mode"):
            construct_storage_path(mock_config)

    @patch("bites.core.run_optimization.ensure_database_for_target")
    def test_postgres_returns_none_raises(self, mock_ensure, mock_config, tmp_path):
        """If ensure_database returns None, should fall through to SQLite."""
        mock_ensure.return_value = None
        url, stype = construct_storage_path(mock_config, tmp_path, "study")
        # Raises RuntimeError("returned None") which triggers SQLite fallback
        assert stype == "sqlite"


# ============================================================================
# TestConstructTrialParametersDict
# ============================================================================

class TestConstructTrialParametersDict:
    """Tests for construct_trial_parameters_dict."""

    def test_returns_dict(self, mock_trial, mock_config):
        result = construct_trial_parameters_dict(mock_trial, mock_config)
        assert isinstance(result, dict)

    def test_contains_all_param_names(self, mock_trial, mock_config):
        """Output should include all param_names_all plus iter_max and tm_timeout."""
        result = construct_trial_parameters_dict(mock_trial, mock_config)
        for name in mock_config.param_names_all:
            assert name in result, f"Missing parameter: {name}"
        assert "iter_max" in result
        assert "tm_timeout" in result

    def test_sampled_params_override_fixed(self, mock_trial, mock_config):
        """Parameters in params_to_optimize should use sampled (not fixed) values."""
        mock_trial.suggest_float.return_value = 0.999
        result = construct_trial_parameters_dict(mock_trial, mock_config)
        # Egr and Act are in params_to_optimize, so they should be sampled
        assert result["Egr"] == 0.999
        assert result["Act"] == 0.999

    def test_fixed_params_used_for_non_optimized(self, mock_trial, mock_config):
        """Parameters NOT in params_to_optimize should use fixed values."""
        result = construct_trial_parameters_dict(mock_trial, mock_config)
        assert result["Mgr"] == 200.0  # from fixed values
        assert result["Rep"] == 1.0

    def test_bias_param_abi_pbi_linking(self, mock_trial, mock_config):
        """When Abi_value_for_Pbi_too is True, Pbi = Abi * factor."""
        mock_config.bias_param_config.Abi_value_for_Pbi_too = True
        mock_config.bias_param_config.Abi_Pbi_factor = 1.5
        mock_config.params_to_optimize = ["Abi"]
        mock_config.params_optimization_ranges = {"Abi": [5.0, 20.0]}
        mock_trial.suggest_float.return_value = 10.0

        result = construct_trial_parameters_dict(mock_trial, mock_config)
        assert result["Pbi"] == pytest.approx(15.0)  # 10.0 * 1.5

    def test_bias_param_bbi_lbi_linking(self, mock_trial, mock_config):
        """When Bbi_value_for_Lbi_too is True, Lbi = Bbi * factor."""
        mock_config.bias_param_config.Bbi_value_for_Lbi_too = True
        mock_config.bias_param_config.Bbi_Lbi_factor = 2.0
        mock_config.params_to_optimize = ["Bbi"]
        mock_config.params_optimization_ranges = {"Bbi": [0.5, 5.0]}
        mock_trial.suggest_float.return_value = 3.0

        result = construct_trial_parameters_dict(mock_trial, mock_config)
        assert result["Lbi"] == pytest.approx(6.0)  # 3.0 * 2.0

    def test_iter_max_added_as_fake_suggest(self, mock_trial, mock_config):
        """iter_max should appear via suggest_int with identical low=high."""
        result = construct_trial_parameters_dict(mock_trial, mock_config)
        assert "iter_max" in result
        # suggest_int was called with (name, low, high, step=1) where low==high==7000
        calls = [c for c in mock_trial.suggest_int.call_args_list if c[0][0] == "iter_max"]
        assert len(calls) == 1
        assert calls[0][0][1] == 7000  # low
        assert calls[0][0][2] == 7000  # high


# ============================================================================
# TestCallbacks
# ============================================================================

class TestMemoryMonitorCallback:
    """Tests for memory_monitor_callback factory."""

    @patch("bites.core.run_optimization.tracemalloc")
    def test_logs_at_step_interval(self, mock_tracemalloc):
        logger = MagicMock()
        callback = memory_monitor_callback(logger, step=10)

        mock_snapshot = MagicMock()
        mock_snapshot.statistics.return_value = []
        mock_tracemalloc.take_snapshot.return_value = mock_snapshot

        # Trial number 0 should trigger (0 % 10 == 0)
        trial = MagicMock()
        trial.number = 0
        callback(MagicMock(), trial)
        logger.info.assert_called()

    def test_skips_non_step_trials(self):
        logger = MagicMock()
        callback = memory_monitor_callback(logger, step=100)

        trial = MagicMock()
        trial.number = 7  # 7 % 100 != 0
        callback(MagicMock(), trial)
        logger.info.assert_not_called()


class TestRoundedParamsLoggingCallback:
    """Tests for rounded_params_logging_callback factory."""

    def test_logs_trial_result(self):
        logger = MagicMock()
        callback = rounded_params_logging_callback(logger, significant_digits=3)

        study = MagicMock()
        study.best_trial.number = 0
        study.best_value = 0.1

        trial = MagicMock()
        trial.number = 5
        trial.value = 0.25
        trial.params = {"Egr": 0.025123, "Act": 1.5}

        callback(study, trial)
        logger.info.assert_called_once()
        assert "Trial 5" in logger.info.call_args[0][0]

    def test_skips_none_value(self):
        logger = MagicMock()
        callback = rounded_params_logging_callback(logger)

        trial = MagicMock()
        trial.value = None
        callback(MagicMock(), trial)
        logger.info.assert_not_called()


class TestSaveTrialCallback:
    """Tests for save_trial_callback factory."""

    @patch("bites.core.run_optimization.trial_export")
    def test_creates_csv_on_first_trial(self, mock_output, tmp_path):
        csv_path = str(tmp_path / "ongoing.csv")
        mock_output.get_fixed_columns.return_value = ["trial_number", "loss", "state"]
        mock_output.extract_trial_data.return_value = {
            "trial_number": 0, "loss": 0.5, "state": "COMPLETE"
        }

        callback = save_trial_callback(csv_path)
        callback(MagicMock(), MagicMock())

        assert Path(csv_path).exists()
        content = Path(csv_path).read_text()
        assert "trial_number" in content

    @patch("bites.core.run_optimization.trial_export")
    def test_appends_to_existing_csv(self, mock_output, tmp_path):
        csv_path = str(tmp_path / "ongoing.csv")
        mock_output.get_fixed_columns.return_value = ["trial_number", "loss"]
        mock_output.extract_trial_data.return_value = {
            "trial_number": 0, "loss": 0.5
        }

        callback = save_trial_callback(csv_path)
        callback(MagicMock(), MagicMock())  # first call — creates
        mock_output.extract_trial_data.return_value = {
            "trial_number": 1, "loss": 0.3
        }
        callback(MagicMock(), MagicMock())  # second call — appends

        content = Path(csv_path).read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows
