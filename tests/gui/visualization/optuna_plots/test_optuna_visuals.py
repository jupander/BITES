"""
Tests for bites.gui.visualization.optuna_plots.visuals module.

Covers:
- find_storage_url_from_config: config file discovery and URL inference
- save_visualizations: HTML file generation (optuna/plotly mocked)

All optuna, plotly, and database interactions are mocked.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from bites.gui.visualization.optuna_plots.visuals import (
    find_storage_url_from_config,
    save_visualizations,
)


# ============================================================================
# TestFindStorageUrlFromConfig
# ============================================================================

class TestFindStorageUrlFromConfig:
    """Tests for find_storage_url_from_config."""

    def test_no_config_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No config file"):
            find_storage_url_from_config(tmp_path)

    @patch("bites.gui.visualization.optuna_plots.visuals.ensure_database_for_target")
    @patch("bites.gui.visualization.optuna_plots.visuals.Config")
    def test_shared_mode_calls_ensure_database(self, mock_config_cls, mock_ensure, tmp_path):
        """Shared storage mode should call ensure_database_for_target."""
        # Create a config file
        config_file = tmp_path / "config_2024-01-01_00-00-00.yaml"
        config_file.write_text("dummy")

        mock_config = MagicMock()
        mock_config.storage_mode = "shared"
        mock_config_cls.from_yaml.return_value = mock_config
        mock_ensure.return_value = "postgresql://localhost/test"

        result = find_storage_url_from_config(tmp_path)
        assert result == "postgresql://localhost/test"
        mock_ensure.assert_called_once_with(mock_config)

    @patch("bites.gui.visualization.optuna_plots.visuals.Config")
    def test_non_shared_mode_returns_none(self, mock_config_cls, tmp_path):
        """Non-shared mode (no else branch implemented) returns None."""
        config_file = tmp_path / "config_2024-01-01_00-00-00.yaml"
        config_file.write_text("dummy")

        mock_config = MagicMock()
        mock_config.storage_mode = "local"
        mock_config_cls.from_yaml.return_value = mock_config

        result = find_storage_url_from_config(tmp_path)
        assert result is None  # no else branch

    @patch("bites.gui.visualization.optuna_plots.visuals.ensure_database_for_target")
    @patch("bites.gui.visualization.optuna_plots.visuals.Config")
    def test_picks_newest_config(self, mock_config_cls, mock_ensure, tmp_path):
        """Should use the last config file when sorted by name."""
        (tmp_path / "config_2024-01-01_00-00-00.yaml").write_text("old")
        (tmp_path / "config_2024-06-01_00-00-00.yaml").write_text("new")

        mock_config = MagicMock()
        mock_config.storage_mode = "shared"
        mock_config_cls.from_yaml.return_value = mock_config
        mock_ensure.return_value = "url"

        find_storage_url_from_config(tmp_path)
        # The newest config (sorted last) should be used
        called_path = mock_config_cls.from_yaml.call_args[0][0]
        assert "2024-06-01" in called_path


# ============================================================================
# TestSaveVisualizations
# ============================================================================

class TestSaveVisualizations:
    """Tests for save_visualizations with mocked optuna study."""

    @patch("bites.gui.visualization.optuna_plots.visuals.vis")
    def test_creates_output_directory(self, mock_vis, tmp_path):
        mock_study = MagicMock()
        mock_fig = MagicMock()
        # Every plot function returns a mock figure
        mock_vis.plot_optimization_history.return_value = mock_fig
        mock_vis.plot_slice.return_value = mock_fig
        mock_vis.plot_parallel_coordinate.return_value = mock_fig
        mock_vis.plot_param_importances.return_value = mock_fig
        mock_vis.plot_contour.return_value = mock_fig

        save_visualizations(mock_study, tmp_path)
        assert (tmp_path / "optuna_visualizations").is_dir()

    @patch("bites.gui.visualization.optuna_plots.visuals.vis")
    def test_writes_five_html_files(self, mock_vis, tmp_path):
        mock_study = MagicMock()
        mock_fig = MagicMock()
        mock_vis.plot_optimization_history.return_value = mock_fig
        mock_vis.plot_slice.return_value = mock_fig
        mock_vis.plot_parallel_coordinate.return_value = mock_fig
        mock_vis.plot_param_importances.return_value = mock_fig
        mock_vis.plot_contour.return_value = mock_fig

        save_visualizations(mock_study, tmp_path)
        assert mock_fig.write_html.call_count == 5

    @patch("bites.gui.visualization.optuna_plots.visuals.vis")
    def test_continues_on_plot_failure(self, mock_vis, tmp_path):
        """If one plot fails, the others should still be generated."""
        mock_study = MagicMock()
        mock_fig = MagicMock()

        # First plot raises, rest succeed
        mock_vis.plot_optimization_history.side_effect = Exception("boom")
        mock_vis.plot_slice.return_value = mock_fig
        mock_vis.plot_parallel_coordinate.return_value = mock_fig
        mock_vis.plot_param_importances.return_value = mock_fig
        mock_vis.plot_contour.return_value = mock_fig

        save_visualizations(mock_study, tmp_path)
        assert mock_fig.write_html.call_count == 4  # 5 - 1 failed

    @patch("bites.gui.visualization.optuna_plots.visuals.vis")
    def test_write_html_called_with_validate_false(self, mock_vis, tmp_path):
        """write_html should be called with validate=False."""
        mock_study = MagicMock()
        mock_fig = MagicMock()
        mock_vis.plot_optimization_history.return_value = mock_fig
        mock_vis.plot_slice.return_value = mock_fig
        mock_vis.plot_parallel_coordinate.return_value = mock_fig
        mock_vis.plot_param_importances.return_value = mock_fig
        mock_vis.plot_contour.return_value = mock_fig

        save_visualizations(mock_study, tmp_path)
        for c in mock_fig.write_html.call_args_list:
            assert c[1].get("validate") is False or c.kwargs.get("validate") is False
