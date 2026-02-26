"""
Tests for bites.gui.visualization.shape_browser.utils module.

Covers:
- _parse_dict_string: JSON / Python literal dict parsing
- _extract_timestamp_from_filename: regex timestamp extraction
- _load_single_csv: single CSV file loading into DataFrame
- get_dataframe_from_csv: multi-CSV merging and deduplication
- get_alignment_algorithm_names: config YAML algorithm extraction
- get_param_txt_filename_from_mesh_identifier: identifier parsing
- get_target_mesh_path: file path construction for cut/uncut targets
- Exception hierarchy

CSV files use semicolon delimiter and QUOTE_NONNUMERIC quoting.
"""

import csv
import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from bites.gui.visualization.shape_browser.utils import (
    ShapeBrowserUtilsError,
    CSVLoadError,
    ConfigLoadError,
    FileNotFoundInFolderError,
    _parse_dict_string,
    _extract_timestamp_from_filename,
    _load_single_csv,
    get_dataframe_from_csv,
    get_alignment_algorithm_names,
    get_param_txt_filename_from_mesh_identifier,
    get_target_mesh_path,
    validate_folder_for_shape_browser,
)


# ============================================================================
# Helpers
# ============================================================================

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    """Write a semicolon-delimited CSV with QUOTE_NONNUMERIC (matching production format)."""
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter=";",
            quoting=csv.QUOTE_NONNUMERIC, escapechar="\\"
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# TestParseDictString
# ============================================================================

class TestParseDictString:
    """Tests for _parse_dict_string."""

    def test_json_format(self):
        result = _parse_dict_string('{"STatac": 6.0}')
        assert result == {"STatac": 6.0}

    def test_python_literal_format(self):
        result = _parse_dict_string("{'STatac': 6.0}")
        assert result == {"STatac": 6.0}

    def test_empty_string_returns_none(self):
        assert _parse_dict_string("") is None

    def test_non_string_returns_none(self):
        assert _parse_dict_string(123) is None
        assert _parse_dict_string(None) is None

    def test_non_dict_result_returns_none(self):
        """A valid JSON list should return None (not a dict)."""
        assert _parse_dict_string("[1, 2, 3]") is None

    def test_invalid_string_returns_none(self):
        assert _parse_dict_string("not a dict at all") is None

    def test_json_with_multiple_keys(self):
        result = _parse_dict_string('{"ICP": 5.0, "STatac": 6.5}')
        assert result == {"ICP": 5.0, "STatac": 6.5}

    def test_python_literal_with_none_value(self):
        result = _parse_dict_string("{'ICP': None}")
        assert result == {"ICP": None}

    def test_row_id_does_not_affect_result(self):
        """row_id is only for warnings — should not change parsed value."""
        result = _parse_dict_string('{"a": 1}', row_id=42)
        assert result == {"a": 1}


# ============================================================================
# TestExtractTimestampFromFilename
# ============================================================================

class TestExtractTimestampFromFilename:
    """Tests for _extract_timestamp_from_filename."""

    def test_standard_filename(self):
        result = _extract_timestamp_from_filename("results_2024-03-15_10-30-00.csv")
        assert result == datetime.datetime(2024, 3, 15, 10, 30, 0)

    def test_config_filename(self):
        result = _extract_timestamp_from_filename("config_2024-01-01_00-00-00.yaml")
        assert result == datetime.datetime(2024, 1, 1, 0, 0, 0)

    def test_no_timestamp_returns_min(self):
        result = _extract_timestamp_from_filename("random_file.csv")
        assert result == datetime.datetime.min

    def test_invalid_date_values_return_min(self):
        """Month 13 is invalid — should fall back to datetime.min."""
        result = _extract_timestamp_from_filename("data_2024-13-01_00-00-00.csv")
        assert result == datetime.datetime.min

    def test_timestamp_embedded_in_path(self):
        result = _extract_timestamp_from_filename("prefix_2025-06-15_08-45-30_suffix.csv")
        assert result == datetime.datetime(2025, 6, 15, 8, 45, 30)

    def test_empty_string(self):
        assert _extract_timestamp_from_filename("") == datetime.datetime.min


# ============================================================================
# TestLoadSingleCsv
# ============================================================================

class TestLoadSingleCsv:
    """Tests for _load_single_csv."""

    def test_loads_valid_csv(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [
            {"chosen_mesh_identifier": "run__1000", "loss": 0.5},
            {"chosen_mesh_identifier": "run__2000", "loss": 0.3},
        ])
        df = _load_single_csv(csv_path)
        assert df is not None
        assert len(df) == 2

    def test_empty_csv_returns_none(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        result = _load_single_csv(csv_path)
        assert result is None

    def test_header_only_csv_returns_none(self, tmp_path):
        """A CSV with only headers and no data rows should return None (empty df)."""
        csv_path = tmp_path / "headers.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["col1", "col2"])
        result = _load_single_csv(csv_path)
        assert result is None

    def test_nonexistent_file_returns_none(self, tmp_path):
        """Missing file should be caught and return None (not raise)."""
        result = _load_single_csv(tmp_path / "nope.csv")
        assert result is None


# ============================================================================
# TestGetDataframeFromCsv
# ============================================================================

class TestGetDataframeFromCsv:
    """Tests for get_dataframe_from_csv."""

    def test_single_csv(self, tmp_path):
        _write_csv(tmp_path / "results_2024-01-01_00-00-00.csv", [
            {"chosen_mesh_identifier": "run__1000", "loss": 0.5},
        ])
        df = get_dataframe_from_csv(tmp_path)
        assert len(df) == 1

    def test_multiple_csvs_merged(self, tmp_path):
        _write_csv(tmp_path / "results_2024-01-01_00-00-00.csv", [
            {"chosen_mesh_identifier": "run_a__1000", "loss": 0.5},
        ])
        _write_csv(tmp_path / "results_2024-01-02_00-00-00.csv", [
            {"chosen_mesh_identifier": "run_b__2000", "loss": 0.3},
        ])
        df = get_dataframe_from_csv(tmp_path)
        assert len(df) == 2

    def test_duplicates_removed(self, tmp_path):
        """Same chosen_mesh_identifier across CSVs: keep newest."""
        _write_csv(tmp_path / "results_2024-01-01_00-00-00.csv", [
            {"chosen_mesh_identifier": "run__1000", "loss": 0.5},
        ])
        _write_csv(tmp_path / "results_2024-01-02_00-00-00.csv", [
            {"chosen_mesh_identifier": "run__1000", "loss": 0.3},
        ])
        df = get_dataframe_from_csv(tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["loss"] == 0.3  # newest value kept

    def test_ongoing_csv_excluded(self, tmp_path):
        """Files starting with 'ongoing_' should be excluded."""
        _write_csv(tmp_path / "ongoing_results.csv", [
            {"chosen_mesh_identifier": "skip", "loss": 0.0},
        ])
        _write_csv(tmp_path / "results_2024-01-01_00-00-00.csv", [
            {"chosen_mesh_identifier": "keep", "loss": 0.5},
        ])
        df = get_dataframe_from_csv(tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["chosen_mesh_identifier"] == "keep"

    def test_nonexistent_folder_raises(self, tmp_path):
        with pytest.raises(CSVLoadError, match="does not exist"):
            get_dataframe_from_csv(tmp_path / "nope")

    def test_not_a_directory_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(CSVLoadError, match="not a directory"):
            get_dataframe_from_csv(f)

    def test_no_csv_files_raises(self, tmp_path):
        with pytest.raises(CSVLoadError, match="No CSV file found"):
            get_dataframe_from_csv(tmp_path)

    def test_accepts_string_path(self, tmp_path):
        _write_csv(tmp_path / "results_2024-01-01_00-00-00.csv", [
            {"chosen_mesh_identifier": "run__1000", "loss": 0.5},
        ])
        df = get_dataframe_from_csv(str(tmp_path))
        assert len(df) == 1


# ============================================================================
# TestGetAlignmentAlgorithmNames
# ============================================================================

class TestGetAlignmentAlgorithmNames:
    """Tests for get_alignment_algorithm_names."""

    def test_finds_algorithms_in_yaml(self, tmp_path):
        config = tmp_path / "config_2024-01-01_00-00-00.yaml"
        config.write_text("alignment_algorithms:\n  - ICP\n  - STatac-CPD\n")
        result = get_alignment_algorithm_names(tmp_path)
        assert result == ["ICP", "STatac-CPD"]

    def test_legacy_config_format(self, tmp_path):
        config = tmp_path / "config_2024-01-01_00-00-00.yaml"
        config.write_text(
            "optimization_config:\n"
            "  alignment_algorithms:\n"
            "    - ICP\n"
        )
        result = get_alignment_algorithm_names(tmp_path)
        assert result == ["ICP"]

    def test_nonexistent_folder_raises(self, tmp_path):
        with pytest.raises(ConfigLoadError, match="does not exist"):
            get_alignment_algorithm_names(tmp_path / "nope")

    def test_no_yaml_files_raises(self, tmp_path):
        with pytest.raises(ConfigLoadError, match="No config file"):
            get_alignment_algorithm_names(tmp_path)

    def test_missing_algorithms_key_raises(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("study_name: test\n")
        with pytest.raises(ConfigLoadError, match="No alignment algorithms"):
            get_alignment_algorithm_names(tmp_path)

    def test_picks_newest_yaml(self, tmp_path):
        """When multiple YAML files exist, should use the one with the newest timestamp."""
        old = tmp_path / "config_2024-01-01_00-00-00.yaml"
        old.write_text("alignment_algorithms:\n  - OLD\n")
        new = tmp_path / "config_2024-06-01_00-00-00.yaml"
        new.write_text("alignment_algorithms:\n  - NEW\n")
        result = get_alignment_algorithm_names(tmp_path)
        assert result == ["NEW"]


# ============================================================================
# TestGetParamTxtFilename
# ============================================================================

class TestGetParamTxtFilename:
    """Tests for get_param_txt_filename_from_mesh_identifier."""

    def test_standard_identifier(self):
        result = get_param_txt_filename_from_mesh_identifier("run240823-145700__7000")
        assert result == "1000___run240823-145700_"

    def test_missing_double_underscore_raises(self):
        with pytest.raises(ValueError, match="Invalid mesh identifier"):
            get_param_txt_filename_from_mesh_identifier("no_separator")

    def test_multiple_double_underscores(self):
        """Should use the first part before the first '__'."""
        result = get_param_txt_filename_from_mesh_identifier("abc__123__456")
        assert result == "1000___abc_"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            get_param_txt_filename_from_mesh_identifier("")


# ============================================================================
# TestGetTargetMeshPath
# ============================================================================

class TestGetTargetMeshPath:
    """Tests for get_target_mesh_path with real filesystem fixtures."""

    @pytest.fixture
    def mesh_folder(self, tmp_path):
        """Set up a folder structure with original and cut target meshes."""
        # Original target mesh folder
        orig_dir = tmp_path / "original_target_mesh"
        orig_dir.mkdir()
        orig_file = orig_dir / "target.off"
        orig_file.write_text("OFF\n0 0 0\n")

        # Cut target mesh folder
        cut_dir = tmp_path / "cut_target_meshes"
        cut_dir.mkdir()
        cut_file = cut_dir / "cut_target__height_6.00.off"
        cut_file.write_text("OFF\n0 0 0\n")

        return tmp_path

    def test_non_cutting_algo_returns_original(self, mesh_folder):
        """An algorithm that doesn't cut should get the original mesh."""
        # ICP does not cut the target
        result = get_target_mesh_path(mesh_folder, "ICP", "run__1000")
        assert "original_target_mesh" in result

    def test_cutting_algo_with_no_cut_heights_returns_original(self, mesh_folder):
        """STatac cuts, but if cut_heights is None, fall back to original."""
        result = get_target_mesh_path(
            mesh_folder, "STatac-C-CPD", "run__1000", cut_heights=None
        )
        assert "original_target_mesh" in result

    def test_cutting_algo_missing_from_cut_heights_returns_original(self, mesh_folder):
        """If the algorithm is not in the cut_heights dict, use original."""
        result = get_target_mesh_path(
            mesh_folder, "STatac-C-CPD", "run__1000",
            cut_heights={"other_algo": 6.0}
        )
        assert "original_target_mesh" in result

    @patch("bites.gui.visualization.shape_browser.utils.get_cut_mesh_filename")
    def test_cutting_algo_with_valid_cut_height(self, mock_fn, mesh_folder):
        """When cut height is present and file exists, return cut mesh path."""
        mock_fn.return_value = "cut_target__height_6.00.off"
        result = get_target_mesh_path(
            mesh_folder, "STatac-C-CPD", "run__1000",
            cut_heights={"STatac-C-CPD": 6.0}
        )
        assert "cut_target_meshes" in result

    @patch("bites.gui.visualization.shape_browser.utils.get_cut_mesh_filename")
    def test_cutting_algo_with_missing_cut_file_falls_back(self, mock_fn, mesh_folder):
        """If the specific cut file doesn't exist, fall back to original."""
        mock_fn.return_value = "nonexistent_cut.off"
        result = get_target_mesh_path(
            mesh_folder, "STatac-C-CPD", "run__1000",
            cut_heights={"STatac-C-CPD": 99.0}
        )
        assert "original_target_mesh" in result

    def test_missing_original_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundInFolderError, match="not found"):
            get_target_mesh_path(tmp_path, "ICP", "run__1000")

    def test_empty_original_folder_raises(self, tmp_path):
        (tmp_path / "original_target_mesh").mkdir()
        with pytest.raises(FileNotFoundInFolderError, match="No files"):
            get_target_mesh_path(tmp_path, "ICP", "run__1000")


# ============================================================================
# TestExceptionHierarchy
# ============================================================================

class TestExceptionHierarchy:
    """Verify the exception hierarchy."""

    def test_csv_load_error_is_shape_browser_error(self):
        assert issubclass(CSVLoadError, ShapeBrowserUtilsError)

    def test_config_load_error_is_shape_browser_error(self):
        assert issubclass(ConfigLoadError, ShapeBrowserUtilsError)

    def test_file_not_found_is_shape_browser_error(self):
        assert issubclass(FileNotFoundInFolderError, ShapeBrowserUtilsError)

    def test_base_is_exception(self):
        assert issubclass(ShapeBrowserUtilsError, Exception)


# ============================================================================
# validate_folder_for_shape_browser
# ============================================================================

class TestValidateFolderForShapeBrowser:
    """Tests for the pre-launch folder validation."""

    def test_nonexistent_folder(self, tmp_path):
        problems = validate_folder_for_shape_browser(tmp_path / "nope")
        assert len(problems) == 1
        assert "does not exist" in problems[0]

    def test_path_is_file_not_dir(self, tmp_path):
        f = tmp_path / "afile.txt"
        f.write_text("hi")
        problems = validate_folder_for_shape_browser(f)
        assert len(problems) == 1
        assert "not a directory" in problems[0]

    def test_empty_folder_reports_all_problems(self, tmp_path):
        problems = validate_folder_for_shape_browser(tmp_path)
        assert any("CSV" in p for p in problems)
        assert any("YAML" in p for p in problems)
        assert any("original_target_mesh" in p for p in problems)
        assert any("aligned_meshes" in p for p in problems)

    def test_valid_folder_returns_no_problems(self, tmp_path):
        # CSV
        (tmp_path / "results_2024-01-01_00-00-00.csv").write_text("data")
        # YAML
        (tmp_path / "config_2024-01-01_00-00-00.yaml").write_text("data")
        # original target mesh
        otm = tmp_path / "original_target_mesh"
        otm.mkdir()
        (otm / "target.ply").write_text("mesh")
        # aligned meshes
        am = tmp_path / "ICP_aligned_meshes"
        am.mkdir()
        (am / "ICP_aligned_run1__1000.ply").write_text("mesh")

        problems = validate_folder_for_shape_browser(tmp_path)
        assert problems == []

    def test_ongoing_csv_excluded(self, tmp_path):
        """ongoing_*.csv backups should not count as results CSVs."""
        (tmp_path / "ongoing_backup.csv").write_text("data")
        problems = validate_folder_for_shape_browser(tmp_path)
        assert any("CSV" in p for p in problems)

    def test_empty_target_mesh_folder(self, tmp_path):
        """original_target_mesh exists but has no mesh files."""
        (tmp_path / "original_target_mesh").mkdir()
        problems = validate_folder_for_shape_browser(tmp_path)
        assert any("No mesh files" in p for p in problems)

    def test_aligned_dirs_without_meshes(self, tmp_path):
        """Aligned mesh folders exist but are empty."""
        (tmp_path / "ICP_aligned_meshes").mkdir()
        problems = validate_folder_for_shape_browser(tmp_path)
        assert any("no mesh files" in p.lower() for p in problems)
