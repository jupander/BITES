"""
Tests for bites.gui.analyze_output_folder module.

Covers:
- analyze_folder: directory scanning, CSV trial counting, visualization listing

All tests use tmp_path to create real folder structures — no mocking needed.
"""

import csv
from pathlib import Path

import pytest

from bites.gui.analyze_output_folder import analyze_folder


# ============================================================================
# Helpers
# ============================================================================

def _write_trial_csv(path: Path, rows: list[dict]):
    """Write a semicolon-delimited CSV with trial data."""
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# TestAnalyzeFolder
# ============================================================================

class TestAnalyzeFolder:
    """Tests for the analyze_folder function."""

    def test_nonexistent_path_returns_empty(self, tmp_path):
        result = analyze_folder(tmp_path / "nope")
        assert result == ""

    def test_empty_folder(self, tmp_path):
        result = analyze_folder(tmp_path)
        assert "Subfolders:" in result
        assert "Trials completed" in result

    def test_subfolders_listed(self, tmp_path):
        sub = tmp_path / "ICP_aligned_meshes"
        sub.mkdir()
        (sub / "mesh1.ply").write_text("dummy")
        (sub / "mesh2.ply").write_text("dummy")

        result = analyze_folder(tmp_path)
        assert "ICP_aligned_meshes" in result
        assert "2 files" in result

    def test_csv_trial_count(self, tmp_path):
        _write_trial_csv(tmp_path / "results.csv", [
            {"state": "COMPLETE", "loss": 0.5},
            {"state": "COMPLETE", "loss": 0.3},
            {"state": "FAIL", "loss": 999},
        ])
        result = analyze_folder(tmp_path)
        assert "2 / 3" in result  # 2 completed out of 3 total

    def test_ongoing_csv_skipped(self, tmp_path):
        """Files with 'ongoing' in the name should be excluded."""
        _write_trial_csv(tmp_path / "ongoing_results.csv", [
            {"state": "COMPLETE", "loss": 0.5},
        ])
        result = analyze_folder(tmp_path)
        assert "ongoing" not in result

    def test_optuna_visualizations_listed(self, tmp_path):
        vis_dir = tmp_path / "optuna_visualizations"
        vis_dir.mkdir()
        (vis_dir / "history.html").write_text("<html>")
        (vis_dir / "slice.html").write_text("<html>")

        result = analyze_folder(tmp_path)
        assert "history.html" in result
        assert "slice.html" in result

    def test_no_visualizations_message(self, tmp_path):
        result = analyze_folder(tmp_path)
        assert "No 'optuna_visualizations'" in result

    def test_csv_missing_state_column(self, tmp_path):
        _write_trial_csv(tmp_path / "results.csv", [
            {"loss": 0.5, "metric": "a"},
        ])
        result = analyze_folder(tmp_path)
        assert "'state' column not found" in result

    def test_accepts_string_path(self, tmp_path):
        result = analyze_folder(str(tmp_path))
        assert isinstance(result, str)
        assert "Subfolders:" in result

    def test_multiple_subfolders_and_csvs(self, tmp_path):
        """Verify all sections appear together."""
        (tmp_path / "sub1").mkdir()
        (tmp_path / "sub2").mkdir()
        _write_trial_csv(tmp_path / "results.csv", [
            {"state": "COMPLETE", "loss": 0.1},
        ])
        result = analyze_folder(tmp_path)
        assert "sub1" in result
        assert "sub2" in result
        assert "1 / 1" in result
