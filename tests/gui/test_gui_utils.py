"""
Tests for bites.gui.gui_utils module.

Covers:
- load_yaml_config: YAML file loading
- set_decimal_places: adaptive decimal formatting (pure function)
- dict_from_txt_file: text-file-to-dict parser
- ErrorNotifier: logging and dialog helpers (Qt mocked)
- open_users_guide: PDF opening via QDesktopServices (mocked)

No real Qt event loop or file dialogs are created.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bites.gui.gui_utils import (
    load_yaml_config,
    set_decimal_places,
    dict_from_txt_file,
    _parse_humppa_format,
    _parse_toothmaker_format,
    ErrorNotifier,
    open_users_guide,
)


# ============================================================================
# TestLoadYamlConfig
# ============================================================================

class TestLoadYamlConfig:
    """Tests for load_yaml_config."""

    def test_loads_valid_yaml(self, sample_config_path):
        """Load the test data sample config and verify it returns a dict."""
        result = load_yaml_config(sample_config_path)
        assert isinstance(result, dict)
        assert "study_name" in result

    def test_loads_simple_yaml(self, tmp_path):
        """Load a trivially small YAML file."""
        f = tmp_path / "simple.yaml"
        f.write_text("key: value\nnumber: 42\n")
        result = load_yaml_config(f)
        assert result == {"key": "value", "number": 42}

    def test_accepts_string_path(self, tmp_path):
        """String path should work as well as Path."""
        f = tmp_path / "test.yaml"
        f.write_text("a: 1\n")
        result = load_yaml_config(str(f))
        assert result == {"a": 1}

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_yaml_config(tmp_path / "nope.yaml")

    def test_empty_file_returns_none(self, tmp_path):
        """An empty YAML file returns None from yaml.safe_load."""
        f = tmp_path / "empty.yaml"
        f.write_text("")
        result = load_yaml_config(f)
        assert result is None


# ============================================================================
# TestSetDecimalPlaces
# ============================================================================

class TestSetDecimalPlaces:
    """Tests for the set_decimal_places pure function.

    Verifies adaptive formatting based on value magnitude.
    """

    def test_zero(self):
        assert set_decimal_places(0) == "0"

    @pytest.mark.parametrize("value, expected_decimals", [
        (0.00000001, 9),   # < 0.0000001
        (0.0000005, 8),    # < 0.000001
        (0.000005, 7),     # < 0.00001
        (0.00005, 6),      # < 0.0001
        (0.0005, 5),       # < 0.001
        (0.005, 4),        # < 0.01
        (0.05, 3),         # < 0.1
        (0.5, 2),          # < 1
    ])
    def test_small_values_precision(self, value, expected_decimals):
        """Smaller values should use more decimal places."""
        result = set_decimal_places(value)
        # The result should have the right number of decimal places
        # (excluding trailing zeros which Python float formatting keeps)
        parts = result.split(".")
        assert len(parts) == 2
        assert len(parts[1]) == expected_decimals

    def test_value_between_1_and_10_integer_like(self):
        """Values near an integer (1-10) should show zero decimal places."""
        assert set_decimal_places(5.0) == "5"
        assert set_decimal_places(1.0) == "1"

    def test_value_between_1_and_10_with_fraction(self):
        """Values between 1 and 10 with significant fraction get 2 decimals."""
        result = set_decimal_places(3.14)
        assert result == "3.14"

    def test_value_ge_10(self):
        """Values >= 10 should show no decimal places."""
        assert set_decimal_places(10) == "10"
        assert set_decimal_places(42.7) == "43"
        assert set_decimal_places(100) == "100"

    def test_boundary_exactly_0_0000001(self):
        """Value exactly at 0.0000001 boundary should use 8 decimal places (< 0.000001 branch)."""
        result = set_decimal_places(0.0000001)
        parts = result.split(".")
        assert len(parts[1]) == 8

    def test_boundary_exactly_1(self):
        """Value exactly at 1 should be in the 1-10 branch."""
        assert set_decimal_places(1.0) == "1"

    def test_returns_string(self):
        """All results should be strings."""
        assert isinstance(set_decimal_places(0), str)
        assert isinstance(set_decimal_places(3.14), str)
        assert isinstance(set_decimal_places(100), str)


# ============================================================================
# TestDictFromTxtFile
# ============================================================================

class TestDictFromTxtFile:
    """Tests for dict_from_txt_file text parser."""

    def test_basic_parsing(self, tmp_path):
        """Standard format: value then name on each line."""
        f = tmp_path / "params.txt"
        f.write_text("0.025 Egr\n200.0 Mgr\n1.0 Rep\n")
        result = dict_from_txt_file(f)
        assert result == {"Egr": 0.025, "Mgr": 200.0, "Rep": 1.0}

    def test_accepts_string_path(self, tmp_path):
        f = tmp_path / "params.txt"
        f.write_text("1.0 Val\n")
        result = dict_from_txt_file(str(f))
        assert result == {"Val": 1.0}

    def test_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "params.txt"
        f.write_text("1.0 A\n\n\n2.0 B\n")
        result = dict_from_txt_file(f)
        assert result == {"A": 1.0, "B": 2.0}

    def test_duplicate_keys_get_suffix(self, tmp_path):
        """Duplicate parameter names should be renamed with _2, _3, etc."""
        f = tmp_path / "params.txt"
        f.write_text("1.0 Dup\n2.0 Dup\n3.0 Dup\n")
        result = dict_from_txt_file(f)
        assert result == {"Dup": 1.0, "Dup_2": 2.0, "Dup_3": 3.0}

    def test_numeric_last_part_not_added_as_name(self, tmp_path):
        """If the last part is numeric, it should not be added as a name.

        values and strings are collected separately, then zipped by position.
        Line "1.0 2.0" adds 1.0 to values but nothing to strings (2.0 is numeric).
        Line "3.0 Name" adds 3.0 to values and "Name" to strings.
        zip pairs: ("Name", 1.0) — the first available value with the first name.
        """
        f = tmp_path / "params.txt"
        f.write_text("1.0 2.0\n3.0 Name\n")
        result = dict_from_txt_file(f)
        assert "Name" in result
        assert result["Name"] == 1.0  # paired positionally via zip

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            dict_from_txt_file(tmp_path / "nope.txt")

    def test_multiword_line_uses_last_part(self, tmp_path):
        """Lines with extra whitespace should use the last part as the name."""
        f = tmp_path / "params.txt"
        f.write_text("0.5 some extra info ParamName\n")
        result = dict_from_txt_file(f)
        assert "ParamName" in result
        assert result["ParamName"] == 0.5

    def test_detects_toothmaker_format(self, tmp_path):
        """Files with '==' lines should be parsed as ToothMaker format."""
        f = tmp_path / "params.txt"
        f.write_text(
            "# comment\n"
            "model==Tribosphenic tooth\n"
            "iter==9000\n"
            "Egr==0.0130000\n"
            "Mgr==16000.0000000\n"
        )
        result = dict_from_txt_file(f)
        assert result == {"Egr": 0.013, "Mgr": 16000.0}

    def test_detects_humppa_format(self, tmp_path):
        """Files without '==' should be parsed as humppa format."""
        f = tmp_path / "params.txt"
        f.write_text(
            "0.025 Egr\n"
            "200.0 Mgr\n"
            "0.0 -\n"
            "25 number of cells\n"
            "1000 number of iterations\n"
        )
        result = dict_from_txt_file(f)
        assert "Egr" in result
        assert "Mgr" in result
        assert "-" not in result
        assert "number" not in result

    def test_toothmaker_without_umgr(self, tmp_path):
        """ToothMaker format without uMgr should still work."""
        f = tmp_path / "params.txt"
        f.write_text("Egr==0.013\nMgr==200.0\n")
        result = dict_from_txt_file(f)
        assert "uMgr" not in result
        assert result == {"Egr": 0.013, "Mgr": 200.0}

    def test_humppa_without_umgr(self, tmp_path):
        """Humppa format without uMgr should still work."""
        f = tmp_path / "params.txt"
        f.write_text("0.013 Egr\n200.0 Mgr\n")
        result = dict_from_txt_file(f)
        assert "uMgr" not in result
        assert result == {"Egr": 0.013, "Mgr": 200.0}

    def test_toothmaker_with_umgr(self, tmp_path):
        """ToothMaker format with uMgr should include it."""
        f = tmp_path / "params.txt"
        f.write_text("Egr==0.013\nuMgr==0.5\n")
        result = dict_from_txt_file(f)
        assert result == {"Egr": 0.013, "uMgr": 0.5}

    def test_humppa_with_umgr(self, tmp_path):
        """Humppa format with uMgr should include it."""
        f = tmp_path / "params.txt"
        f.write_text("0.013 Egr\n0.5 uMgr\n")
        result = dict_from_txt_file(f)
        assert result == {"Egr": 0.013, "uMgr": 0.5}


# ============================================================================
# TestParseHumppaFormat
# ============================================================================

class TestParseHumppaFormat:
    """Tests for _parse_humppa_format helper."""

    def test_standard_parameters(self):
        lines = ["0.025 Egr\n", "200.0 Mgr\n", "1.0 Rep\n"]
        result = _parse_humppa_format(lines)
        assert result == {"Egr": 0.025, "Mgr": 200.0, "Rep": 1.0}

    def test_skips_dash_placeholders(self):
        lines = ["0.0 -\n", "37.0 -\n", "0.025 Egr\n"]
        result = _parse_humppa_format(lines)
        assert "-" not in result
        assert result == {"Egr": 0.025}

    def test_skips_number_of_cells_and_iterations(self):
        lines = ["25 number of cells\n", "1000 number of iterations\n"]
        result = _parse_humppa_format(lines)
        assert len(result) == 0

    def test_empty_lines_skipped(self):
        lines = ["0.025 Egr\n", "\n", "200.0 Mgr\n"]
        result = _parse_humppa_format(lines)
        assert result == {"Egr": 0.025, "Mgr": 200.0}

    def test_duplicate_keys(self):
        lines = ["1.0 Dup\n", "2.0 Dup\n"]
        result = _parse_humppa_format(lines)
        assert result == {"Dup": 1.0, "Dup_2": 2.0}

    def test_scientific_notation(self):
        lines = ["4.7528598457574844E-002 Egr\n", "1.0000000474974513E-003 Adh\n"]
        result = _parse_humppa_format(lines)
        assert abs(result["Egr"] - 0.047528598457574844) < 1e-15
        assert abs(result["Adh"] - 0.001) < 1e-10


# ============================================================================
# TestParseToothmakerFormat
# ============================================================================

class TestParseToothmakerFormat:
    """Tests for _parse_toothmaker_format helper."""

    def test_standard_parameters(self):
        lines = ["Egr==0.0130000\n", "Mgr==16000.0000000\n"]
        result = _parse_toothmaker_format(lines)
        assert result == {"Egr": 0.013, "Mgr": 16000.0}

    def test_skips_comments(self):
        lines = ["# This is a comment\n", "Egr==0.013\n"]
        result = _parse_toothmaker_format(lines)
        assert result == {"Egr": 0.013}

    def test_skips_metadata_keys(self):
        lines = [
            "model==Tribosphenic tooth\n",
            "viewthresh==0.080000\n",
            "viewmode==4\n",
            "iter==9000\n",
            "Egr==0.013\n",
        ]
        result = _parse_toothmaker_format(lines)
        assert "model" not in result
        assert "viewthresh" not in result
        assert "viewmode" not in result
        assert "iter" not in result
        assert result == {"Egr": 0.013}

    def test_empty_lines_skipped(self):
        lines = ["Egr==0.013\n", "\n", "Mgr==200.0\n"]
        result = _parse_toothmaker_format(lines)
        assert result == {"Egr": 0.013, "Mgr": 200.0}

    def test_duplicate_keys(self):
        lines = ["Dup==1.0\n", "Dup==2.0\n"]
        result = _parse_toothmaker_format(lines)
        assert result == {"Dup": 1.0, "Dup_2": 2.0}

    def test_case_sensitive_param_names(self):
        """Parameter names are case-sensitive (uMgr != umgr)."""
        lines = ["uMgr==0.5\n"]
        result = _parse_toothmaker_format(lines)
        assert "uMgr" in result
        assert result["uMgr"] == 0.5

    def test_lines_without_equals_skipped(self):
        lines = ["some random text\n", "Egr==0.013\n"]
        result = _parse_toothmaker_format(lines)
        assert result == {"Egr": 0.013}


# ============================================================================
# TestErrorNotifier
# ============================================================================

class TestErrorNotifier:
    """Tests for ErrorNotifier static methods (Qt dialogs are mocked)."""

    @patch("bites.gui.gui_utils.ErrorNotifier._show_dialog")
    def test_show_error_logs_message(self, mock_dialog, caplog):
        """show_error should log at ERROR level and call _show_dialog."""
        with caplog.at_level(logging.ERROR, logger="bites.gui.gui_utils"):
            ErrorNotifier.show_error(None, "Title", "Something failed")
        assert "Something failed" in caplog.text
        mock_dialog.assert_called_once()

    @patch("bites.gui.gui_utils.ErrorNotifier._show_dialog")
    def test_show_error_includes_details_in_log(self, mock_dialog, caplog):
        with caplog.at_level(logging.ERROR, logger="bites.gui.gui_utils"):
            ErrorNotifier.show_error(None, "T", "msg", details="traceback here")
        assert "traceback here" in caplog.text

    @patch("bites.gui.gui_utils.ErrorNotifier._show_dialog")
    def test_show_warning_logs_at_warning_level(self, mock_dialog, caplog):
        with caplog.at_level(logging.WARNING, logger="bites.gui.gui_utils"):
            ErrorNotifier.show_warning(None, "Warn", "Watch out")
        assert "Watch out" in caplog.text

    @patch("bites.gui.gui_utils.ErrorNotifier._show_dialog")
    def test_show_info_logs_at_info_level(self, mock_dialog, caplog):
        with caplog.at_level(logging.INFO, logger="bites.gui.gui_utils"):
            ErrorNotifier.show_info(None, "Info", "All good")
        assert "All good" in caplog.text

    def test_log_exception_returns_traceback_string(self):
        try:
            raise ValueError("test error")
        except ValueError as e:
            details = ErrorNotifier.log_exception("context", e)
        assert isinstance(details, str)
        assert "ValueError" in details

    @patch("bites.gui.gui_utils.QMessageBox")
    def test_show_dialog_with_none_parent(self, mock_qmb_cls):
        """_show_dialog should handle None parent gracefully."""
        from PySide6.QtWidgets import QMessageBox
        ErrorNotifier._show_dialog(None, "T", "msg", None, QMessageBox.Critical)
        mock_qmb_cls.assert_called_once_with(None)

    @patch("bites.gui.gui_utils.QMessageBox")
    def test_show_dialog_with_destroyed_parent(self, mock_qmb_cls):
        """A parent whose isVisible() raises RuntimeError should be treated as None."""
        from PySide6.QtWidgets import QMessageBox
        parent = MagicMock()
        parent._is_closed = False
        parent.isVisible.side_effect = RuntimeError("C++ deleted")
        ErrorNotifier._show_dialog(parent, "T", "msg", None, QMessageBox.Warning)
        # Should fall back to None parent
        mock_qmb_cls.assert_called_once_with(None)

    @patch("bites.gui.gui_utils.QMessageBox")
    def test_show_dialog_sets_detailed_text(self, mock_qmb_cls):
        from PySide6.QtWidgets import QMessageBox
        ErrorNotifier._show_dialog(None, "T", "msg", "extra detail", QMessageBox.Critical)
        instance = mock_qmb_cls.return_value
        instance.setDetailedText.assert_called_once_with("extra detail")

    @patch("bites.gui.gui_utils.QMessageBox")
    def test_show_dialog_no_details_skips_detailed_text(self, mock_qmb_cls):
        from PySide6.QtWidgets import QMessageBox
        ErrorNotifier._show_dialog(None, "T", "msg", None, QMessageBox.Information)
        instance = mock_qmb_cls.return_value
        instance.setDetailedText.assert_not_called()


# ============================================================================
# TestOpenUsersGuide
# ============================================================================

class TestOpenUsersGuide:
    """Tests for open_users_guide (file system and Qt calls mocked)."""

    @patch("bites.gui.gui_utils.QDesktopServices")
    @patch("bites.gui.gui_utils.get_asset")
    def test_opens_existing_pdf(self, mock_get_asset, mock_desktop):
        """Should return True when PDF exists and openUrl succeeds."""
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/users_guide.pdf"
        mock_get_asset.return_value = fake_path
        mock_desktop.openUrl.return_value = True

        result = open_users_guide()
        assert result is True

    @patch("bites.gui.gui_utils.QDesktopServices")
    @patch("bites.gui.gui_utils.get_asset")
    def test_returns_false_when_pdf_missing(self, mock_get_asset, mock_desktop):
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = False
        mock_get_asset.return_value = fake_path

        result = open_users_guide()
        assert result is False

    @patch("bites.gui.gui_utils.QDesktopServices")
    @patch("bites.gui.gui_utils.get_asset")
    def test_returns_false_when_openurl_fails(self, mock_get_asset, mock_desktop):
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/users_guide.pdf"
        mock_get_asset.return_value = fake_path
        mock_desktop.openUrl.return_value = False

        result = open_users_guide()
        assert result is False

    @patch("bites.gui.gui_utils.QUrl")
    @patch("bites.gui.gui_utils.QDesktopServices")
    @patch("bites.gui.gui_utils.get_asset")
    def test_page_fragment_set_when_page_provided(self, mock_get_asset, mock_desktop, mock_qurl):
        """When page is given, URL should have a #page=N fragment."""
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/users_guide.pdf"
        mock_get_asset.return_value = fake_path
        mock_desktop.openUrl.return_value = True

        url_instance = MagicMock()
        mock_qurl.fromLocalFile.return_value = url_instance

        open_users_guide(page=5)
        url_instance.setFragment.assert_called_once_with("page=5")
