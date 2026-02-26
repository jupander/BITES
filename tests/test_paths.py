"""
Tests for bites.paths module.

Covers:
- find_project_root(): marker-based project root detection
- get_project_root() / reset_project_root(): caching behavior
- set/get folder overrides: configs, assets, results
- resolve_path(): relative/absolute path resolution
- get_asset() / get_config(): shorthand path helpers
- ensure_folder_exists(): directory creation
"""

from pathlib import Path

import pytest

from bites import paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_paths_state():
    """Reset all cached state before and after each test."""
    paths.reset_project_root()
    paths.set_configs_folder(None)
    paths.set_assets_folder(None)
    paths.set_results_folder(None)
    yield
    paths.reset_project_root()
    paths.set_configs_folder(None)
    paths.set_assets_folder(None)
    paths.set_results_folder(None)


# ---------------------------------------------------------------------------
# find_project_root
# ---------------------------------------------------------------------------

class TestFindProjectRoot:
    """Tests for find_project_root()."""

    def test_finds_pyproject_toml(self, tmp_path):
        """Finds directory containing pyproject.toml."""
        (tmp_path / "pyproject.toml").touch()
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)

        result = paths.find_project_root(start_path=sub)
        assert result == tmp_path

    def test_finds_bites_root_marker(self, tmp_path):
        """Finds directory containing .bites-root marker."""
        (tmp_path / ".bites-root").touch()
        sub = tmp_path / "deep" / "dir"
        sub.mkdir(parents=True)

        result = paths.find_project_root(start_path=sub)
        assert result == tmp_path

    def test_finds_setup_py(self, tmp_path):
        """Finds directory containing setup.py."""
        (tmp_path / "setup.py").touch()
        result = paths.find_project_root(start_path=tmp_path)
        assert result == tmp_path

    def test_returns_path_object(self, tmp_path):
        """Always returns a Path object."""
        (tmp_path / "pyproject.toml").touch()
        result = paths.find_project_root(start_path=tmp_path)
        assert isinstance(result, Path)

    def test_frozen_executable(self, monkeypatch, tmp_path):
        """Uses exe directory when sys.frozen is True."""
        import sys
        monkeypatch.setattr(sys, 'frozen', True, raising=False)
        monkeypatch.setattr(sys, 'executable', str(tmp_path / "app.exe"))

        result = paths.find_project_root()
        assert result == tmp_path


# ---------------------------------------------------------------------------
# get_project_root / reset_project_root
# ---------------------------------------------------------------------------

class TestProjectRootCaching:
    """Tests for get_project_root() and reset_project_root() caching."""

    def test_caching(self):
        """Second call returns same result without re-detection."""
        a = paths.get_project_root()
        b = paths.get_project_root()
        assert a == b

    def test_reset_clears_cache(self):
        """After reset, get_project_root re-detects."""
        first = paths.get_project_root()
        paths.reset_project_root()
        second = paths.get_project_root()
        # Both should find the same root (we're in the real project)
        assert first == second


# ---------------------------------------------------------------------------
# Folder overrides
# ---------------------------------------------------------------------------

class TestFolderOverrides:
    """Tests for set/get folder override functions."""

    def test_set_configs_folder(self, tmp_path):
        """set_configs_folder changes get_configs_folder result."""
        paths.set_configs_folder(tmp_path)
        assert paths.get_configs_folder() == tmp_path

    def test_clear_configs_folder(self):
        """Empty string clears override, returns default."""
        paths.set_configs_folder("/some/path")
        paths.set_configs_folder("")
        result = paths.get_configs_folder()
        assert result == paths.get_project_root() / paths.DEFAULT_CONFIGS_FOLDER

    def test_none_clears_configs_folder(self):
        """None clears override."""
        paths.set_configs_folder("/some/path")
        paths.set_configs_folder(None)
        result = paths.get_configs_folder()
        assert result == paths.get_project_root() / paths.DEFAULT_CONFIGS_FOLDER

    def test_set_assets_folder(self, tmp_path):
        """set_assets_folder changes get_assets_folder result."""
        paths.set_assets_folder(tmp_path)
        assert paths.get_assets_folder() == tmp_path

    def test_clear_assets_folder(self):
        """Empty string clears assets override."""
        paths.set_assets_folder("/some/path")
        paths.set_assets_folder("")
        result = paths.get_assets_folder()
        assert result == paths.get_project_root() / paths.DEFAULT_ASSETS_FOLDER

    def test_set_results_folder(self, tmp_path):
        """set_results_folder changes get_results_folder result."""
        paths.set_results_folder(tmp_path)
        assert paths.get_results_folder() == tmp_path

    def test_clear_results_folder(self):
        """Empty string clears results override."""
        paths.set_results_folder("/some/path")
        paths.set_results_folder("")
        result = paths.get_results_folder()
        assert result == paths.get_project_root() / paths.DEFAULT_RESULTS_FOLDER

    def test_default_configs_folder(self):
        """Default configs folder is project_root / 'configs'."""
        result = paths.get_configs_folder()
        assert result.name == paths.DEFAULT_CONFIGS_FOLDER

    def test_default_assets_folder(self):
        """Default assets folder is project_root / 'assets'."""
        result = paths.get_assets_folder()
        assert result.name == paths.DEFAULT_ASSETS_FOLDER


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    """Tests for resolve_path()."""

    def test_relative_path_resolved(self):
        """Relative path is resolved against project root."""
        result = paths.resolve_path("some/file.txt")
        assert result.is_absolute()
        assert result == paths.get_project_root() / "some" / "file.txt"

    def test_absolute_path_unchanged(self, tmp_path):
        """Absolute path is returned as-is."""
        abs_path = tmp_path / "file.txt"
        result = paths.resolve_path(abs_path)
        assert result == abs_path

    def test_string_input(self):
        """Accepts string input."""
        result = paths.resolve_path("file.txt")
        assert isinstance(result, Path)

    def test_path_input(self):
        """Accepts Path input."""
        result = paths.resolve_path(Path("file.txt"))
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# get_asset / get_config
# ---------------------------------------------------------------------------

class TestGetAssetAndConfig:
    """Tests for get_asset() and get_config() shortcuts."""

    def test_get_asset(self):
        """get_asset returns path within assets folder."""
        result = paths.get_asset("bites_icon.png")
        assert result == paths.get_assets_folder() / "bites_icon.png"

    def test_get_config(self):
        """get_config returns path within configs folder."""
        result = paths.get_config("example.yaml")
        assert result == paths.get_configs_folder() / "example.yaml"


# ---------------------------------------------------------------------------
# get_bin_folder / get_logs_folder
# ---------------------------------------------------------------------------

class TestOtherFolders:
    """Tests for bin and logs folder helpers."""

    def test_get_bin_folder(self):
        """get_bin_folder returns project_root / 'bin'."""
        result = paths.get_bin_folder()
        assert result.name == paths.DEFAULT_BIN_FOLDER

    def test_get_logs_folder(self):
        """get_logs_folder returns project_root / 'logs'."""
        result = paths.get_logs_folder()
        assert result.name == paths.DEFAULT_LOGS_FOLDER


# ---------------------------------------------------------------------------
# ensure_folder_exists
# ---------------------------------------------------------------------------

class TestEnsureFolderExists:
    """Tests for ensure_folder_exists()."""

    def test_creates_folder(self, tmp_path):
        """Creates a folder that doesn't exist."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        result = paths.ensure_folder_exists(new_dir)
        assert new_dir.exists()
        assert result == new_dir

    def test_existing_folder_is_fine(self, tmp_path):
        """Does not raise for already-existing folder."""
        result = paths.ensure_folder_exists(tmp_path)
        assert result == tmp_path

    def test_returns_same_path(self, tmp_path):
        """Returns the input path (for chaining)."""
        result = paths.ensure_folder_exists(tmp_path)
        assert result is tmp_path
