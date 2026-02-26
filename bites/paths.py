"""
Centralized path management for BITES application.

Provides:
- Automatic project root detection (via pyproject.toml or marker file)
- Configurable folder paths with sensible defaults
- Path resolution utilities using pathlib.Path throughout

Best practices implemented:
- All paths use pathlib.Path for cross-platform compatibility
- Project root is auto-detected and cached
- Relative paths are resolved against project root
- Override paths can be set from technical settings
"""

import sys
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

# Default folder names (relative to project root)
DEFAULT_CONFIGS_FOLDER = "configs"
DEFAULT_ASSETS_FOLDER = "assets"
DEFAULT_LOGS_FOLDER = "logs"
DEFAULT_RESULTS_FOLDER = "results"
DEFAULT_BIN_FOLDER = "bin"

# Project root detection markers (searched in order)
PROJECT_MARKERS = ("pyproject.toml", ".bites-root", "setup.py")

# Cached project root (computed once)
_project_root: Optional[Path] = None

# Configurable overrides (set by technical settings)
_configs_folder_override: Optional[Path] = None
_assets_folder_override: Optional[Path] = None
_results_folder_override: Optional[Path] = None


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find project root by searching upward for marker files.

    For frozen executables (PyInstaller), uses the directory containing the exe.
    Otherwise, searches parent directories starting from start_path (or cwd) until
    a marker file (pyproject.toml, .bites-root, setup.py) is found.

    Args:
        start_path: Starting directory for search (defaults to cwd)

    Returns:
        Path to project root, or cwd if no marker found
    """
    # If running as a frozen executable (PyInstaller), use the exe's directory
    if getattr(sys, 'frozen', False):
        exe_dir = Path(sys.executable).parent
        logger.debug(f"Running as frozen executable, using exe directory: {exe_dir}")
        return exe_dir

    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    # Search upward through parent directories
    for parent in [current, *current.parents]:
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                logger.debug(f"Project root found at {parent} (marker: {marker})")
                return parent

    # Fallback to cwd if no marker found
    logger.warning(
        f"No project root marker found searching from {start_path}. "
        f"Using current working directory: {Path.cwd()}"
    )
    return Path.cwd()


def get_project_root() -> Path:
    """
    Get cached project root path.

    The project root is detected once and cached for performance.
    Call reset_project_root() to force re-detection.

    Returns:
        Path to the project root directory
    """
    global _project_root
    if _project_root is None:
        _project_root = find_project_root()
    return _project_root


def reset_project_root() -> None:
    """Reset cached project root (forces re-detection on next call)."""
    global _project_root
    _project_root = None


def set_configs_folder(path: Optional[Union[str, Path]]) -> None:
    """
    Set override for configs folder (from technical settings).

    Args:
        path: Override path, or None/empty string to use default
    """
    global _configs_folder_override
    if path and str(path).strip():
        _configs_folder_override = Path(path)
        logger.debug(f"Configs folder override set to: {_configs_folder_override}")
    else:
        _configs_folder_override = None
        logger.debug("Configs folder override cleared (using default)")


def set_assets_folder(path: Optional[Union[str, Path]]) -> None:
    """
    Set override for assets folder (from technical settings).

    Args:
        path: Override path, or None/empty string to use default
    """
    global _assets_folder_override
    if path and str(path).strip():
        _assets_folder_override = Path(path)
        logger.debug(f"Assets folder override set to: {_assets_folder_override}")
    else:
        _assets_folder_override = None
        logger.debug("Assets folder override cleared (using default)")


def set_results_folder(path: Optional[Union[str, Path]]) -> None:
    """
    Set override for results folder (from technical settings).

    Args:
        path: Override path, or None/empty string to use default
    """
    global _results_folder_override
    if path and str(path).strip():
        _results_folder_override = Path(path)
        logger.debug(f"Results folder override set to: {_results_folder_override}")
    else:
        _results_folder_override = None
        logger.debug("Results folder override cleared (using default)")


def get_configs_folder() -> Path:
    """
    Get configs folder path (override or default).

    Returns:
        Path to configs folder (absolute)
    """
    if _configs_folder_override is not None:
        return _configs_folder_override
    return get_project_root() / DEFAULT_CONFIGS_FOLDER


def get_assets_folder() -> Path:
    """
    Get assets folder path (override or default).

    Returns:
        Path to assets folder (absolute)
    """
    if _assets_folder_override is not None:
        return _assets_folder_override
    return get_project_root() / DEFAULT_ASSETS_FOLDER


def get_logs_folder() -> Path:
    """
    Get logs folder path.

    Returns:
        Path to logs folder (absolute)
    """
    return get_project_root() / DEFAULT_LOGS_FOLDER


def get_results_folder() -> Path:
    """
    Get results folder path (override or default).

    Returns:
        Path to results folder (absolute)
    """
    if _results_folder_override is not None:
        return _results_folder_override
    return get_project_root() / DEFAULT_RESULTS_FOLDER


def get_bin_folder() -> Path:
    """
    Get bin folder path (for executables).

    Returns:
        Path to bin folder (absolute)
    """
    return get_project_root() / DEFAULT_BIN_FOLDER


def resolve_path(relative_path: Union[str, Path]) -> Path:
    """
    Resolve a relative path against project root.

    If the path is already absolute, returns it as-is.

    Args:
        relative_path: Path to resolve

    Returns:
        Absolute path
    """
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return get_project_root() / path


def get_asset(filename: str) -> Path:
    """
    Get full path to an asset file.

    Args:
        filename: Name of the asset file (e.g., "bites_icon.png")

    Returns:
        Full path to the asset file
    """
    return get_assets_folder() / filename


def get_config(filename: str) -> Path:
    """
    Get full path to a config file.

    Args:
        filename: Name of the config file (e.g., "example_config.yaml")

    Returns:
        Full path to the config file
    """
    return get_configs_folder() / filename


def ensure_folder_exists(folder: Path) -> Path:
    """
    Ensure a folder exists, creating it if necessary.

    Args:
        folder: Path to the folder

    Returns:
        The same path (for chaining)
    """
    folder.mkdir(parents=True, exist_ok=True)
    return folder
