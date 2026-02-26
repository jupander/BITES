"""
Logging Configuration - Centralized logging setup for the application.

This module provides functions to configure logging for the BITES application,
ensuring consistent log formatting and output across all modules.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

from bites.paths import get_logs_folder


def setup_gui_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console_level: Optional[int] = None
) -> None:
    """
    Configure logging for the GUI application.

    This sets up logging with:
    - Console output (INFO level by default)
    - Optional file output (DEBUG level)
    - Consistent formatting across all loggers

    Args:
        log_level: Overall logging level (default: INFO)
        log_file: Optional path to log file
        console_level: Console log level (default: same as log_level)
    """
    # Use provided console level or fall back to log_level
    if console_level is None:
        console_level = log_level

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything, filter in handlers

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {log_file}")

    root_logger.info(f"Logging configured (console={logging.getLevelName(console_level)}, file={log_file is not None})")


def setup_subprocess_logging(
    log_level: int = logging.INFO,
    subprocess_name: str = "subprocess"
) -> None:
    """
    Configure logging for subprocess operations.

    This is a simplified logging setup for subprocesses that typically
    only need console output.

    Args:
        log_level: Logging level (default: INFO)
        subprocess_name: Name for the subprocess logger
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create simple formatter for subprocess
    formatter = logging.Formatter(
        f'[{subprocess_name}] [%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    root_logger.info(f"Subprocess logging configured for '{subprocess_name}'")


def get_default_log_file() -> Path:
    """
    Get the default log file path.

    Returns:
        Path: Default log file path (logs/bites_gui.log relative to project root)
    """
    return get_logs_folder() / "bites_gui.log"


def configure_module_logger(module_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger for a specific module.

    Args:
        module_name: Name of the module (usually __name__)
        level: Logging level for this module

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    return logger


def setup_logger(log_dir: Path, run_name: str) -> Tuple[logging.Logger, Path]:
    """
    Create a named logger that writes to both console and a per-run log file.

    Used by optimization subprocesses that need isolated, run-specific logging.

    Args:
        log_dir: Directory to store log files.
        run_name: Unique name for this run (used in filename and logger name).

    Returns:
        Tuple of (configured Logger, path to the log file).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"log_{run_name}.log"

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, log_file


# Convenience function for quick setup
def setup_default_logging(debug: bool = False) -> None:
    """
    Set up default logging configuration for the application.

    Args:
        debug: If True, set log level to DEBUG
    """
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = get_default_log_file()

    setup_gui_logging(
        log_level=log_level,
        log_file=log_file,
        console_level=log_level
    )
