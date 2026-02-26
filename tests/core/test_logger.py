"""
Tests for bites.core.logging_config module.

Covers:
- setup_logger: creates logger with file + console handlers, log file placement
"""

import logging
from pathlib import Path

import pytest

from bites.core.logging_config import setup_logger


# ============================================================================
# TestSetupLogger
# ============================================================================

class TestSetupLogger:
    """Tests for setup_logger."""

    def test_returns_logger_and_path(self, tmp_path):
        logger, log_file = setup_logger(tmp_path, "test_run")
        assert isinstance(logger, logging.Logger)
        assert isinstance(log_file, Path)

    def test_creates_log_directory(self, tmp_path):
        log_dir = tmp_path / "logs" / "nested"
        setup_logger(log_dir, "test_run")
        assert log_dir.exists()

    def test_creates_log_file(self, tmp_path):
        _, log_file = setup_logger(tmp_path, "test_run")
        assert log_file.exists()
        assert log_file.name == "log_test_run.log"

    def test_logger_has_two_handlers(self, tmp_path):
        """Should have a FileHandler and a StreamHandler."""
        logger, _ = setup_logger(tmp_path, "test_run")
        assert len(logger.handlers) == 2
        handler_types = {type(h) for h in logger.handlers}
        assert logging.FileHandler in handler_types
        assert logging.StreamHandler in handler_types

    def test_logger_level_is_debug(self, tmp_path):
        logger, _ = setup_logger(tmp_path, "test_run")
        assert logger.level == logging.DEBUG

    def test_file_handler_logs_debug(self, tmp_path):
        logger, log_file = setup_logger(tmp_path, "test_run")
        logger.debug("debug message")
        # Flush handlers
        for h in logger.handlers:
            h.flush()
        content = log_file.read_text()
        assert "debug message" in content

    def test_no_propagation(self, tmp_path):
        """Logger should not propagate to avoid duplicate messages."""
        logger, _ = setup_logger(tmp_path, "test_run")
        assert logger.propagate is False

    def test_clears_existing_handlers(self, tmp_path):
        """Calling setup_logger twice should not accumulate handlers."""
        logger1, _ = setup_logger(tmp_path, "same_name")
        logger2, _ = setup_logger(tmp_path, "same_name")
        assert logger1 is logger2  # same name = same logger
        assert len(logger2.handlers) == 2  # not 4

    def test_different_run_names_produce_different_files(self, tmp_path):
        _, log1 = setup_logger(tmp_path, "run_a")
        _, log2 = setup_logger(tmp_path, "run_b")
        assert log1 != log2
        assert log1.name == "log_run_a.log"
        assert log2.name == "log_run_b.log"
