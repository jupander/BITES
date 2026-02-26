"""
Tests for bites.utils.subprocess_utils module.

Covers:
- terminate_process_tree: forceful process termination
- graceful_terminate_process: interrupt then force-kill workflow

All subprocess operations are mocked — no real processes are spawned.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from bites.utils.subprocess_utils import (
    terminate_process_tree,
    graceful_terminate_process,
)


# ============================================================================
# TestTerminateProcessTree
# ============================================================================

class TestTerminateProcessTree:
    """Tests for terminate_process_tree."""

    def test_none_process_does_nothing(self):
        """Passing None should return silently."""
        terminate_process_tree(None)  # should not raise

    @patch("bites.utils.subprocess_utils.sys")
    @patch("bites.utils.subprocess_utils.subprocess")
    def test_windows_uses_taskkill(self, mock_subprocess, mock_sys):
        """On Windows, should use taskkill /F /T /PID."""
        mock_sys.platform = "win32"
        process = MagicMock()
        process.pid = 12345

        terminate_process_tree(process)
        mock_subprocess.run.assert_called_once()
        args = mock_subprocess.run.call_args[0][0]
        assert "taskkill" in args
        assert str(12345) in args

    @patch("bites.utils.subprocess_utils.sys")
    @patch("bites.utils.subprocess_utils.subprocess")
    def test_windows_fallback_to_terminate(self, mock_subprocess, mock_sys):
        """If taskkill fails, should fall back to process.terminate()."""
        mock_sys.platform = "win32"
        mock_subprocess.run.side_effect = Exception("taskkill not found")
        process = MagicMock()
        process.pid = 12345

        terminate_process_tree(process)
        process.terminate.assert_called_once()


# ============================================================================
# TestGracefulTerminateProcess
# ============================================================================

class TestGracefulTerminateProcess:
    """Tests for graceful_terminate_process."""

    def test_none_process(self):
        """None process should return (True, None)."""
        graceful, code = graceful_terminate_process(None)
        assert graceful is True
        assert code is None

    def test_already_exited(self):
        """If process already exited, should return (True, returncode)."""
        process = MagicMock()
        process.poll.return_value = 0
        process.returncode = 0

        graceful, code = graceful_terminate_process(process)
        assert graceful is True
        assert code == 0

    @patch("bites.utils.subprocess_utils.sys")
    def test_graceful_exit_on_windows(self, mock_sys):
        """Process that exits after CTRL_BREAK should return (True, code)."""
        mock_sys.platform = "win32"
        process = MagicMock()
        process.poll.return_value = None  # first call: not exited
        process.wait.return_value = 0
        process.pid = 12345

        with patch("bites.utils.subprocess_utils.os") as mock_os:
            graceful, code = graceful_terminate_process(process, timeout=5)
            assert graceful is True
            assert code == 0

    @patch("bites.utils.subprocess_utils.terminate_process_tree")
    @patch("bites.utils.subprocess_utils.sys")
    def test_timeout_forces_kill(self, mock_sys, mock_terminate):
        """If process doesn't exit gracefully, should be force-killed."""
        mock_sys.platform = "win32"
        process = MagicMock()
        process.poll.return_value = None
        process.pid = 12345
        process.wait.side_effect = subprocess.TimeoutExpired("cmd", 30)

        with patch("bites.utils.subprocess_utils.os"):
            graceful, code = graceful_terminate_process(process, timeout=1)
            assert graceful is False
            mock_terminate.assert_called_once()
