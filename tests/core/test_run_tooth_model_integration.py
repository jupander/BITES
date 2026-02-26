"""
Tests for bites.core.run_tooth_model subprocess-related functions.

Covers:
- _get_last_iteration_from_output: file scanning for iteration numbers
- _get_newest_mesh_file: file discovery by mtime
- _delete_if_corrupted_mesh: mesh validation and cleanup
- run_tooth_model_as_subprocess: subprocess orchestration with timeout handling

Subprocess execution and mesh reading are mocked; filesystem is real (tmp_path).
"""

import datetime
import logging
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bites.core.run_tooth_model import (
    _get_last_iteration_from_output,
    _get_newest_mesh_file,
    _delete_if_corrupted_mesh,
    run_tooth_model_as_subprocess,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def logger():
    return logging.getLogger("test_rtm")


@pytest.fixture
def output_dir(tmp_path):
    """Return a temporary directory simulating tooth model output."""
    d = tmp_path / "tm_output"
    d.mkdir()
    return d


def _create_mesh_file(output_dir: Path, iter_val: int, run_id: str = "run250101-120000abc"):
    """Create a fake mesh file matching tooth model naming convention."""
    filename = f"{iter_val}___{run_id}_.off"
    f = output_dir / filename
    f.write_text("OFF\n4 4 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2\n3 0 1 3\n3 1 2 3\n3 0 2 3\n")
    return f


# ============================================================================
# TestGetLastIterationFromOutput
# ============================================================================

class TestGetLastIterationFromOutput:
    """Tests for _get_last_iteration_from_output."""

    def test_returns_highest_iteration(self, output_dir):
        run_id = "run250101-120000abc"
        _create_mesh_file(output_dir, 3000, run_id)
        _create_mesh_file(output_dir, 5000, run_id)
        _create_mesh_file(output_dir, 7000, run_id)

        result = _get_last_iteration_from_output(output_dir, run_id)
        assert result == 7000

    def test_no_files_returns_none(self, output_dir):
        result = _get_last_iteration_from_output(output_dir, "run250101-120000abc")
        assert result is None

    def test_ignores_other_run_ids(self, output_dir):
        _create_mesh_file(output_dir, 5000, "run250101-120000abc")
        _create_mesh_file(output_dir, 9000, "run250101-999999xyz")

        result = _get_last_iteration_from_output(output_dir, "run250101-120000abc")
        assert result == 5000

    def test_single_file(self, output_dir):
        run_id = "run250101-120000abc"
        _create_mesh_file(output_dir, 1000, run_id)
        result = _get_last_iteration_from_output(output_dir, run_id)
        assert result == 1000


# ============================================================================
# TestGetNewestMeshFile
# ============================================================================

class TestGetNewestMeshFile:
    """Tests for _get_newest_mesh_file."""

    def test_returns_newest_by_mtime(self, output_dir):
        run_id = "run250101-120000abc"
        f1 = _create_mesh_file(output_dir, 3000, run_id)
        time.sleep(0.05)  # ensure different mtime
        f2 = _create_mesh_file(output_dir, 5000, run_id)

        result = _get_newest_mesh_file(output_dir, run_id)
        assert result is not None
        path, iter_val = result
        assert iter_val == 5000

    def test_no_files_returns_none(self, output_dir):
        result = _get_newest_mesh_file(output_dir, "run250101-120000abc")
        assert result is None

    def test_returns_tuple(self, output_dir):
        run_id = "run250101-120000abc"
        _create_mesh_file(output_dir, 1000, run_id)
        result = _get_newest_mesh_file(output_dir, run_id)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Path)


# ============================================================================
# TestDeleteIfCorruptedMesh
# ============================================================================

class TestDeleteIfCorruptedMesh:
    """Tests for _delete_if_corrupted_mesh."""

    def test_no_files_returns_none(self, output_dir, logger):
        result = _delete_if_corrupted_mesh(output_dir, "run250101-120000abc", logger)
        assert result is None

    @patch("bites.core.run_tooth_model.read_coff_mesh")
    def test_valid_mesh_not_deleted(self, mock_read, output_dir, logger):
        """A valid mesh file should be kept — return None."""
        run_id = "run250101-120000abc"
        f = _create_mesh_file(output_dir, 5000, run_id)
        mock_read.return_value = MagicMock()  # successful read

        result = _delete_if_corrupted_mesh(output_dir, run_id, logger)
        assert result is None
        assert f.exists()  # file still there

    @patch("bites.core.run_tooth_model.read_coff_mesh")
    def test_corrupted_mesh_deleted(self, mock_read, output_dir, logger):
        """A corrupted mesh (read fails) should be deleted."""
        run_id = "run250101-120000abc"
        f = _create_mesh_file(output_dir, 5000, run_id)
        mock_read.side_effect = ValueError("truncated file")

        result = _delete_if_corrupted_mesh(output_dir, run_id, logger)
        assert result == 5000
        assert not f.exists()  # file was deleted

    @patch("bites.core.run_tooth_model.read_coff_mesh")
    def test_corrupted_mesh_oserror(self, mock_read, output_dir, logger):
        """If read raises OSError, mesh should also be deleted."""
        run_id = "run250101-120000abc"
        f = _create_mesh_file(output_dir, 5000, run_id)
        mock_read.side_effect = OSError("bad file")

        result = _delete_if_corrupted_mesh(output_dir, run_id, logger)
        assert result == 5000
        assert not f.exists()


# ============================================================================
# TestRunToothModelAsSubprocess
# ============================================================================

class TestRunToothModelAsSubprocess:
    """Tests for run_tooth_model_as_subprocess with mocked subprocess."""

    @pytest.fixture
    def exe_path(self, tmp_path):
        """Create a fake executable file."""
        exe = tmp_path / "tooth_model.exe"
        exe.write_text("fake")
        return exe

    @pytest.fixture
    def params(self):
        return {
            "Egr": 0.025, "Mgr": 200.0, "Rep": 1.0, "Swi": 0.0,
            "Adh": 0.001, "Act": 0.22, "Inh": 1.5, "-": 0.0,
            "Sec": 0.03, "Da": 0.2, "Di": 0.2, "Ds": 0.2,
            "Int": 0.19, "Set": 0.95, "Boy": 0.1, "Dff": 0.0004,
            "Bgr": 1.0, "Abi": 12.0, "Pbi": 18.0, "Lbi": 1.0,
            "Bbi": 1.0, "Rad": 2.0, "Deg": 0.1, "Dgr": 10000.0,
            "Ntr": 1e-05, "Bwi": 0.8, "Ina": 0.0, "uMgr": 0.0,
            "iter_max": 7000,
        }

    @patch("bites.core.run_tooth_model.run_subprocess_with_cleanup")
    def test_successful_run(self, mock_subproc, exe_path, params, output_dir, logger):
        """Successful subprocess should return tm_run_complete=True."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_subproc.return_value = mock_process

        complete, duration, start, run_id = run_tooth_model_as_subprocess(
            exe_path, params, 300, 1000, output_dir, logger
        )

        assert complete is True
        assert isinstance(duration, int)
        assert isinstance(start, datetime.datetime)
        assert isinstance(run_id, str)
        assert "run" in run_id

    @patch("bites.core.run_tooth_model._get_last_iteration_from_output", return_value=5000)
    @patch("bites.core.run_tooth_model._delete_if_corrupted_mesh", return_value=None)
    @patch("bites.core.run_tooth_model.run_subprocess_with_cleanup")
    def test_timeout_returns_incomplete(
        self, mock_subproc, mock_del, mock_last, exe_path, params, output_dir, logger
    ):
        """Timeout should return tm_run_complete=False."""
        mock_subproc.side_effect = subprocess.TimeoutExpired("cmd", 300)

        complete, duration, start, run_id = run_tooth_model_as_subprocess(
            exe_path, params, 300, 1000, output_dir, logger
        )

        assert complete is False
        mock_del.assert_called_once()

    @patch("bites.core.run_tooth_model.run_subprocess_with_cleanup")
    def test_keyboard_interrupt_returns_incomplete(
        self, mock_subproc, exe_path, params, output_dir, logger
    ):
        """KeyboardInterrupt should return tm_run_complete=False."""
        mock_subproc.side_effect = KeyboardInterrupt()

        complete, duration, start, run_id = run_tooth_model_as_subprocess(
            exe_path, params, 300, 1000, output_dir, logger
        )

        assert complete is False

    @patch("bites.core.run_tooth_model.run_subprocess_with_cleanup")
    def test_generic_exception_returns_incomplete(
        self, mock_subproc, exe_path, params, output_dir, logger
    ):
        """An unexpected exception should return tm_run_complete=False."""
        mock_subproc.side_effect = RuntimeError("unknown error")

        complete, duration, start, run_id = run_tooth_model_as_subprocess(
            exe_path, params, 300, 1000, output_dir, logger
        )

        assert complete is False

    def test_missing_executable_raises(self, params, output_dir, logger):
        """Missing executable should raise FileNotFoundError."""
        fake_exe = Path("/no/such/tooth_model.exe")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            run_tooth_model_as_subprocess(
                fake_exe, params, 300, 1000, output_dir, logger
            )

    @patch("bites.core.run_tooth_model.run_subprocess_with_cleanup")
    def test_nonzero_returncode_still_complete(
        self, mock_subproc, exe_path, params, output_dir, logger
    ):
        """A non-zero return code is a warning, not a failure."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = b"some warning"
        mock_subproc.return_value = mock_process

        complete, duration, start, run_id = run_tooth_model_as_subprocess(
            exe_path, params, 300, 1000, output_dir, logger
        )

        # The process DID complete (returncode exists), so True
        assert complete is True

    @patch("bites.core.run_tooth_model.run_subprocess_with_cleanup")
    def test_creates_output_directory(self, mock_subproc, exe_path, params, tmp_path, logger):
        """Should create the output directory if it doesn't exist."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_subproc.return_value = mock_process

        new_dir = tmp_path / "new_output"
        assert not new_dir.exists()

        run_tooth_model_as_subprocess(exe_path, params, 300, 1000, new_dir, logger)
        assert new_dir.exists()
