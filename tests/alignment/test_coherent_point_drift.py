"""
Tests for bites.alignment.coherent_point_drift module.

Covers:
- perform_CPD_on_mesh: return structure, error handling
- CPDConvergenceError: custom exception

pycpd.RigidRegistration is mocked to avoid heavy computation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import trimesh

from bites.alignment.coherent_point_drift import (
    perform_CPD_on_mesh,
    CPDConvergenceError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def source_mesh():
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


@pytest.fixture
def target_mesh():
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    mesh.apply_translation([1.0, 1.0, 1.0])
    return mesh


# ============================================================================
# TestCPDConvergenceError
# ============================================================================

class TestCPDConvergenceError:
    def test_is_exception(self):
        assert issubclass(CPDConvergenceError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(CPDConvergenceError, match="diverged"):
            raise CPDConvergenceError("diverged")


# ============================================================================
# TestPerformCPDOnMesh
# ============================================================================

class TestPerformCPDOnMesh:
    """Tests for perform_CPD_on_mesh with mocked RigidRegistration."""

    @patch("bites.alignment.coherent_point_drift.RigidRegistration")
    def test_returns_four_values(self, mock_reg_cls, source_mesh, target_mesh):
        """Should return (aligned_mesh, scale, rotation, translation)."""
        n_verts = len(source_mesh.vertices)
        mock_reg = MagicMock()
        # register() returns (transformed_points, (scale, rotation, translation))
        mock_reg.register.return_value = (
            np.random.randn(n_verts, 3),  # TY
            (1.0, np.eye(3), np.zeros(3)),
        )
        mock_reg_cls.return_value = mock_reg

        aligned, s, R, t = perform_CPD_on_mesh(source_mesh, target_mesh)
        assert isinstance(aligned, trimesh.Trimesh)
        assert isinstance(s, float)
        assert R.shape == (3, 3)
        assert t.shape == (3,)

    @patch("bites.alignment.coherent_point_drift.RigidRegistration")
    def test_aligned_mesh_preserves_face_count(self, mock_reg_cls, source_mesh, target_mesh):
        """The aligned mesh should have the same number of faces as the source."""
        n_verts = len(source_mesh.vertices)
        mock_reg = MagicMock()
        mock_reg.register.return_value = (
            np.random.randn(n_verts, 3),
            (1.0, np.eye(3), np.zeros(3)),
        )
        mock_reg_cls.return_value = mock_reg

        aligned, *_ = perform_CPD_on_mesh(source_mesh, target_mesh)
        assert len(aligned.faces) == len(source_mesh.faces)

    @patch("bites.alignment.coherent_point_drift.RigidRegistration")
    def test_init_failure_raises_convergence_error(self, mock_reg_cls, source_mesh, target_mesh):
        """If RigidRegistration init fails, should raise CPDConvergenceError."""
        mock_reg_cls.side_effect = RuntimeError("bad matrix")

        with pytest.raises(CPDConvergenceError, match="initialize"):
            perform_CPD_on_mesh(source_mesh, target_mesh)

    @patch("bites.alignment.coherent_point_drift.RigidRegistration")
    def test_linalg_error_raises_convergence_error(self, mock_reg_cls, source_mesh, target_mesh):
        """LinAlgError during registration should raise CPDConvergenceError."""
        mock_reg = MagicMock()
        mock_reg.register.side_effect = np.linalg.LinAlgError("singular matrix")
        mock_reg_cls.return_value = mock_reg

        with pytest.raises(CPDConvergenceError, match="LinAlgError"):
            perform_CPD_on_mesh(source_mesh, target_mesh)

    @patch("bites.alignment.coherent_point_drift.RigidRegistration")
    def test_unexpected_error_raises_convergence_error(self, mock_reg_cls, source_mesh, target_mesh):
        """Any unexpected error during registration should raise CPDConvergenceError."""
        mock_reg = MagicMock()
        mock_reg.register.side_effect = ValueError("unexpected")
        mock_reg_cls.return_value = mock_reg

        with pytest.raises(CPDConvergenceError, match="Unexpected"):
            perform_CPD_on_mesh(source_mesh, target_mesh)

    @patch("bites.alignment.coherent_point_drift.RigidRegistration")
    def test_registration_called_with_scale(self, mock_reg_cls, source_mesh, target_mesh):
        """Registration should be initialized with scale=True."""
        n_verts = len(source_mesh.vertices)
        mock_reg = MagicMock()
        mock_reg.register.return_value = (
            np.random.randn(n_verts, 3),
            (1.0, np.eye(3), np.zeros(3)),
        )
        mock_reg_cls.return_value = mock_reg

        perform_CPD_on_mesh(source_mesh, target_mesh)
        _, kwargs = mock_reg_cls.call_args
        assert kwargs.get("scale") is True
