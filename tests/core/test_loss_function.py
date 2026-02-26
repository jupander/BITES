"""
Tests for bites.core.loss_function module.

Covers:
- align_meshes: algorithm dispatch, combination algorithms, error handling
- evaluate_loss: end-to-end mesh evaluation pipeline with mocked dependencies

All mesh loading, alignment, and file I/O is mocked.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import optuna
import pytest
import trimesh

from bites.core.loss_function import align_meshes, evaluate_loss


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Return a MagicMock config with standard test settings."""
    config = MagicMock()
    config.alignment_algorithms = ["ICP"]  # default; override per test
    config.logger = logging.getLogger("test_loss")
    config.target_shape_path = "/fake/target.off"
    config.min_cut_height = None
    return config


@pytest.fixture
def valid_mesh():
    """Return a simple valid trimesh with faces."""
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


@pytest.fixture
def mock_target_mesh():
    """Return a MagicMock that looks like a valid trimesh."""
    mesh = MagicMock(spec=trimesh.Trimesh)
    mesh.faces = np.array([[0, 1, 2]])
    return mesh


# ============================================================================
# TestAlignMeshes
# ============================================================================

class TestAlignMeshes:
    """Tests for align_meshes algorithm dispatch and error handling."""

    @patch("bites.core.loss_function.statac")
    def test_st_algorithm(self, mock_statac, mock_config, valid_mesh):
        """ST algorithm should call scale_and_center_mesh."""
        mock_config.alignment_algorithms = ["ST"]
        mock_statac.scale_and_center_mesh.return_value = (
            valid_mesh, 1.0, np.array([0.0, 0.0, 0.0])
        )

        aligned, sf, rm, tv, tgt, ch = align_meshes(mock_config, valid_mesh, valid_mesh)
        assert "ST" in aligned
        assert sf["ST"] == 1.0
        np.testing.assert_array_equal(rm["ST"], np.eye(3))

    @patch("bites.core.loss_function.coherent_point_drift")
    def test_cpd_only_algorithm(self, mock_cpd, mock_config, valid_mesh):
        """CPD-only should call perform_CPD_on_mesh."""
        mock_config.alignment_algorithms = ["CPD-only"]
        mock_cpd.perform_CPD_on_mesh.return_value = (
            valid_mesh, 1.0, np.eye(3), np.zeros(3)
        )

        aligned, sf, rm, tv, tgt, ch = align_meshes(mock_config, valid_mesh, valid_mesh)
        assert "CPD-only" in aligned
        mock_cpd.perform_CPD_on_mesh.assert_called_once()

    @patch("bites.core.loss_function.coherent_point_drift")
    @patch("bites.core.loss_function.statac")
    def test_st_cpd_combines_transforms(self, mock_statac, mock_cpd, mock_config, valid_mesh):
        """ST-CPD should combine transforms from ST and CPD steps."""
        mock_config.alignment_algorithms = ["ST-CPD"]
        st_translation = np.array([1.0, 2.0, 3.0])
        cpd_translation = np.array([0.1, 0.2, 0.3])
        mock_statac.scale_and_center_mesh.return_value = (
            valid_mesh, 2.0, st_translation
        )
        mock_cpd.perform_CPD_on_mesh.return_value = (
            valid_mesh, 0.5, np.eye(3), cpd_translation
        )

        aligned, sf, rm, tv, tgt, ch = align_meshes(mock_config, valid_mesh, valid_mesh)
        assert "ST-CPD" in aligned
        # Scale factors multiply
        assert sf["ST-CPD"] == pytest.approx(1.0)  # 2.0 * 0.5
        # Translation vectors add
        np.testing.assert_array_almost_equal(
            tv["ST-CPD"], cpd_translation + st_translation
        )

    @patch("bites.core.loss_function.statac")
    def test_statac_c_replaces_target(self, mock_statac, mock_config, valid_mesh):
        """STatac-C should replace the target mesh with the cut version."""
        mock_config.alignment_algorithms = ["STatac-C"]
        cut_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_statac.ST_align_tops_and_cut.return_value = (
            valid_mesh, 1.0, np.zeros(3), cut_mesh, MagicMock(), 6.0
        )

        aligned, sf, rm, tv, tgt, ch = align_meshes(mock_config, valid_mesh, valid_mesh)
        assert tgt["STatac-C"] is cut_mesh
        assert ch["STatac-C"] == 6.0

    def test_unknown_algorithm_skipped(self, mock_config, valid_mesh):
        """An unknown algorithm should be logged and skipped."""
        mock_config.alignment_algorithms = ["NonExistent"]
        aligned, sf, rm, tv, tgt, ch = align_meshes(mock_config, valid_mesh, valid_mesh)
        assert len(aligned) == 0

    @patch("bites.core.loss_function.statac")
    def test_algorithm_exception_skipped(self, mock_statac, mock_config, valid_mesh):
        """If an algorithm raises an exception, it should be skipped."""
        mock_config.alignment_algorithms = ["ST"]
        mock_statac.scale_and_center_mesh.side_effect = RuntimeError("convergence failed")

        aligned, sf, rm, tv, tgt, ch = align_meshes(mock_config, valid_mesh, valid_mesh)
        assert len(aligned) == 0

    @patch("bites.core.loss_function.statac")
    def test_none_output_skipped(self, mock_statac, mock_config, valid_mesh):
        """If alignment returns None for a required field, skip the algorithm."""
        mock_config.alignment_algorithms = ["ST"]
        mock_statac.scale_and_center_mesh.return_value = (
            None, 1.0, np.zeros(3)  # aligned_mesh is None
        )

        aligned, sf, rm, tv, tgt, ch = align_meshes(mock_config, valid_mesh, valid_mesh)
        assert len(aligned) == 0

    @patch("bites.core.loss_function.statac")
    def test_multiple_algorithms_independent(self, mock_statac, mock_config, valid_mesh):
        """Multiple algorithms should run independently — one failure doesn't affect others."""
        mock_config.alignment_algorithms = ["ST", "STatac-C"]
        # ST succeeds
        mock_statac.scale_and_center_mesh.return_value = (
            valid_mesh, 1.0, np.zeros(3)
        )
        # STatac-C fails
        mock_statac.ST_align_tops_and_cut.side_effect = RuntimeError("cut failed")

        aligned, sf, rm, tv, tgt, ch = align_meshes(mock_config, valid_mesh, valid_mesh)
        assert "ST" in aligned
        assert "STatac-C" not in aligned


# ============================================================================
# TestEvaluateLoss
# ============================================================================

class TestEvaluateLoss:
    """Tests for evaluate_loss integration with mocked align_meshes and I/O."""

    @patch("bites.core.loss_function.output_manager")
    @patch("bites.core.loss_function.chamfer_distance")
    @patch("bites.core.loss_function.align_meshes")
    @patch("bites.core.loss_function.trimesh")
    def test_successful_evaluation(
        self, mock_trimesh, mock_align, mock_chamfer, mock_output, mock_config, valid_mesh
    ):
        """Full successful evaluation should return loss, properties, and other meshes."""
        mock_config.alignment_algorithms = ["ICP"]

        # Mock target mesh loading
        mock_trimesh.load.return_value = valid_mesh

        # Mock alignment
        mock_align.return_value = (
            {"ICP": valid_mesh},    # aligned_meshes
            {"ICP": 1.0},           # scale_factors
            {"ICP": np.eye(3)},     # rotation_matrices
            {"ICP": np.zeros(3)},   # translation_vectors
            {"ICP": valid_mesh},    # target_meshes
            {"ICP": None},          # used_cut_heights
        )

        # Mock chamfer distance
        mock_chamfer.calculate_chamfer_distances.return_value = {"ICP": 0.123}

        # Mock file writing
        mock_output.write_aligned_meshes_to_files.return_value = {5000: "run__5000"}

        mesh_dict = {5000: valid_mesh}
        loss, props, others = evaluate_loss(mock_config, mesh_dict, "run123", Path("/out"))

        assert loss == pytest.approx(0.123)
        assert props["best_alignment_algorithm"] == "ICP"
        assert props["iter_of_chosen_mesh"] == 5000
        assert "best_chamfer_distance_by_iter_dict" in others

    @patch("bites.core.loss_function.trimesh")
    def test_empty_target_mesh_prunes_trial(self, mock_trimesh, mock_config, valid_mesh):
        """A target mesh with no faces should raise TrialPruned."""
        empty_mesh = MagicMock(spec=trimesh.Trimesh)
        empty_mesh.faces = np.array([])
        mock_trimesh.load.return_value = empty_mesh

        with pytest.raises(optuna.TrialPruned, match="no faces"):
            evaluate_loss(mock_config, {5000: valid_mesh}, "run", Path("/out"))

    @patch("bites.core.loss_function.align_meshes")
    @patch("bites.core.loss_function.trimesh")
    def test_all_alignment_failures_prune_trial(
        self, mock_trimesh, mock_align, mock_config, valid_mesh
    ):
        """If all iterations fail alignment, trial should be pruned."""
        mock_trimesh.load.return_value = valid_mesh
        mock_align.side_effect = RuntimeError("alignment exploded")

        with pytest.raises(optuna.TrialPruned):
            evaluate_loss(mock_config, {5000: valid_mesh}, "run", Path("/out"))

    @patch("bites.core.loss_function.output_manager")
    @patch("bites.core.loss_function.chamfer_distance")
    @patch("bites.core.loss_function.align_meshes")
    @patch("bites.core.loss_function.trimesh")
    def test_picks_best_iteration(
        self, mock_trimesh, mock_align, mock_chamfer, mock_output, mock_config, valid_mesh
    ):
        """With multiple iterations, the one with lowest chamfer distance should be chosen."""
        mock_config.alignment_algorithms = ["ICP"]
        mock_trimesh.load.return_value = valid_mesh

        mock_align.return_value = (
            {"ICP": valid_mesh}, {"ICP": 1.0}, {"ICP": np.eye(3)},
            {"ICP": np.zeros(3)}, {"ICP": valid_mesh}, {"ICP": None},
        )

        # Chamfer returns different values for different iterations
        mock_chamfer.calculate_chamfer_distances.side_effect = [
            {"ICP": 0.5},   # iter 5000 — worse
            {"ICP": 0.1},   # iter 7000 — best
        ]
        mock_output.write_aligned_meshes_to_files.return_value = {
            5000: "run__5000", 7000: "run__7000"
        }

        mesh_dict = {5000: valid_mesh, 7000: valid_mesh}
        loss, props, _ = evaluate_loss(mock_config, mesh_dict, "run", Path("/out"))

        assert loss == pytest.approx(0.1)
        assert props["iter_of_chosen_mesh"] == 7000

    @patch("bites.core.loss_function.align_meshes")
    @patch("bites.core.loss_function.trimesh")
    def test_none_mesh_skipped(self, mock_trimesh, mock_align, mock_config, valid_mesh):
        """Iterations with None meshes should be skipped during pre-check."""
        mock_trimesh.load.return_value = valid_mesh

        # Only iter 7000 has a valid mesh
        mock_align.return_value = (
            {"ICP": valid_mesh}, {"ICP": 1.0}, {"ICP": np.eye(3)},
            {"ICP": np.zeros(3)}, {"ICP": valid_mesh}, {"ICP": None},
        )

        mesh_dict = {5000: None, 7000: valid_mesh}

        with patch("bites.core.loss_function.chamfer_distance") as mock_cd, \
             patch("bites.core.loss_function.output_manager") as mock_ou:
            mock_cd.calculate_chamfer_distances.return_value = {"ICP": 0.2}
            mock_ou.write_aligned_meshes_to_files.return_value = {7000: "run__7000"}

            loss, props, _ = evaluate_loss(mock_config, mesh_dict, "run", Path("/out"))
            # Only called once (for iter 7000, not for iter 5000 which was None)
            assert mock_align.call_count == 1

    @patch("bites.core.loss_function.trimesh")
    def test_min_cut_height_parsing(self, mock_trimesh, mock_config, valid_mesh):
        """config.min_cut_height should be parsed as float; None is handled."""
        mock_config.min_cut_height = "6.5"
        mock_trimesh.load.return_value = valid_mesh

        with patch("bites.core.loss_function.align_meshes") as mock_align, \
             patch("bites.core.loss_function.chamfer_distance") as mock_cd, \
             patch("bites.core.loss_function.output_manager") as mock_ou:
            mock_align.return_value = (
                {"ICP": valid_mesh}, {"ICP": 1.0}, {"ICP": np.eye(3)},
                {"ICP": np.zeros(3)}, {"ICP": valid_mesh}, {"ICP": None},
            )
            mock_cd.calculate_chamfer_distances.return_value = {"ICP": 0.1}
            mock_ou.write_aligned_meshes_to_files.return_value = {5000: "run__5000"}

            evaluate_loss(mock_config, {5000: valid_mesh}, "run", Path("/out"))
            # Verify min_cut_height was passed to align_meshes
            _, kwargs = mock_align.call_args
            # It's passed as a positional arg
            assert mock_align.call_args[0][3] == 6.5  # min_cut_height
