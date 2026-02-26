"""
Tests for bites.core.chamfer_distance module.

Covers:
- calculate_chamfer_distances(): Chamfer distance computation between meshes
  - Identical meshes -> near-zero distance
  - Different meshes -> positive distance
  - Missing mesh handling
  - Non-trimesh object handling
  - Empty vertex handling
- sample_surface_points(): uniform surface sampling
"""

import numpy as np
import pytest
import trimesh

from bites.core.chamfer_distance import (
    calculate_chamfer_distances,
    sample_surface_points,
)


# ---------------------------------------------------------------------------
# calculate_chamfer_distances
# ---------------------------------------------------------------------------

class TestCalculateChamferDistances:
    """Tests for calculate_chamfer_distances()."""

    def test_identical_meshes_near_zero(self, simple_trimesh, mock_logger):
        """Chamfer distance between identical meshes is near zero."""
        aligned = {"algo1": simple_trimesh}
        target = {"algo1": simple_trimesh.copy()}
        result = calculate_chamfer_distances(aligned, target, ["algo1"],
                                             logger=mock_logger)
        assert "algo1" in result
        assert result["algo1"] < 1e-6

    def test_different_meshes_positive(self, mock_logger):
        """Chamfer distance between different meshes is positive."""
        mesh_a = trimesh.creation.box(extents=[1, 1, 1])
        mesh_b = trimesh.creation.box(extents=[1, 1, 1])
        mesh_b.apply_translation([5.0, 0.0, 0.0])

        aligned = {"algo1": mesh_a}
        target = {"algo1": mesh_b}
        result = calculate_chamfer_distances(aligned, target, ["algo1"],
                                             logger=mock_logger)
        assert result["algo1"] > 0.0

    def test_missing_aligned_mesh_skipped(self, simple_trimesh, mock_logger):
        """Missing aligned mesh for an algorithm is skipped."""
        aligned = {}  # no mesh for algo1
        target = {"algo1": simple_trimesh}
        result = calculate_chamfer_distances(aligned, target, ["algo1"],
                                             logger=mock_logger)
        assert "algo1" not in result

    def test_missing_target_mesh_skipped(self, simple_trimesh, mock_logger):
        """Missing target mesh for an algorithm is skipped."""
        aligned = {"algo1": simple_trimesh}
        target = {}  # no mesh for algo1
        result = calculate_chamfer_distances(aligned, target, ["algo1"],
                                             logger=mock_logger)
        assert "algo1" not in result

    def test_non_trimesh_skipped(self, simple_trimesh, mock_logger):
        """Non-Trimesh object is skipped."""
        aligned = {"algo1": "not a mesh"}
        target = {"algo1": simple_trimesh}
        result = calculate_chamfer_distances(aligned, target, ["algo1"],
                                             logger=mock_logger)
        assert "algo1" not in result

    def test_multiple_algorithms(self, mock_logger):
        """Handles multiple algorithms independently."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        aligned = {"algo1": mesh, "algo2": mesh}
        target = {"algo1": mesh.copy(), "algo2": mesh.copy()}
        result = calculate_chamfer_distances(aligned, target,
                                             ["algo1", "algo2"],
                                             logger=mock_logger)
        assert len(result) == 2

    def test_returns_dict(self, simple_trimesh, mock_logger):
        """Returns a dict mapping algorithm names to float distances."""
        aligned = {"algo1": simple_trimesh}
        target = {"algo1": simple_trimesh.copy()}
        result = calculate_chamfer_distances(aligned, target, ["algo1"],
                                             logger=mock_logger)
        assert isinstance(result, dict)
        assert isinstance(result["algo1"], float)

    def test_empty_algorithms_list(self, mock_logger):
        """Empty algorithm list returns empty dict."""
        result = calculate_chamfer_distances({}, {}, [], logger=mock_logger)
        assert result == {}


# ---------------------------------------------------------------------------
# sample_surface_points
# ---------------------------------------------------------------------------

class TestSampleSurfacePoints:
    """Tests for sample_surface_points()."""

    def test_returns_correct_count(self):
        """Returns the requested number of sample points."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        points = sample_surface_points(mesh, num_samples=100)
        assert points.shape == (100, 3)

    def test_returns_float64(self):
        """Returns float64 array."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        points = sample_surface_points(mesh, num_samples=10)
        assert points.dtype == np.float64

    def test_points_within_mesh_bounds(self):
        """Sampled points lie within the mesh bounding box."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        points = sample_surface_points(mesh, num_samples=100)
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        assert np.all(points >= bounds[0] - 1e-6)
        assert np.all(points <= bounds[1] + 1e-6)
