"""
Tests for bites.mesh.trimesh_utils module.

Covers:
- compute_edges_from_faces(): edge extraction from face arrays
- validate_trimesh(): mesh validity checks
- convert_color_to_rgba(): color format conversion
"""

import numpy as np
import pytest
import trimesh

from bites.mesh.trimesh_utils import (
    compute_edges_from_faces,
    validate_trimesh,
    convert_color_to_rgba,
)


# ---------------------------------------------------------------------------
# compute_edges_from_faces
# ---------------------------------------------------------------------------

class TestComputeEdgesFromFaces:
    """Tests for compute_edges_from_faces()."""

    def test_single_triangle_produces_3_edges(self):
        """A single triangle has exactly 3 unique edges."""
        faces = np.array([[0, 1, 2]])
        edges = compute_edges_from_faces(faces)
        assert len(edges) == 3

    def test_two_triangles_shared_edge(self):
        """Two triangles sharing one edge produce 5 unique edges."""
        faces = np.array([
            [0, 1, 2],
            [1, 2, 3],
        ])
        edges = compute_edges_from_faces(faces)
        assert len(edges) == 5

    def test_edge_pairs_input(self):
        """Handles (N, 2) edge pair input correctly."""
        edge_pairs = np.array([
            [0, 1],
            [1, 2],
            [2, 0],
        ])
        edges = compute_edges_from_faces(edge_pairs)
        assert len(edges) == 3

    def test_duplicate_edge_pairs_deduplicated(self):
        """Duplicate edge pairs (reversed) are counted once."""
        edge_pairs = np.array([
            [0, 1],
            [1, 0],  # same edge, reversed
        ])
        edges = compute_edges_from_faces(edge_pairs)
        assert len(edges) == 1

    def test_output_shape(self):
        """Output is (M, 2) array."""
        faces = np.array([[0, 1, 2]])
        edges = compute_edges_from_faces(faces)
        assert edges.ndim == 2
        assert edges.shape[1] == 2

    def test_empty_faces(self):
        """Empty face array returns (0, 2) array."""
        faces = np.zeros((0, 3), dtype=int)
        edges = compute_edges_from_faces(faces)
        assert edges.shape == (0, 2)

    def test_invalid_shape_raises(self):
        """Non-(N, 2) or (N, 3) input raises ValueError."""
        with pytest.raises(ValueError, match="must be an"):
            compute_edges_from_faces(np.array([[0, 1, 2, 3]]))

    def test_1d_input_raises(self):
        """1D input raises ValueError."""
        with pytest.raises(ValueError, match="must be an"):
            compute_edges_from_faces(np.array([0, 1, 2]))


# ---------------------------------------------------------------------------
# validate_trimesh
# ---------------------------------------------------------------------------

class TestValidateTrimesh:
    """Tests for validate_trimesh()."""

    def test_valid_mesh_returns_true(self, simple_trimesh):
        """Returns True for a valid non-empty mesh."""
        assert validate_trimesh(simple_trimesh) is True

    def test_empty_mesh_returns_false(self):
        """Returns False for a mesh with no faces."""
        empty = trimesh.Trimesh()
        assert validate_trimesh(empty) is False

    def test_non_trimesh_returns_false(self):
        """Returns False for a non-Trimesh object."""
        assert validate_trimesh("not a mesh") is False
        assert validate_trimesh(None) is False
        assert validate_trimesh(42) is False

    def test_custom_operation_name(self, simple_trimesh):
        """Accepts custom operation name for error messages."""
        assert validate_trimesh(simple_trimesh, operation="alignment") is True


# ---------------------------------------------------------------------------
# convert_color_to_rgba
# ---------------------------------------------------------------------------

class TestConvertColorToRgba:
    """Tests for convert_color_to_rgba()."""

    def test_rgb_tuple_gets_alpha(self):
        """RGB tuple (3 components) gets alpha appended."""
        result = convert_color_to_rgba((1.0, 0.5, 0.0))
        assert len(result) == 4
        assert result == (1.0, 0.5, 0.0, 1.0)

    def test_rgba_passthrough(self):
        """RGBA tuple (4 components) is returned as-is."""
        result = convert_color_to_rgba((1.0, 0.5, 0.0, 0.8))
        assert result == (1.0, 0.5, 0.0, 0.8)

    def test_list_input_works(self):
        """List input is converted to tuple."""
        result = convert_color_to_rgba([1.0, 0.5, 0.0])
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_string_passthrough(self):
        """String color is returned unchanged."""
        result = convert_color_to_rgba("red")
        assert result == "red"

    def test_custom_alpha(self):
        """Custom default_alpha is used for RGB input."""
        result = convert_color_to_rgba((1.0, 0.5, 0.0), default_alpha=0.5)
        assert result[3] == 0.5

    def test_invalid_length_raises(self):
        """Color with wrong number of components raises ValueError."""
        with pytest.raises(ValueError, match="Invalid color format"):
            convert_color_to_rgba((1.0, 0.5))

        with pytest.raises(ValueError, match="Invalid color format"):
            convert_color_to_rgba((1.0, 0.5, 0.0, 0.8, 0.2))
