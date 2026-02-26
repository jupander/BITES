"""
Tests for bites.mesh.read_coff module.

Covers:
- read_coff_parts(): COFF file parsing (vertices, faces, colors)
  - Valid triangle and quad meshes
  - Color parsing in 0-1 and 0-255 ranges
  - Triangulation of quads and n-gons
  - Duplicate face removal
  - Error handling for malformed files
- read_coff_mesh(): COFF-to-trimesh conversion
"""

import numpy as np
import pytest
import trimesh

from bites.mesh.read_coff import read_coff_parts, read_coff_mesh


# ---------------------------------------------------------------------------
# Helpers to write COFF files
# ---------------------------------------------------------------------------

def _write_coff(path, header="COFF", counts="4 4 0", vertices=None, faces=None):
    """Write a minimal COFF file for testing."""
    if vertices is None:
        vertices = [
            "0.0 0.0 0.0",
            "1.0 0.0 0.0",
            "0.5 1.0 0.0",
            "0.5 0.5 1.0",
        ]
    if faces is None:
        faces = [
            "3 0 1 2",
            "3 0 1 3",
            "3 1 2 3",
            "3 0 2 3",
        ]
    lines = [header, counts] + vertices + faces
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# read_coff_parts -- valid files
# ---------------------------------------------------------------------------

class TestReadCoffPartsValid:
    """Tests for read_coff_parts() with valid COFF files."""

    def test_reads_triangle_mesh(self, tmp_path):
        """Reads a valid triangle COFF file returning arrays."""
        coff = _write_coff(tmp_path / "tri.off")
        vertices, faces, colors = read_coff_parts(coff)
        assert isinstance(vertices, np.ndarray)
        assert isinstance(faces, np.ndarray)

    def test_vertex_count(self, tmp_path):
        """Returns correct number of vertices."""
        coff = _write_coff(tmp_path / "tri.off")
        vertices, _, _ = read_coff_parts(coff)
        assert len(vertices) == 4

    def test_vertex_shape(self, tmp_path):
        """Vertices have shape (N, 3)."""
        coff = _write_coff(tmp_path / "tri.off")
        vertices, _, _ = read_coff_parts(coff)
        assert vertices.shape == (4, 3)

    def test_face_count(self, tmp_path):
        """Returns correct number of triangulated faces."""
        coff = _write_coff(tmp_path / "tri.off")
        _, faces, _ = read_coff_parts(coff)
        assert len(faces) == 4

    def test_face_shape(self, tmp_path):
        """Faces have shape (N, 3) (all triangles)."""
        coff = _write_coff(tmp_path / "tri.off")
        _, faces, _ = read_coff_parts(coff)
        assert faces.shape[1] == 3

    def test_no_colors_returns_none(self, tmp_path):
        """File without color data returns None for colors."""
        coff = _write_coff(tmp_path / "tri.off")
        _, _, colors = read_coff_parts(coff)
        assert colors is None

    def test_vertex_dtype(self, tmp_path):
        """Vertex array has float64 dtype."""
        coff = _write_coff(tmp_path / "tri.off")
        vertices, _, _ = read_coff_parts(coff)
        assert vertices.dtype == np.float64

    def test_face_dtype(self, tmp_path):
        """Face array has int32 dtype."""
        coff = _write_coff(tmp_path / "tri.off")
        _, faces, _ = read_coff_parts(coff)
        assert faces.dtype == np.int32


# ---------------------------------------------------------------------------
# read_coff_parts -- colors
# ---------------------------------------------------------------------------

class TestReadCoffPartsColors:
    """Tests for color parsing in read_coff_parts()."""

    def test_colors_01_range_scaled_to_255(self, tmp_path):
        """Colors in 0-1 range are scaled to 0-255 uint8."""
        vertices = [
            "0.0 0.0 0.0 1.0 0.0 0.0",  # red
            "1.0 0.0 0.0 0.0 1.0 0.0",  # green
            "0.5 1.0 0.0 0.0 0.0 1.0",  # blue
        ]
        faces = ["3 0 1 2"]
        coff = _write_coff(tmp_path / "color01.off", counts="3 1 0",
                           vertices=vertices, faces=faces)
        _, _, colors = read_coff_parts(coff)

        assert colors is not None
        assert colors.dtype == np.uint8
        # Red vertex: (1.0, 0.0, 0.0) -> (255, 0, 0, 255)
        assert colors[0][0] == 255
        assert colors[0][1] == 0
        assert colors[0][3] == 255  # alpha

    def test_colors_0255_range_kept(self, tmp_path):
        """Colors in 0-255 range are kept as-is."""
        vertices = [
            "0.0 0.0 0.0 200 100 50",
            "1.0 0.0 0.0 100 200 50",
            "0.5 1.0 0.0 50 100 200",
        ]
        faces = ["3 0 1 2"]
        coff = _write_coff(tmp_path / "color255.off", counts="3 1 0",
                           vertices=vertices, faces=faces)
        _, _, colors = read_coff_parts(coff)

        assert colors is not None
        assert colors[0][0] == 200
        assert colors[0][1] == 100
        assert colors[0][2] == 50

    def test_colors_shape(self, tmp_path):
        """Color array has shape (N, 4) -- RGBA."""
        vertices = [
            "0.0 0.0 0.0 1.0 0.0 0.0",
            "1.0 0.0 0.0 0.0 1.0 0.0",
            "0.5 1.0 0.0 0.0 0.0 1.0",
        ]
        faces = ["3 0 1 2"]
        coff = _write_coff(tmp_path / "color.off", counts="3 1 0",
                           vertices=vertices, faces=faces)
        _, _, colors = read_coff_parts(coff)
        assert colors.shape == (3, 4)


# ---------------------------------------------------------------------------
# read_coff_parts -- triangulation
# ---------------------------------------------------------------------------

class TestReadCoffPartsTriangulation:
    """Tests for quad and n-gon triangulation."""

    def test_quad_splits_into_two_triangles(self, tmp_path):
        """A quad face is split into 2 triangles."""
        vertices = [
            "0.0 0.0 0.0",
            "1.0 0.0 0.0",
            "1.0 1.0 0.0",
            "0.0 1.0 0.0",
        ]
        faces = ["4 0 1 2 3"]  # one quad
        coff = _write_coff(tmp_path / "quad.off", counts="4 1 0",
                           vertices=vertices, faces=faces)
        _, tri_faces, _ = read_coff_parts(coff)
        assert len(tri_faces) == 2

    def test_pentagon_fan_triangulated(self, tmp_path):
        """A pentagon (5-gon) is fan-triangulated into 3 triangles."""
        vertices = [
            "0.0 0.0 0.0",
            "1.0 0.0 0.0",
            "1.5 1.0 0.0",
            "0.5 1.5 0.0",
            "-0.5 1.0 0.0",
        ]
        faces = ["5 0 1 2 3 4"]
        coff = _write_coff(tmp_path / "pent.off", counts="5 1 0",
                           vertices=vertices, faces=faces)
        _, tri_faces, _ = read_coff_parts(coff)
        assert len(tri_faces) == 3


# ---------------------------------------------------------------------------
# read_coff_parts -- duplicate removal
# ---------------------------------------------------------------------------

class TestReadCoffPartsDuplicates:
    """Tests for duplicate face removal."""

    def test_duplicate_faces_removed(self, tmp_path):
        """Duplicate triangles (same vertices, any winding) are removed."""
        vertices = [
            "0.0 0.0 0.0",
            "1.0 0.0 0.0",
            "0.5 1.0 0.0",
        ]
        faces = [
            "3 0 1 2",
            "3 2 1 0",  # same triangle, reversed winding
        ]
        coff = _write_coff(tmp_path / "dup.off", counts="3 2 0",
                           vertices=vertices, faces=faces)
        _, tri_faces, _ = read_coff_parts(coff)
        assert len(tri_faces) == 1


# ---------------------------------------------------------------------------
# read_coff_parts -- error handling
# ---------------------------------------------------------------------------

class TestReadCoffPartsErrors:
    """Tests for error handling in read_coff_parts()."""

    def test_empty_file_raises(self, tmp_path):
        """Empty file raises ValueError."""
        empty = tmp_path / "empty.off"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_coff_parts(empty)

    def test_invalid_header_raises(self, tmp_path):
        """Non-COFF header raises ValueError."""
        bad = tmp_path / "bad.off"
        bad.write_text("PLY\n4 4 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n")
        with pytest.raises(ValueError, match="does not appear to be a COFF"):
            read_coff_parts(bad)

    def test_missing_counts_raises(self, tmp_path):
        """File with only header and no counts raises ValueError."""
        short = tmp_path / "short.off"
        short.write_text("COFF\n")
        with pytest.raises(ValueError, match="missing"):
            read_coff_parts(short)

    def test_truncated_vertices_raises(self, tmp_path):
        """File ending mid-vertex-block raises ValueError."""
        trunc = tmp_path / "trunc.off"
        trunc.write_text("COFF\n4 1 0\n0 0 0\n1 0 0\n")  # only 2 of 4 vertices
        with pytest.raises(ValueError, match="ended unexpectedly"):
            read_coff_parts(trunc)

    def test_invalid_vertex_indices_skipped(self, tmp_path):
        """Faces with out-of-range vertex indices are skipped."""
        vertices = [
            "0.0 0.0 0.0",
            "1.0 0.0 0.0",
            "0.5 1.0 0.0",
        ]
        faces = [
            "3 0 1 2",    # valid
            "3 0 1 99",   # invalid index
        ]
        coff = _write_coff(tmp_path / "badidx.off", counts="3 2 0",
                           vertices=vertices, faces=faces)
        _, tri_faces, _ = read_coff_parts(coff)
        assert len(tri_faces) == 1  # only the valid face

    def test_comments_and_blank_lines_ignored(self, tmp_path):
        """Comments and blank lines are skipped."""
        content = "# A comment\nCOFF\n\n3 1 0\n0 0 0\n1 0 0\n0.5 1 0\n# Another comment\n3 0 1 2\n"
        coff = tmp_path / "comments.off"
        coff.write_text(content)
        vertices, faces, _ = read_coff_parts(coff)
        assert len(vertices) == 3
        assert len(faces) == 1


# ---------------------------------------------------------------------------
# read_coff_mesh
# ---------------------------------------------------------------------------

class TestReadCoffMesh:
    """Tests for read_coff_mesh() -> trimesh.Trimesh."""

    def test_returns_trimesh(self, tmp_path):
        """Returns a trimesh.Trimesh object."""
        coff = _write_coff(tmp_path / "mesh.off")
        mesh = read_coff_mesh(coff)
        assert isinstance(mesh, trimesh.Trimesh)

    def test_mesh_vertex_count(self, tmp_path):
        """Mesh has correct number of vertices."""
        coff = _write_coff(tmp_path / "mesh.off")
        mesh = read_coff_mesh(coff)
        assert len(mesh.vertices) == 4

    def test_mesh_face_count(self, tmp_path):
        """Mesh has correct number of faces."""
        coff = _write_coff(tmp_path / "mesh.off")
        mesh = read_coff_mesh(coff)
        assert len(mesh.faces) == 4

    def test_mesh_with_colors(self, tmp_path):
        """Mesh has vertex colors when COFF file has color data."""
        vertices = [
            "0.0 0.0 0.0 1.0 0.0 0.0",
            "1.0 0.0 0.0 0.0 1.0 0.0",
            "0.5 1.0 0.0 0.0 0.0 1.0",
        ]
        faces = ["3 0 1 2"]
        coff = _write_coff(tmp_path / "color.off", counts="3 1 0",
                           vertices=vertices, faces=faces)
        mesh = read_coff_mesh(coff)
        # trimesh stores vertex colors in visual
        assert mesh.visual.kind == "vertex"
