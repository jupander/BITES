"""
Tests for bites.alignment.statac module.

Covers:
- validate_mesh(): mesh validation with diagnostic info
- determine_cut_height_to_use(): cut height calculation logic
- scale_and_center_mesh(): mesh scaling and centering
- perform_cut_on_target_mesh(): mesh cutting (plane_slice, face_filter)
- _create_submesh_from_face_mask(): submesh extraction
"""

import numpy as np
import pytest
import trimesh

from bites.alignment.statac import (
    validate_mesh,
    scale_and_center_mesh,
    determine_cut_height_to_use,
    perform_cut_on_target_mesh,
    _create_submesh_from_face_mask,
)


# ---------------------------------------------------------------------------
# validate_mesh
# ---------------------------------------------------------------------------

class TestValidateMesh:
    """Tests for validate_mesh()."""

    def test_valid_mesh_returns_dict(self, simple_trimesh):
        """Returns a diagnostic info dict for a valid mesh."""
        info = validate_mesh(simple_trimesh, name="test")
        assert isinstance(info, dict)
        assert info["name"] == "test"
        assert info["is_trimesh"] is True
        assert info["vertices"] == 4
        assert info["faces"] == 4

    def test_info_contains_bounds(self, simple_trimesh):
        """Info dict contains bounds and centroid."""
        info = validate_mesh(simple_trimesh)
        assert info["bounds"] is not None
        assert info["centroid"] is not None
        assert info["extents"] is not None

    def test_non_trimesh_raises(self):
        """Raises ValueError for non-Trimesh input."""
        with pytest.raises(ValueError, match="not a trimesh.Trimesh"):
            validate_mesh("not a mesh", name="bad")

    def test_empty_mesh_raises(self):
        """Raises ValueError for an empty mesh."""
        empty = trimesh.Trimesh()
        with pytest.raises(ValueError, match="empty"):
            validate_mesh(empty, name="empty")

    def test_default_name(self, simple_trimesh):
        """Default name is 'mesh'."""
        info = validate_mesh(simple_trimesh)
        assert info["name"] == "mesh"


# ---------------------------------------------------------------------------
# determine_cut_height_to_use
# ---------------------------------------------------------------------------

class TestDetermineCutHeightToUse:
    """Tests for determine_cut_height_to_use()."""

    def test_default_uses_source_max_z(self):
        """Without override, returns source's max z-coordinate."""
        source_bounds = np.array([[0, 0, 0], [1, 1, 10]])
        target_bounds = np.array([[0, 0, 0], [1, 1, 20]])
        result = determine_cut_height_to_use(source_bounds, target_bounds)
        assert result == 10.0

    def test_override_takes_precedence(self):
        """override_cut_height is returned directly."""
        source_bounds = np.array([[0, 0, 0], [1, 1, 10]])
        target_bounds = np.array([[0, 0, 0], [1, 1, 20]])
        result = determine_cut_height_to_use(source_bounds, target_bounds,
                                              override_cut_height=5.0)
        assert result == 5.0

    def test_min_cut_height_clamps(self):
        """Cut height won't go below min_cut_height."""
        source_bounds = np.array([[0, 0, 0], [1, 1, 3]])  # max_z = 3
        target_bounds = np.array([[0, 0, 0], [1, 1, 20]])
        result = determine_cut_height_to_use(source_bounds, target_bounds,
                                              min_cut_height=8)
        assert result == 8

    def test_no_clamping_when_above_min(self):
        """Cut height is not modified when already above min_cut_height."""
        source_bounds = np.array([[0, 0, 0], [1, 1, 15]])  # max_z = 15
        target_bounds = np.array([[0, 0, 0], [1, 1, 20]])
        result = determine_cut_height_to_use(source_bounds, target_bounds,
                                              min_cut_height=8)
        assert result == 15.0

    def test_override_ignores_min_cut_height(self):
        """override_cut_height bypasses min_cut_height check."""
        source_bounds = np.array([[0, 0, 0], [1, 1, 10]])
        target_bounds = np.array([[0, 0, 0], [1, 1, 20]])
        result = determine_cut_height_to_use(source_bounds, target_bounds,
                                              min_cut_height=8,
                                              override_cut_height=2.0)
        assert result == 2.0

    def test_no_min_cut_height_no_clamping(self):
        """Without min_cut_height, no clamping occurs."""
        source_bounds = np.array([[0, 0, 0], [1, 1, 3]])
        target_bounds = np.array([[0, 0, 0], [1, 1, 20]])
        result = determine_cut_height_to_use(source_bounds, target_bounds)
        assert result == 3.0


# ---------------------------------------------------------------------------
# scale_and_center_mesh
# ---------------------------------------------------------------------------

class TestScaleAndCenterMesh:
    """Tests for scale_and_center_mesh()."""

    def test_returns_three_values(self, box_source_mesh, box_target_mesh):
        """Returns (aligned_mesh, scale_factor, translation)."""
        result = scale_and_center_mesh(box_source_mesh, box_target_mesh)
        assert len(result) == 3

    def test_returns_trimesh(self, box_source_mesh, box_target_mesh):
        """First return value is a trimesh.Trimesh."""
        aligned, _, _ = scale_and_center_mesh(box_source_mesh, box_target_mesh)
        assert isinstance(aligned, trimesh.Trimesh)

    def test_does_not_mutate_input(self, box_source_mesh, box_target_mesh):
        """Original source mesh is not modified."""
        original_vertices = box_source_mesh.vertices.copy()
        scale_and_center_mesh(box_source_mesh, box_target_mesh)
        np.testing.assert_array_equal(box_source_mesh.vertices, original_vertices)

    def test_scale_factor_positive(self, box_source_mesh, box_target_mesh):
        """Scale factor is positive."""
        _, scale_factor, _ = scale_and_center_mesh(box_source_mesh, box_target_mesh)
        assert scale_factor > 0

    def test_scale_factor_when_target_larger(self, box_source_mesh, box_target_mesh):
        """Scale factor > 1 when target is larger than source."""
        _, scale_factor, _ = scale_and_center_mesh(box_source_mesh, box_target_mesh)
        assert scale_factor > 1.0

    @pytest.mark.parametrize("scale_axis", ["x", "y", "xy", "all"])
    def test_valid_scale_axes(self, box_source_mesh, box_target_mesh, scale_axis):
        """All valid scale_axis options work without error."""
        aligned, sf, tr = scale_and_center_mesh(box_source_mesh, box_target_mesh,
                                                 scale_axis=scale_axis)
        assert isinstance(aligned, trimesh.Trimesh)

    def test_invalid_scale_axis_raises(self, box_source_mesh, box_target_mesh):
        """Invalid scale_axis raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scale_axis"):
            scale_and_center_mesh(box_source_mesh, box_target_mesh,
                                  scale_axis="invalid")

    def test_invalid_center_raises(self, box_source_mesh, box_target_mesh):
        """Invalid center option raises ValueError."""
        with pytest.raises(ValueError, match="Invalid center"):
            scale_and_center_mesh(box_source_mesh, box_target_mesh,
                                  center="invalid")

    @pytest.mark.parametrize("center", ["AABB", "centroid"])
    def test_valid_center_options(self, box_source_mesh, box_target_mesh, center):
        """Both center options work without error."""
        aligned, _, _ = scale_and_center_mesh(box_source_mesh, box_target_mesh,
                                               center=center)
        assert isinstance(aligned, trimesh.Trimesh)

    def test_aligned_near_target_centroid(self, box_source_mesh, box_target_mesh):
        """Aligned mesh centroid is near target mesh centroid."""
        aligned, _, _ = scale_and_center_mesh(box_source_mesh, box_target_mesh,
                                               center="centroid")
        np.testing.assert_allclose(aligned.centroid, box_target_mesh.centroid,
                                   atol=0.1)

    def test_empty_source_raises(self, box_target_mesh):
        """Empty source mesh raises ValueError."""
        empty = trimesh.Trimesh()
        with pytest.raises(ValueError):
            scale_and_center_mesh(empty, box_target_mesh)

    def test_empty_target_raises(self, box_source_mesh):
        """Empty target mesh raises ValueError."""
        empty = trimesh.Trimesh()
        with pytest.raises(ValueError):
            scale_and_center_mesh(box_source_mesh, empty)


# ---------------------------------------------------------------------------
# perform_cut_on_target_mesh
# ---------------------------------------------------------------------------

class TestPerformCutOnTargetMesh:
    """Tests for perform_cut_on_target_mesh()."""

    def _make_tall_mesh(self):
        """Create a box mesh extending from z=-5 to z=5."""
        mesh = trimesh.creation.box(extents=[2, 2, 10])
        # box is centered at origin, so z ranges from -5 to 5
        return mesh

    def test_plane_slice_returns_two_parts(self):
        """plane_slice method returns (kept, cut_part) tuple."""
        mesh = self._make_tall_mesh()
        kept, cut_part = perform_cut_on_target_mesh(mesh, 0.0, method="plane_slice")
        assert kept is not None
        assert cut_part is not None

    def test_face_filter_returns_two_parts(self):
        """face_filter method returns (kept, cut_part) tuple."""
        mesh = self._make_tall_mesh()
        kept, cut_part = perform_cut_on_target_mesh(mesh, 0.0, method="face_filter")
        # face_filter may return None for cut_part if no whole faces are above
        assert kept is not None or cut_part is not None

    def test_invalid_method_raises(self):
        """Unknown cutting method raises ValueError."""
        mesh = self._make_tall_mesh()
        with pytest.raises(ValueError, match="Unknown cutting method"):
            perform_cut_on_target_mesh(mesh, 0.0, method="invalid")

    def test_plane_slice_kept_below_cut(self):
        """plane_slice kept part has vertices mostly below cut height."""
        mesh = self._make_tall_mesh()
        kept, _ = perform_cut_on_target_mesh(mesh, 0.0, method="plane_slice")
        if kept is not None and len(kept.vertices) > 0:
            # Most vertices should be at z <= 0 (with some tolerance for new cut vertices)
            assert np.max(kept.vertices[:, 2]) <= 0.1

    def test_face_filter_kept_below_cut(self):
        """face_filter kept part has all vertices at or below cut height."""
        mesh = self._make_tall_mesh()
        kept, _ = perform_cut_on_target_mesh(mesh, 0.0, method="face_filter")
        if kept is not None and len(kept.vertices) > 0:
            assert np.all(kept.vertices[:, 2] <= 0.0 + 1e-10)


# ---------------------------------------------------------------------------
# _create_submesh_from_face_mask
# ---------------------------------------------------------------------------

class TestCreateSubmeshFromFaceMask:
    """Tests for _create_submesh_from_face_mask()."""

    def test_all_faces_selected(self, simple_trimesh):
        """Selecting all faces returns a mesh with same face count."""
        mask = np.ones(len(simple_trimesh.faces), dtype=bool)
        result = _create_submesh_from_face_mask(simple_trimesh, mask)
        assert result is not None
        assert len(result.faces) == len(simple_trimesh.faces)

    def test_no_faces_selected(self, simple_trimesh):
        """Selecting no faces returns None."""
        mask = np.zeros(len(simple_trimesh.faces), dtype=bool)
        result = _create_submesh_from_face_mask(simple_trimesh, mask)
        assert result is None

    def test_partial_selection(self, simple_trimesh):
        """Selecting some faces returns a smaller mesh."""
        mask = np.array([True, True, False, False])
        result = _create_submesh_from_face_mask(simple_trimesh, mask)
        assert result is not None
        assert len(result.faces) == 2

    def test_vertex_reindexing(self, simple_trimesh):
        """Face indices are reindexed to match new vertex array."""
        mask = np.array([True, False, False, False])
        result = _create_submesh_from_face_mask(simple_trimesh, mask)
        if result is not None:
            max_face_idx = result.faces.max()
            assert max_face_idx < len(result.vertices)
