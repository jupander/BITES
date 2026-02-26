"""
Mesh alignment module for tooth geometry processing.

Coordinate System Convention
============================
This module uses a coordinate system where:
- Tooth CUSPS (tips/tops) are at MINIMUM z values (most negative z)
- Tooth ROOTS are at MAXIMUM z values (most positive z)

When visualized with standard settings (positive z pointing down on screen):
- Cusps appear at the TOP of the visualization
- Roots appear at the BOTTOM of the visualization

This means:
- "toward min_z" = toward cusps = visually upward
- "toward max_z" = toward roots = visually downward

When cutting meshes:
- We keep the CUSP SIDE (z < cut_height, smaller z values)
- We discard the ROOT SIDE overshoot (z > cut_height, larger z values)

To avoid confusion, comments in this module use:
- "cusp side" or "z < value" instead of ambiguous "above/below"
- "root side" or "z > value" for the opposite direction
"""

import trimesh
import numpy as np
from pathlib import Path


# Helper functions

def validate_mesh(mesh, name="mesh") -> None:
    """
    Validate that a mesh is properly formed and return diagnostic info.

    Parameters:
    - mesh: The mesh to validate
    - name: Name for error messages

    Returns:
    - dict with diagnostic information (vertices, faces, extents, bounds, centroid)

    Raises:
    - ValueError if mesh is invalid (wrong type, empty, or degenerate)
    """
    info = {
        'name': name,
        'type': type(mesh).__name__,
        'is_trimesh': isinstance(mesh, trimesh.Trimesh),
        'vertices': None,
        'faces': None,
        'extents': None,
        'bounds': None,
        'centroid': None
    }

    try:
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"{name} is not a trimesh.Trimesh object (type: {type(mesh)})")

        info['vertices'] = len(mesh.vertices)
        info['faces'] = len(mesh.faces)

        # is_empty covers both no vertices and no faces
        if mesh.is_empty:
            raise ValueError(f"{name} is empty (no vertices or faces)")

        info['extents'] = mesh.extents
        info['bounds'] = mesh.bounds
        info['centroid'] = mesh.centroid

        if info['extents'] is None or np.any(info['extents'] == 0):
            raise ValueError(f"{name} has zero or invalid extent: {info['extents']}")

        return info

    except Exception as e:
        # Print diagnostic info before re-raising
        print(f"\nMesh validation failed for '{name}':")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print(f"  Error: {e}\n")
        raise


def scale_and_center_mesh(source_mesh, target_mesh, scale_axis='all', center='centroid') -> tuple:
    """
    Scale source mesh to match target's extent and align their centers.

    This function performs:
    1. Scale source to match target's extent (on specified axes)
    2. Align centers (AABB or centroid)

    Parameters:
    - source_mesh: The source mesh to align (will be copied, not modified)
    - target_mesh: The target mesh to align to (not modified)
    - scale_axis: Which axes to use for scaling. Options: 'x', 'y', 'xy', 'all'
    - center: Center calculation method. 'AABB' = bounding box center,
              'centroid' = vertex centroid

    Returns:
    - aligned_mesh: The scaled and centered source mesh (a modified copy)
    - scale_factor: The uniform scaling factor applied
    - translation: The translation vector applied
    """
    # Work on a copy to avoid mutating the input
    aligned_mesh = source_mesh.copy()

    # Validate meshes
    if not isinstance(aligned_mesh, trimesh.Trimesh) or len(aligned_mesh.vertices) == 0:
        raise ValueError("source_mesh must be a valid trimesh.Trimesh with vertices")
    if not isinstance(target_mesh, trimesh.Trimesh) or len(target_mesh.vertices) == 0:
        raise ValueError("target_mesh must be a valid trimesh.Trimesh with vertices")

    source_extents = aligned_mesh.extents
    target_extents = target_mesh.extents

    # Validate extents
    if source_extents is None or target_extents is None:
        raise ValueError("Unable to compute mesh extents. Check that meshes are valid.")
    if np.any(source_extents == 0):
        raise ValueError(f"Source mesh has zero extent in some dimension: {source_extents}")
    if np.any(target_extents == 0):
        raise ValueError(f"Target mesh has zero extent in some dimension: {target_extents}")

    # 1. Scale source to target
    scale_factors = target_extents / source_extents

    # Map scale_axis options to the corresponding scaling factor
    scale_axis_dict = {
        'x': scale_factors[0],
        'y': scale_factors[1],
        'xy': np.mean(scale_factors[:2]),
        'all': np.mean(scale_factors)
    }

    if scale_axis not in scale_axis_dict:
        raise ValueError(f"Invalid scale_axis: {scale_axis}. Choose 'x', 'y', 'xy', or 'all'.")

    scale_factor = scale_axis_dict[scale_axis]

    # Get center before scaling
    if center == 'AABB':
        center_of_mesh = aligned_mesh.bounding_box.centroid
    elif center == 'centroid':
        center_of_mesh = aligned_mesh.centroid
    else:
        raise ValueError(f"Invalid center option: {center}. Choose 'AABB' or 'centroid'.")

    # Apply scaling around the chosen center
    aligned_mesh.apply_scale(scale_factor)

    # Restore center position after scaling
    if center == 'AABB':
        aligned_mesh.apply_translation(center_of_mesh - aligned_mesh.bounding_box.centroid)
    else:
        aligned_mesh.apply_translation(center_of_mesh - aligned_mesh.centroid)

    # 2. Align centers
    if center == 'AABB':
        translation = target_mesh.bounding_box.centroid - aligned_mesh.bounding_box.centroid
    else:  # centroid
        translation = target_mesh.centroid - aligned_mesh.centroid

    aligned_mesh.apply_translation(translation)

    return aligned_mesh, scale_factor, translation


def ST_align_tops_and_cut(source_mesh, target_mesh, scale_axis='xy', center='AABB', min_cut_height=None, override_cut_height=None) -> tuple:
    """
    Scale, translate, and align source mesh to target, optionally cutting target overshoot.

    This function performs a complete alignment pipeline:
    1. Scale source to match target's extent (on specified axes)
    2. Align centers (AABB or centroid)
    3. Align min_z faces (tooth cusps)
    4. Optionally cut target mesh if it extends past source toward max_z (root side)

    See module docstring for coordinate system conventions.

    Parameters:
    - source_mesh: The source mesh to align (will be copied, not modified)
    - target_mesh: The target mesh to align to (not modified)
    - scale_axis: Which axes to use for scaling. Options: 'x', 'y', 'xy', 'all'
    - center: Center calculation method. 'AABB' = bounding box center,
              'centroid' = vertex centroid
    - min_cut_height: If provided, enables cutting. Acts as lower bound for cut
                      z-position (won't cut at z < min_cut_height). If None, no cutting.
    - override_cut_height: Force cut at this z-height (for testing). Overrides
                           automatic cut height calculation.

    Returns:
    - source_mesh: The aligned source mesh (a modified copy)
    - total_scale: The uniform scaling factor applied
    - total_translation: The total translation vector applied
    - cropped_target_mesh: Target with root-side overshoot cut off (or original if no cut)
    - cut_part_of_target_mesh: The cut-off root-side overshoot (or None if no cut)
    - used_cut_height: The z-height where cut occurred (or None if no cut)
    """
    # 1 & 2. Scale and center the source mesh
    source_mesh, scale_factor, sc_translation = scale_and_center_mesh(
        source_mesh, target_mesh, scale_axis=scale_axis, center=center
    )

    # 3. Align min_z faces (cusps) and optionally cut target's root-side overshoot
    source_mesh, cropped_target_mesh, cut_part_of_target_mesh, atac_translation, used_cut_height = align_tops_and_cut(
        source_mesh, target_mesh, min_cut_height=min_cut_height, override_cut_height=override_cut_height
    )

    # Calculate the total translation
    total_translation = sc_translation + atac_translation

    return source_mesh, scale_factor, total_translation, cropped_target_mesh, cut_part_of_target_mesh, used_cut_height


def determine_cut_height_to_use(source_bounds, target_bounds, min_cut_height=None, override_cut_height=None) -> float:
    """
    Determine the z-coordinate at which to cut the target mesh.

    By default, cuts at source's max_z (root-side boundary). The min_cut_height
    parameter prevents cutting too far toward the cusps (too small z values).

    Parameters:
    - source_bounds: Bounds of the source mesh [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    - target_bounds: Bounds of the target mesh
    - min_cut_height: Lower bound for cut position (won't cut at z < min_cut_height)
    - override_cut_height: If specified, use this z-value instead of calculating

    Returns:
    - The z-coordinate at which to cut
    """
    if override_cut_height is not None:
        return override_cut_height

    # Default: cut at source's max_z (where source mesh ends on root side)
    cut_height_to_use = source_bounds[1, 2]

    # Clamp: don't cut further toward cusps than min_cut_height allows
    if min_cut_height is not None and cut_height_to_use < min_cut_height:
        cut_height_to_use = min_cut_height

    return cut_height_to_use


def perform_cut_on_target_mesh(target_mesh, cut_height_to_use, method='plane_slice') -> tuple:
    """
    Cut the target mesh at the specified z-height into two parts.

    Parameters:
    - target_mesh: The mesh to cut (not modified)
    - cut_height_to_use: The z-coordinate at which to cut
    - method: Cutting method to use:
        - 'plane_slice': Slice along plane, creating new edges at cut boundary.
                         Produces clean cut with new vertices on the cut plane.
        - 'face_filter': Keep/discard whole faces based on vertex positions.
                         Produces rugged edge following original mesh topology.
                         Faces crossing the boundary are discarded.

    Returns:
    - cropped_target_mesh: The CUSP SIDE (z < cut_height), kept for alignment
    - cut_part_of_target_mesh: The ROOT SIDE (z > cut_height), the discarded overshoot

    Note on trimesh.slice_plane (used by 'plane_slice' method):
        Returns the portion of mesh on the side the normal points toward.
        - Normal [0, 0, -1] (toward min_z) → keeps vertices where z < plane_z
        - Normal [0, 0, 1] (toward max_z) → keeps vertices where z > plane_z
    """
    if method == 'plane_slice':
        return _cut_by_plane_slice(target_mesh, cut_height_to_use)
    elif method == 'face_filter':
        return _cut_by_face_filter(target_mesh, cut_height_to_use)
    else:
        raise ValueError(f"Unknown cutting method: {method}. Use 'plane_slice' or 'face_filter'.")


def _cut_by_plane_slice(mesh, cut_height) -> tuple:
    """
    Cut mesh using plane slicing (creates new geometry at cut boundary).

    This method intersects the mesh with a horizontal plane, creating new
    vertices and edges where the plane crosses existing geometry. Results
    in a clean cut boundary.

    Parameters:
    - mesh: The mesh to cut
    - cut_height: The z-coordinate at which to cut

    Returns:
    - kept_mesh: Portion with z < cut_height (cusp side)
    - cut_mesh: Portion with z > cut_height (root side, discarded)
    """
    # Keep the CUSP SIDE: vertices with z < cut_height (toward min_z / cusps)
    kept_mesh = mesh.slice_plane(
        plane_origin=[0, 0, cut_height],
        plane_normal=[0, 0, -1],  # Points toward min_z → keeps z < cut_height
        cap=False # No need to cap the cut (would create a watertight mesh)
    )

    # Keep the ROOT SIDE: vertices with z > cut_height (the overshoot to discard)
    cut_mesh = mesh.slice_plane(
        plane_origin=[0, 0, cut_height],
        plane_normal=[0, 0, 1],  # Points toward max_z → keeps z > cut_height
        cap=False
    )

    return kept_mesh, cut_mesh


def _cut_by_face_filter(mesh, cut_height) -> tuple:
    """
    Cut mesh by filtering whole faces based on vertex positions.

    This method keeps or discards entire faces based on whether ALL their
    vertices are on one side of the cut height. Faces that straddle the
    boundary are discarded entirely. Results in a rugged edge that follows
    the original mesh topology.

    Parameters:
    - mesh: The mesh to cut
    - cut_height: The z-coordinate threshold

    Returns:
    - kept_mesh: Faces with ALL vertices at z <= cut_height (cusp side)
    - cut_mesh: Faces with ALL vertices at z > cut_height (root side)

    Note: Faces crossing the boundary (some vertices above, some below)
    are not included in either output mesh.
    """
    cut_height = float(cut_height)

    # Get vertex z-coordinates for each face
    vertex_z = mesh.vertices[:, 2]
    face_vertex_z = vertex_z[mesh.faces]  # Shape: (n_faces, 3)

    # Classify faces by whether ALL vertices are on one side
    kept_mask = np.all(face_vertex_z <= cut_height, axis=1)  # All vertices on cusp side
    cut_mask = np.all(face_vertex_z > cut_height, axis=1)    # All vertices on root side

    # Create submeshes
    kept_mesh = _create_submesh_from_face_mask(mesh, kept_mask)
    cut_mesh = _create_submesh_from_face_mask(mesh, cut_mask)

    return kept_mesh, cut_mesh


def _create_submesh_from_face_mask(mesh, face_mask) -> trimesh.Trimesh:
    """
    Create a new mesh from selected faces with proper vertex reindexing.

    Parameters:
    - mesh: Original mesh
    - face_mask: Boolean array indicating which faces to keep

    Returns:
    - trimesh.Trimesh or None if no valid faces
    """
    if not np.any(face_mask):
        return None

    selected_faces = mesh.faces[face_mask]

    if len(selected_faces) == 0:
        return None

    # Find unique vertices used by selected faces
    unique_vertex_ids = np.unique(selected_faces.flatten())

    if len(unique_vertex_ids) < 3:
        return None

    # Extract vertices and create old→new index mapping
    new_vertices = mesh.vertices[unique_vertex_ids]
    vertex_map = np.full(len(mesh.vertices), -1, dtype=np.int32)
    vertex_map[unique_vertex_ids] = np.arange(len(unique_vertex_ids))

    # Remap face indices to new vertex indices
    new_faces = vertex_map[selected_faces]

    if np.any(new_faces < 0):
        # This shouldn't happen if face_mask is consistent
        return None

    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)


def align_tops_and_cut(source_mesh, target_mesh, min_cut_height=None, override_cut_height=None) -> tuple:
    """
    Translate source mesh along z-axis to align cusp faces, optionally cutting target.

    Aligns the min_z faces (tooth cusps) of both meshes. If the target extends
    further toward max_z (root direction) than the source, the excess is cut off.

    Parameters:
    - source_mesh: The source mesh to be aligned (will be copied, not modified)
    - target_mesh: The target mesh to align to (not modified)
    - min_cut_height: If provided, enables cutting. The cut position won't go
                      below this z-value (toward cusps). If None, no cutting occurs.
    - override_cut_height: If provided, forces cut at this exact z-height (for testing)

    Returns:
    - source_mesh: The aligned source mesh (a modified copy)
    - cropped_target_mesh: Target with root-side overshoot removed (or original if no cut)
    - cut_part_of_target_mesh: The removed root-side overshoot (or None if no cut)
    - atac_translation: The translation vector applied [0, 0, z_translation]
    - used_cut_height: The z-height where cut was made (or None if no cut)
    """
    # Work on a copy to avoid mutating the input
    source_mesh = source_mesh.copy()

    # Get bounding boxes
    source_bounds = source_mesh.bounds
    target_bounds = target_mesh.bounds

    # Translate source along z-axis to align min_z faces (cusp tips)
    z_translation = target_bounds[0, 2] - source_bounds[0, 2]
    source_mesh.apply_translation([0, 0, z_translation])
    source_bounds = source_mesh.bounds  # Recompute since the mesh has been translated

    # Cut target if it extends past source on the root side (larger max_z)
    if target_bounds[1, 2] > source_bounds[1, 2] and min_cut_height is not None:
        cut_height_to_use = determine_cut_height_to_use(source_bounds, target_bounds, min_cut_height=min_cut_height, override_cut_height=override_cut_height)
        cropped_target_mesh, cut_part_of_target_mesh = perform_cut_on_target_mesh(target_mesh, cut_height_to_use)
    else:
        # No cut needed, return the original target mesh
        cut_height_to_use = None
        cropped_target_mesh = target_mesh
        cut_part_of_target_mesh = None
        
    # calculate the total translation
    atac_translation = np.array([0, 0, z_translation])

    return source_mesh, cropped_target_mesh, cut_part_of_target_mesh, atac_translation, cut_height_to_use


# =============================================================================
# Demo / Test Script
# =============================================================================
#
# This script demonstrates the mesh alignment and cutting functionality.
# Run with: python -m bites.alignment.statac
#
# Tests:
#   1. Alignment test: Aligns source to target using different scale axes
#   2. Cut height test: Demonstrates cutting at various z-heights
#
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    # -------------------------------------------------------------------------
    # Visualization helper
    # -------------------------------------------------------------------------

    def visualize_meshes(source=None, target=None, cut_part=None, title="Mesh Viewer") -> None:
        """
        Display meshes in a VisPy viewer with distinct colors.

        Colors:
        - Source (aligned): Orange
        - Target (cropped): Blue
        - Cut part (discarded): Red
        """
        from vispy import app
        from bites.gui.visualization.viewers.vispy_mesh_viewer import create_standalone_viewer

        COLORS = {
            'source': ([1.0, 0.706, 0.0], "orange"),
            'target': ([0.0, 0.651, 0.929], "blue"),
            'cut': ([1.0, 0.0, 0.0], "red"),
        }

        def colorize(mesh, color_rgb) -> trimesh.Trimesh:
            """Apply uniform vertex color to mesh."""
            colored = mesh.copy()
            color_uint8 = (np.array(color_rgb) * 255).astype(np.uint8)
            colored.visual.vertex_colors = np.tile(color_uint8, (len(colored.vertices), 1))
            return colored

        meshes_to_display = []
        for mesh, key in [(source, 'source'), (target, 'target'), (cut_part, 'cut')]:
            if mesh is not None and not mesh.is_empty:
                color_rgb, _ = COLORS[key]
                meshes_to_display.append((colorize(mesh, color_rgb), color_rgb, key))

        if not meshes_to_display:
            print(f"  [Warning] No meshes to display for: {title}")
            return

        viewer = create_standalone_viewer(meshes=meshes_to_display, title=title)
        viewer.show()
        app.run()

    # -------------------------------------------------------------------------
    # Test functions
    # -------------------------------------------------------------------------

    def run_alignment_test(source_mesh, target_mesh, min_cut_height) -> None:
        """Test alignment with different scaling axis options."""
        print("\n" + "=" * 60)
        print("ALIGNMENT TEST")
        print("=" * 60)

        for scale_axis in ['x', 'y', 'xy', 'all']:
            print(f"\n  Testing scale_axis='{scale_axis}'...")

            aligned, scale, translation, cropped, cut_part, cut_height = ST_align_tops_and_cut(
                source_mesh, target_mesh,
                scale_axis=scale_axis,
                min_cut_height=min_cut_height
            )

            print(f"    Scale factor: {scale:.4f}")
            print(f"    Translation: {translation}")
            print(f"    Cut height: {cut_height}")

            visualize_meshes(
                source=aligned,
                target=cropped,
                cut_part=cut_part,
                title=f"Alignment Test - scale_axis='{scale_axis}'"
            )

    def run_cut_height_test(source_mesh, target_mesh, min_cut_height) -> None:
        """Test cutting at various z-heights to visualize the cut behavior."""
        print("\n" + "=" * 60)
        print("CUT HEIGHT TEST")
        print("=" * 60)

        # Generate test cut heights as fractions of target mesh height
        target_max_z = target_mesh.bounds[1, 2]
        target_min_z = target_mesh.bounds[0, 2]
        z_range = target_max_z - target_min_z

        # Test at 25%, 50%, 75% of the mesh height
        fractions = [0.25, 0.5, 0.75]
        cut_heights = [target_min_z + z_range * f for f in fractions]

        print(f"\n  Target mesh z-range: [{target_min_z:.2f}, {target_max_z:.2f}]")
        print(f"  Min cut height constraint: {min_cut_height}")

        for i, cut_height in enumerate(cut_heights):
            fraction = fractions[i]
            print(f"\n  Testing cut at z={cut_height:.2f} ({fraction:.0%} of height)...")

            aligned, scale, translation, cropped, cut_part, used_height = ST_align_tops_and_cut(
                source_mesh, target_mesh,
                scale_axis='xy',
                min_cut_height=min_cut_height,
                override_cut_height=cut_height
            )

            print(f"    Requested cut: {cut_height:.2f}")
            print(f"    Actual cut: {used_height}")

            # Check if cut produced valid meshes
            cropped_valid = cropped is not None and not cropped.is_empty
            cut_valid = cut_part is not None and not cut_part.is_empty
            print(f"    Cropped mesh valid: {cropped_valid}")
            print(f"    Cut part valid: {cut_valid}")

            visualize_meshes(
                source=aligned,
                target=cropped,
                cut_part=cut_part,
                title=f"Cut Test - z={cut_height:.2f} ({fraction:.0%})"
            )

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Demo script for ST_align_tops_and_cut mesh alignment",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument(
            "--source", "-s",
            type=Path,
            default=Path("./assets/example_source.ply"),
            help="Path to source mesh file"
        )
        parser.add_argument(
            "--target", "-t",
            type=Path,
            default=Path("./assets/benchmark_target.ply"),
            help="Path to target mesh file"
        )
        parser.add_argument(
            "--min-cut-height",
            type=float,
            default=None,
            help="Minimum cut height constraint (enables cutting)"
        )
        parser.add_argument(
            "--test",
            choices=["alignment", "cut", "all"],
            default="all",
            help="Which test to run"
        )
        args = parser.parse_args()

        # Validate paths
        for path, name in [(args.source, "Source"), (args.target, "Target")]:
            if not path.exists():
                print(f"Error: {name} mesh not found at: {path}")
                sys.exit(1)

        # Load meshes
        print(f"Loading meshes...")
        print(f"  Source: {args.source}")
        print(f"  Target: {args.target}")

        try:
            source_mesh = trimesh.load(args.source, process=False)
            target_mesh = trimesh.load(args.target, process=False)
            validate_mesh(source_mesh, name="source")
            validate_mesh(target_mesh, name="target")
        except Exception as e:
            print(f"\nError loading meshes: {e}")
            sys.exit(1)

        print(f"  Source: {len(source_mesh.vertices)} vertices, {len(source_mesh.faces)} faces")
        print(f"  Target: {len(target_mesh.vertices)} vertices, {len(target_mesh.faces)} faces")

        # Run tests
        if args.test in ("alignment", "all"):
            run_alignment_test(source_mesh, target_mesh, args.min_cut_height)

        if args.test in ("cut", "all"):
            run_cut_height_test(source_mesh, target_mesh, args.min_cut_height)

        print("\n" + "=" * 60)
        print("Demo complete.")
        print("=" * 60)

    main()