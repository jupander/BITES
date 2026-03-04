#!/usr/bin/env python

"""
Mesh simplifier core - Pure mesh logic, configuration, and utilities.

This module contains the non-GUI components of the mesh simplifier:
- Mesh loading and simplification algorithms
- Configuration dataclass and validation
- Constants and enumerations
- IPC writer for communicating results to parent processes
"""

from __future__ import annotations
import logging
import tempfile
import trimesh
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# IPC Writer for communicating saved mesh path back to parent process
# ============================================================================

class SimplifierIPCWriter:
    """Handles writing simplified mesh path to IPC file for parent process."""

    IPC_FILENAME = 'bites_simplifier_result.txt'

    @classmethod
    def get_ipc_path(cls) -> str:
        return str(Path(tempfile.gettempdir()) / cls.IPC_FILENAME)

    @classmethod
    def write_result(cls, output_path: str) -> bool:
        """
        Write the simplified mesh path to IPC file.

        Args:
            output_path: Path where the simplified mesh was saved

        Returns:
            True if write succeeded, False otherwise
        """
        ipc_file = cls.get_ipc_path()
        try:
            with open(ipc_file, 'w') as f:
                f.write(output_path)
            logger.info(f"Simplified mesh path saved to IPC file: {ipc_file}")
            return True
        except Exception as e:
            logger.error(f"Error writing simplified mesh path to IPC file: {e}")
            return False


# ============================================================================
# Constants
# ============================================================================

class BenchmarkAlgorithm(Enum):
    """Available benchmark algorithms."""
    CPD = "CPD"
    CHAMFER_DISTANCE = "CD"


# Default configuration values
DEFAULT_MIN_EPSILON = 0.0
DEFAULT_MAX_EPSILON = 20.0
DEFAULT_N_SELECTABLE = 11      # Number of candidate meshes user can select from
DEFAULT_N_PLOT_SAMPLES = 200   # Number of samples for the plot curve (each is a simplified mesh)
DEFAULT_WINDOW_SIZE = (600, 400)
DEFAULT_VISUALIZER_SIZE = (400, 400)
DEFAULT_VISUALIZER_POSITION = (650, 50)
VISUALIZER_UPDATE_INTERVAL_MS = 30
WINDOW_ACTIVATION_DELAY = 0.05  # seconds

# Progress bar phases
# Plotting always takes 10%, the rest is divided between simplification and benchmarking
PROGRESS_PLOTTING = 10
PROGRESS_SIMPLIFICATION_ONLY = 90  # When no benchmark
PROGRESS_SIMPLIFICATION_WITH_BENCH = 45  # When benchmark is enabled
PROGRESS_BENCHMARKING = 45

# File filters
MESH_FILE_FILTER = "Mesh Files (*.ply *.off *.stl *.obj *.glb *.gltf *.3mf)"
PLY_FILE_FILTER = "PLY Files (*.ply);;All Files (*)"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SimplificationConfig:
    """
    Configuration for mesh simplification.

    Attributes:
    -----------
    min_epsilon : float
        Minimum simplification threshold (0 = original mesh)
    max_epsilon : float
        Maximum simplification threshold (higher = more aggressive)
    n_selectable : int
        Number of candidate meshes the user can select from via slider
    n_plot_samples : int
        Total number of samples for the plot curve (each requires mesh simplification)
    """
    min_epsilon: float
    max_epsilon: float
    n_selectable: int
    n_plot_samples: int

    def validate(self) -> Tuple[bool, str]:
        """
        Validate configuration parameters.

        Returns:
        --------
        tuple
            (is_valid, error_message) where error_message is empty if valid
        """
        # Check for NaN or infinity
        if not np.isfinite(self.min_epsilon) or not np.isfinite(self.max_epsilon):
            return False, "Epsilon values must be finite numbers."

        if self.min_epsilon < 0 or self.max_epsilon < 0:
            return False, "Epsilon values cannot be negative."

        if self.min_epsilon >= self.max_epsilon:
            return False, "Min epsilon must be less than max epsilon."

        if not isinstance(self.n_selectable, int) or self.n_selectable < 2:
            return False, "Number of selectable meshes must be an integer >= 2."

        if not isinstance(self.n_plot_samples, int) or self.n_plot_samples < 2:
            return False, "Number of plot samples must be an integer >= 2."

        return True, ""

    def get_effective_n_selectable(self) -> int:
        """Get the effective number of selectable meshes (clamped to n_plot_samples if needed)."""
        return min(self.n_selectable, self.n_plot_samples)


# ============================================================================
# Utility Functions
# ============================================================================

def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a mesh from file with proper normal computation.

    Parameters:
    -----------
    path : str
        Path to mesh file

    Returns:
    --------
    trimesh.Trimesh
        Loaded mesh with computed normals
    """
    mesh = trimesh.load(path, force='mesh')

    # Ensure it's a Trimesh object (not Scene)
    if isinstance(mesh, trimesh.Scene):
        # If it's a scene, concatenate all geometries
        mesh = trimesh.util.concatenate(
            [geom for geom in mesh.geometry.values()
             if isinstance(geom, trimesh.Trimesh)]
        )

    # Ensure vertex normals are computed (accessing the property triggers computation)
    _ = mesh.vertex_normals

    return mesh


def simplify_mesh(mesh: trimesh.Trimesh,
                 epsilon: float) -> trimesh.Trimesh:
    """
    Simplify a mesh by merging close vertices.

    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Input mesh
    epsilon : float
        Distance threshold for merging vertices

    Returns:
    --------
    trimesh.Trimesh
        Simplified mesh (never returns None values for vertices/faces)
    """
    # Return empty mesh with proper arrays (not None) for edge cases
    def empty_mesh() -> None:
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64)
        )

    if epsilon <= 0:
        return mesh.copy()

    # Handle empty or invalid input mesh
    if mesh.vertices is None or mesh.faces is None:
        return empty_mesh()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return empty_mesh()

    from scipy.spatial import cKDTree

    mesh_copy = mesh.copy()
    vertices = mesh_copy.vertices
    faces = mesh_copy.faces

    # Build KDTree for efficient nearest neighbor search
    tree = cKDTree(vertices)

    # Find groups of vertices within epsilon distance
    groups = tree.query_ball_tree(tree, epsilon)

    # Create mapping from old vertex indices to new vertex indices
    vertex_map = np.arange(len(vertices))
    visited = np.zeros(len(vertices), dtype=bool)

    for i, group in enumerate(groups):
        if not visited[i]:
            # Mark all vertices in group as visited and map to first vertex
            group_indices = [j for j in group if j >= i]  # Only consider j >= i to avoid duplicates
            for j in group_indices:
                vertex_map[j] = i
                visited[j] = True

    # Get unique vertices and remap faces
    unique_indices = np.unique(vertex_map)
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}

    new_vertices = vertices[unique_indices]
    new_faces = np.array([[index_mapping[vertex_map[v]] for v in face] for face in faces])

    # Remove degenerate faces (where any two vertices are the same)
    valid_faces = []
    for face in new_faces:
        if face[0] != face[1] and face[1] != face[2] and face[0] != face[2]:
            valid_faces.append(face)

    if len(valid_faces) == 0:
        # Return empty mesh with proper arrays (not None)
        return empty_mesh()

    new_faces = np.array(valid_faces)

    # Create new mesh with merged vertices
    simplified = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False  # We already cleaned up manually
    )

    return simplified
