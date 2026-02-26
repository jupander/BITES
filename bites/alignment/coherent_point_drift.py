import trimesh
import numpy as np
from pycpd import RigidRegistration

class CPDConvergenceError(Exception):
    pass

def perform_CPD_on_mesh(source, target) -> tuple:
    """
    Aligns the source mesh to the target mesh using the Coherent Point Drift (CPD) algorithm.

    Parameters:
    - source: The source mesh to be aligned. Type: trimesh.Trimesh
    - target: The target mesh to align to. Type: trimesh.Trimesh

    Returns:
    - The aligned source mesh. Type: trimesh.Trimesh
    - The total scaling factor applied.
    - The total rotation matrix applied.
    - The total translation vector applied.
    """

    # Get the vertices of the source and target meshes
    #source_vertices = source.vertices
    #source_vertices = source.vertices.astype(np.float64) # "is only useful if you explicitly need to enforce float64"
    source_vertices = np.array(source.vertices)
    #target_vertices = target.vertices
    target_vertices = np.array(target.vertices)

    # parent class of RigidRegistration: class expectation_maximization_registration(X, Y, max_iterations=100, tolerance=1e-05, w=0.0, verbose=True, print_reg_params=True, *args, **kwargs)

    # Perform rigid registration with scaling
    try:
        reg = RigidRegistration(**{'X': target_vertices, 'Y': source_vertices}, scale=True)
    except Exception as e:
        raise CPDConvergenceError(f"CPD: Failed to initialize RigidRegistration: {e}. Skipping CPD-alignment.")
    try:
        TY, (s, R, t) = reg.register()
    except np.linalg.LinAlgError as e:
        raise CPDConvergenceError(f"CPD: np.linalg.LinAlgError: {e}. Skipping CPD-alignment.")
    except Exception as e:
        raise CPDConvergenceError(f"CPD: Unexpected error: {e}. Skipping CPD-alignment.")

    # Create a new mesh with the aligned vertices
    aligned_mesh = trimesh.Trimesh(
        vertices=TY,
        faces=source.faces,
        vertex_colors=source.visual.vertex_colors if hasattr(source.visual, 'vertex_colors') else None
    )

    return aligned_mesh, s, R, t