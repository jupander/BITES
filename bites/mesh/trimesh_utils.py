import logging
import numpy as np
import trimesh
from typing import Optional, Tuple, List, Union

logger = logging.getLogger(__name__)

# ============================================================================
# Utility Functions
# ============================================================================

def compute_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """
    Compute unique edges from triangle faces.
    
    Parameters:
    -----------
    faces : ndarray
        Face indices (N, 2) for edges or (N, 3) for triangles
    
    Returns:
    --------
    edges : ndarray
        Unique edge indices (M, 2)
    
    Raises:
    -------
    ValueError
        If faces array has invalid shape
    """
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] not in (2, 3):
        raise ValueError("faces must be an (N,2) or (N,3) array")
    
    edges_set = set()
    
    if faces.shape[1] == 2:
        # Already edge pairs: canonicalize and make unique
        for a, b in faces:
            edge = tuple(sorted((int(a), int(b))))
            edges_set.add(edge)
    else:
        # Triangle faces: extract 3 edges per face
        for face in faces:
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            edges_set.add(tuple(sorted((a, b))))
            edges_set.add(tuple(sorted((b, c))))
            edges_set.add(tuple(sorted((c, a))))
    
    if not edges_set:
        return np.zeros((0, 2), dtype=int)
    
    return np.array(list(edges_set), dtype=int)


def validate_trimesh(mesh: trimesh.Trimesh, operation: str = "operation") -> bool:
    """
    Validate that a trimesh object has required geometry.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        Mesh to validate
    operation : str
        Description of operation for error messages
    
    Returns:
    --------
    bool : True if valid, False otherwise
    """
    if not isinstance(mesh, trimesh.Trimesh):
        logger.error(f"Invalid mesh type for {operation}")
        return False

    if len(mesh.faces) == 0:
        logger.error(f"Mesh has no triangles for {operation}")
        return False

    if len(mesh.vertices) == 0:
        logger.error(f"Mesh has no vertices for {operation}")
        return False
    
    return True


def convert_color_to_rgba(color: Union[str, Tuple, List], 
                          default_alpha: float = 1.0) -> Tuple[float, ...]:
    """
    Convert color to RGBA tuple format.
    
    Parameters:
    -----------
    color : str, tuple, or list
        Color specification (RGB or RGBA)
    default_alpha : float
        Alpha value to use if not provided
    
    Returns:
    --------
    tuple : RGBA color (r, g, b, a) in 0-1 range
    """
    if isinstance(color, str):
        return color  # Let VisPy handle string colors
    
    color = tuple(color) if not isinstance(color, tuple) else color
    
    if len(color) == 3:
        return (*color, default_alpha)
    elif len(color) == 4:
        return color
    else:
        raise ValueError(f"Invalid color format: {color}")