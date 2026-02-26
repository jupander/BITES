import logging
from pathlib import Path
import trimesh
import bites.mesh.read_coff as read_coff

logger = logging.getLogger(__name__)

def read_target_mesh_file(file_name) -> trimesh.Trimesh:
    """Read a mesh file using trimesh.

    Natively supports PLY, STL, OBJ, GLB/GLTF, and many other formats
    via trimesh. OFF files use a custom COFF reader to preserve vertex colors.

    Args:
        file_name: Path to the mesh file (str or Path object)

    Returns:
        A trimesh.Trimesh object representing the mesh.
    """

    # Convert to Path object for consistent handling
    file_path = Path(file_name)

    # Get file extension (suffix returns '.ply', so strip the dot)
    file_extension = file_path.suffix.lstrip('.').lower()

    # OFF files use custom COFF reader to preserve vertex colors
    if file_extension == 'off':
        return read_coff.read_coff_mesh(str(file_path))

    # All other formats: delegate to trimesh
    try:
        mesh = trimesh.load(str(file_path), process=False)
        # trimesh.load can return a Scene for multi-mesh files (e.g. GLB)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return mesh
    except Exception as e:
        supported = sorted(trimesh.available_formats())
        logger.error(f"Failed to read mesh file '{file_path.name}': {e}")
        raise ValueError(
            f"Could not load '{file_path.name}' (format: .{file_extension}). "
            f"Supported formats include: {', '.join(supported)}"
        ) from e