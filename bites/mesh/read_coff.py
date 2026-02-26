"""
COFF (Colored OFF) file loader for trimesh.

Robust handling of:
- Missing or malformed color data
- Triangulation of quads and n-gons
- Removal of duplicate triangles
- Informative warnings and messages
"""

from pathlib import Path
import numpy as np
import trimesh


def read_coff_parts(filename, logger=None) -> tuple:
    """
    Reads a COFF file and returns vertices, faces, and colors as numpy arrays.
    
    Parameters:
    -----------
    filename : str or Path
        Path to the COFF file
    
    Returns:
    --------
    vertices : np.ndarray
        Array of shape (n, 3) with vertex coordinates
    faces : np.ndarray
        Array of shape (m, 3) with triangulated face indices
    colors : np.ndarray or None
        Array of shape (n, 4) with RGBA colors in 0-255 uint8 range, or None if no colors
    """

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    # Convert to Path for consistent handling
    file_path = Path(filename)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip comments and empty lines
    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    
    if not lines:
        raise ValueError(f"File '{file_path}' is empty or contains only comments")
    
    # Check header
    header = lines[0].upper()
    if 'COFF' not in header:
        raise ValueError(f"File '{file_path}' does not appear to be a COFF file (header: {lines[0]})")
    
    # Parse counts
    if len(lines) < 2:
        raise ValueError(f"File '{file_path}' is missing vertex/face count line")
    
    counts = lines[1].split()
    if len(counts) < 2:
        raise ValueError(f"Invalid count line in '{file_path}': expected at least 2 values")
    
    n_vertices = int(counts[0])
    n_faces = int(counts[1])
    
    vertices = []
    colors = []
    has_colors = False
    
    # Parse vertices
    line_idx = 2
    for i in range(n_vertices):
        if line_idx >= len(lines):
            raise ValueError(
                f"File '{file_path}' ended unexpectedly while reading vertices "
                f"(expected {n_vertices}, got {i})"
            )
        
        parts = lines[line_idx].split()
        
        if len(parts) < 3:
            raise ValueError(f"Vertex {i} in '{file_path}' has fewer than 3 coordinates")
        
        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        # Try to extract RGB colors
        if len(parts) >= 6:
            has_colors = True
            r, g, b = float(parts[3]), float(parts[4]), float(parts[5])
            
            # Detect if colors are in 0-1 or 0-255 range and convert to 0-255 uint8
            if r <= 1.0 and g <= 1.0 and b <= 1.0:
                # Colors in 0-1 range, scale to 0-255
                colors.append([int(r * 255), int(g * 255), int(b * 255), 255])
            elif r <= 255.0 and g <= 255.0 and b <= 255.0:
                # Colors already in 0-255 range
                colors.append([int(r), int(g), int(b), 255])
            else:
                #print(
                logger.warning(
                    f"Warning: Vertex {i} in '{file_path}' has color values out of expected range, "
                    f"treating as 0-255"
                )
                colors.append([int(r), int(g), int(b), 255])
        elif len(parts) > 3:
            # Partial color data
            #print(
            logger.warning(
                f"Warning: Vertex {i} in '{file_path}' has {len(parts)-3} color components "
                f"(expected 3 or 0)"
            )
        
        line_idx += 1
    
    if not has_colors:
        #print(f"Info: No color data found in '{file_path}'")
        logger.debug(f"No color data found in '{file_path}'")
    
    # Parse faces
    raw_faces = []
    skipped_faces = 0
    for i in range(n_faces):
        if line_idx >= len(lines):
            raise ValueError(
                f"File '{file_path}' ended unexpectedly while reading faces "
                f"(expected {n_faces}, got {i})"
            )
        
        parts = lines[line_idx].split()
        
        if len(parts) < 1:
            #print(f"Warning: Face {i} in '{file_path}' is empty, skipping")
            logger.debug(f"Face {i} in '{file_path}' is empty, skipping")
            skipped_faces += 1
            line_idx += 1
            continue
        
        n_verts = int(parts[0])
        
        if len(parts) < n_verts + 1:
            #print(
            logger.debug(
                f"Warning: Face {i} in '{file_path}' claims {n_verts} vertices "
                f"but only has {len(parts)-1}, skipping"
            )
            skipped_faces += 1
            line_idx += 1
            continue
        
        face = [int(parts[j+1]) for j in range(n_verts)]
        
        # Validate vertex indices
        if any(v < 0 or v >= n_vertices for v in face):
            #print(f"Warning: Face {i} in '{file_path}' contains invalid vertex indices, skipping")
            logger.debug(f"Face {i} in '{file_path}' contains invalid vertex indices, skipping")
            skipped_faces += 1
            line_idx += 1
            continue
        
        raw_faces.append(face)
        line_idx += 1
    
    if skipped_faces > 0:
        #print(f"Warning: Skipped {skipped_faces} invalid faces in '{file_path}'")
        logger.debug(f"Skipped {skipped_faces} invalid faces in '{file_path}'")
    
    # Triangulate all faces
    triangulated_faces = []
    n_triangles = 0
    n_quads = 0
    n_ngons = 0
    
    for face in raw_faces:
        if len(face) < 3:
            #print(
            logger.debug(
                f"Warning: Found a face with {len(face)} vertices in '{file_path}', "
                f"skipping (need at least 3)"
            )
            continue
        elif len(face) == 3:
            triangulated_faces.append(face)
            n_triangles += 1
        elif len(face) == 4:
            # Split quad into two triangles
            triangulated_faces.append([face[0], face[1], face[2]])
            triangulated_faces.append([face[0], face[2], face[3]])
            n_quads += 1
        else:
            # Fan triangulation for n-gons
            n_ngons += 1
            for i in range(1, len(face) - 1):
                triangulated_faces.append([face[0], face[i], face[i+1]])
    
    if n_quads > 0:
        #print(f"Info: Triangulated {n_quads} quadrilateral faces in '{file_path}'")
        logger.debug(f"Triangulated {n_quads} quadrilateral faces in '{file_path}'")
    if n_ngons > 0:
        #print(
        logger.debug(
            f"Info: Triangulated {n_ngons} polygons with >4 vertices in '{file_path}' "
            f"using fan triangulation"
        )
    
    # Remove duplicate faces
    unique_faces = []
    seen = set()
    duplicates = 0
    
    for face in triangulated_faces:
        # Sort the face indices to detect duplicates regardless of winding order
        face_sorted = tuple(sorted(face))
        if face_sorted not in seen:
            seen.add(face_sorted)
            unique_faces.append(face)
        else:
            duplicates += 1
    
    if duplicates > 0:
        #print(f"Info: Removed {duplicates} duplicate triangular faces from '{file_path}'")
        logger.debug(f"Removed {duplicates} duplicate triangular faces from '{file_path}'")
    
    # Convert to numpy arrays
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(unique_faces, dtype=np.int32)
    colors = np.array(colors, dtype=np.uint8) if has_colors else None
    
    #print(f"Successfully loaded '{file_path}': {len(vertices)} vertices, {len(faces)} triangular faces")
    logger.debug(f"Successfully loaded '{file_path}': {len(vertices)} vertices, {len(faces)} triangular faces")
    
    return vertices, faces, colors


def read_coff_mesh(filename, logger=None) -> trimesh.Trimesh:
    """
    Reads a COFF file and returns a trimesh.Trimesh object.
    
    Parameters:
    -----------
    filename : str or Path
        Path to the COFF file
    
    Returns:
    --------
    mesh : trimesh.Trimesh
        The loaded mesh with vertex colors if available
    """

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    # Convert to Path (read_coff_parts will handle it)
    vertices, faces, colors = read_coff_parts(filename, logger=logger)
    
    # Create trimesh object
    if colors is not None:
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            process=False
        )
    else:
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False
        )
    
    return mesh


# Example usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python coff_loader.py <coff_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        mesh = read_coff_mesh(filename)
        print(f"\nMesh summary:")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Has vertex colors: {mesh.visual.kind == 'vertex'}")
        
        if mesh.visual.kind == 'vertex':
            print(f"  Color shape: {mesh.visual.vertex_colors.shape}")
            print(f"  Color dtype: {mesh.visual.vertex_colors.dtype}")
        
    except Exception as e:
        print(f"Error loading mesh: {e}")
        import traceback
        traceback.print_exc()