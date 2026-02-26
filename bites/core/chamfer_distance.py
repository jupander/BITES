import trimesh
import logging
import numpy as np
import point_cloud_utils as pcu
from typing import Dict, List, Optional


def calculate_chamfer_distances(
    aligned_meshes: Dict[str, trimesh.Trimesh],
    target_meshes: Dict[str, trimesh.Trimesh],
    alignment_algorithms: List[str],
    logger: Optional[logging.Logger] = None,
    num_samples: Optional[int] = None,
    trial_prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate Chamfer distances between aligned and target meshes.

    The Chamfer distance is computed between the vertex point clouds of the meshes.
    Optionally, you can sample points uniformly from the mesh surfaces for more
    accurate distance computation.

    Parameters
    ----------
    aligned_meshes : Dict[str, trimesh.Trimesh]
        Maps alignment algorithm names to aligned trimesh meshes.
    target_meshes : Dict[str, trimesh.Trimesh]
        Maps alignment algorithm names to target trimesh meshes.
    alignment_algorithms : List[str]
        Names of alignment algorithms to evaluate.
    logger : Optional[logging.Logger]
        Logger instance to record messages. If None, uses module logger.
    num_samples : Optional[int]
        If provided, sample this many points uniformly from mesh surfaces
        instead of using vertices. Useful for meshes with different vertex densities.

    Returns
    -------
    Dict[str, float]
        Chamfer distances per algorithm.

    Examples
    --------
    >>> aligned = {"ICP": trimesh.load("aligned.ply")}
    >>> target = {"ICP": trimesh.load("target.ply")}
    >>> distances = calculate_chamfer_distances(aligned, target, ["ICP"])
    >>> print(f"ICP distance: {distances['ICP']:.6f}")
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    chamfer_distances = {}

    for algo in alignment_algorithms:
        # Check if meshes exist for this algorithm
        if algo not in aligned_meshes:
            logger.warning(f"{trial_prefix}Missing aligned mesh for '{algo}'. Skipping.")
            continue
        if algo not in target_meshes:
            logger.warning(f"{trial_prefix}Missing target mesh for '{algo}'. Skipping.")
            continue

        try:
            aligned_mesh = aligned_meshes[algo]
            target_mesh = target_meshes[algo]

            # Validate mesh objects
            if not isinstance(aligned_mesh, trimesh.Trimesh):
                logger.error(f"{trial_prefix}Aligned mesh for '{algo}' is not a trimesh.Trimesh object. Skipping.")
                continue
            if not isinstance(target_mesh, trimesh.Trimesh):
                logger.error(f"{trial_prefix}Target mesh for '{algo}' is not a trimesh.Trimesh object. Skipping.")
                continue

            # Extract or sample point clouds
            if num_samples is not None:
                # Sample points uniformly from mesh surfaces
                pcd_aligned = sample_surface_points(aligned_mesh, num_samples)
                pcd_target = sample_surface_points(target_mesh, num_samples)
            else:
                # Use mesh vertices directly
                pcd_aligned = aligned_mesh.vertices.astype(np.float64)
                pcd_target = target_mesh.vertices.astype(np.float64)

            # Validate point clouds
            if pcd_aligned.shape[0] == 0:
                logger.error(f"{trial_prefix}Aligned mesh for '{algo}' has no vertices. Skipping.")
                continue
            if pcd_target.shape[0] == 0:
                logger.error(f"{trial_prefix}Target mesh for '{algo}' has no vertices. Skipping.")
                continue

            # Compute Chamfer distance
            distance = pcu.chamfer_distance(pcd_aligned, pcd_target)
            chamfer_distances[algo] = float(distance)

            logger.debug(f"{trial_prefix}{algo} Chamfer distance: {distance:.1f}")

        except Exception as e:
            logger.exception(f"{trial_prefix}Failed to compute Chamfer distance for '{algo}': {e}")
            continue

    return chamfer_distances


def sample_surface_points(
    mesh: trimesh.Trimesh,
    num_samples: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Sample points uniformly from the surface of a trimesh mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to sample from.
    num_samples : int
        Number of points to sample.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sampled points with shape (num_samples, 3) in float64.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Trimesh's sample method samples uniformly based on triangle areas
    samples, _ = trimesh.sample.sample_surface(mesh, num_samples)
    return samples.astype(np.float64)