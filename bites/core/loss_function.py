
import optuna
import trimesh
import numpy as np
from pathlib import Path

import bites.alignment.coherent_point_drift as coherent_point_drift
import bites.alignment.statac as statac
import bites.core.chamfer_distance as chamfer_distance
import bites.core.output_manager as output_manager


def align_meshes(config, source_mesh, og_target_mesh, min_cut_height=None, trial_number=None) -> tuple:
    """
    Aligns the source mesh to the target mesh using various alignment algorithms. 
    If some algorithms fail, they are skipped and left out of returned dicts.

    Parameters:
    -----------
    config : object
        Configuration object with alignment settings and logger
    source_mesh : trimesh.Trimesh
        Source mesh to align
    og_target_mesh : trimesh.Trimesh
        Target mesh to align to
    min_cut_height : float, optional
        Minimum height for cutting operations

    Returns:
    --------
    aligned_meshes : dict
        Dictionary of aligned meshes for each algorithm
    scale_factors : dict
        Dictionary of scale factors for each algorithm
    rotation_matrices : dict
        Dictionary of rotation matrices for each algorithm
    translation_vectors : dict
        Dictionary of translation vectors for each algorithm
    target_meshes : dict
        Dictionary of target meshes for each algorithm (if applicable)
    used_cut_heights : dict
        Dictionary of used cut heights for each algorithm (if applicable)
    """

    alignment_algorithms = config.alignment_algorithms
    logger = config.logger
    trial_prefix = f"[Trial {trial_number}] " if trial_number is not None else ""

    aligned_meshes = {}
    scale_factors = {}
    rotation_matrices = {}
    translation_vectors = {}
    target_meshes = {}
    used_cut_heights = {}  # added 2025-05-28

    for algo in alignment_algorithms:
        # reset the variables for each algorithm
        aligned_mesh = scale_factor = rotation_matrix = translation_vector = used_cut_height = None
        target_mesh = og_target_mesh  # default unless replaced

        try:

            if algo == 'ST':
                aligned_mesh, scale_factor, translation_vector = statac.scale_and_center_mesh(source_mesh, target_mesh, scale_axis='all', center='centroid')
                rotation_matrix = np.eye(3)  # identity matrix, since the mesh is only scaled and translated

            elif algo == 'CPD-only':
                aligned_mesh, scale_factor, rotation_matrix, translation_vector = \
                    coherent_point_drift.perform_CPD_on_mesh(source_mesh, target_mesh)  

            elif algo == 'ST-CPD':
                # first align with ST, then with CPD
                ST_aligned_mesh, ST_scale_factor, ST_translation_vector = \
                    statac.scale_and_center_mesh(source_mesh, target_mesh, scale_axis='all', center='centroid')
                aligned_mesh, scale_factor, rotation_matrix, translation_vector = \
                    coherent_point_drift.perform_CPD_on_mesh(ST_aligned_mesh, target_mesh)  
                # combine the translation vectors and scale factors
                translation_vector = translation_vector + ST_translation_vector
                scale_factor = scale_factor * ST_scale_factor
            
            elif algo == 'STatac-AB':  # Scale and Translate, then Align the Tops And Cut the target mesh 
                                       # (AB = using AABB center, the original version)
                aligned_mesh, scale_factor, translation_vector, cut_target_mesh, cut_part_of_target_mesh, used_cut_height = \
                    statac.ST_align_tops_and_cut(source_mesh, target_mesh, scale_axis='xy', 
                                                 center='AABB', min_cut_height=min_cut_height)
                rotation_matrix = np.eye(3)  # identity matrix, since the mesh is only scaled and translated
                target_mesh = cut_target_mesh  # the original target mesh is replaced with the cut one
            
            elif algo == 'STatac-C':  # (C = using centroid, a newer alternative version - more similar to 
                                      # the widely used Procrustes analysis, might be better with asymmetric meshes)
                aligned_mesh, scale_factor, translation_vector, cut_target_mesh, cut_part_of_target_mesh, used_cut_height = \
                    statac.ST_align_tops_and_cut(source_mesh, target_mesh, scale_axis='xy', 
                                                 center='centroid', min_cut_height=min_cut_height)
                rotation_matrix = np.eye(3)  # identity matrix, since the mesh is only scaled and translated
                target_mesh = cut_target_mesh  # the original target mesh is replaced with the cut one

            elif algo == 'STatac-AB-CPD':
                # first align with STatac, then with CPD
                STatac_aligned_mesh, STatac_scale_factor, STatac_translation_vector, cut_target_mesh, \
                    cut_part_of_target_mesh, used_cut_height = \
                    statac.ST_align_tops_and_cut(source_mesh, target_mesh, scale_axis='xy', 
                                                 center='AABB', min_cut_height=min_cut_height)
                aligned_mesh, scale_factor, rotation_matrix, translation_vector = \
                    coherent_point_drift.perform_CPD_on_mesh(STatac_aligned_mesh, cut_target_mesh)  
                # combine the translation vectors and scale factors
                translation_vector = translation_vector + STatac_translation_vector
                scale_factor = scale_factor * STatac_scale_factor
                target_mesh = cut_target_mesh  # the original target mesh is replaced with the cut one

            elif algo == 'STatac-C-CPD':
                # first align with STatac, then with CPD
                STatac_aligned_mesh, STatac_scale_factor, STatac_translation_vector, cut_target_mesh, \
                    cut_part_of_target_mesh, used_cut_height = \
                    statac.ST_align_tops_and_cut(source_mesh, target_mesh, scale_axis='xy', 
                                                 center='centroid', min_cut_height=min_cut_height)
                aligned_mesh, scale_factor, rotation_matrix, translation_vector = \
                    coherent_point_drift.perform_CPD_on_mesh(STatac_aligned_mesh, cut_target_mesh)  
                # combine the translation vectors and scale factors
                translation_vector = translation_vector + STatac_translation_vector
                scale_factor = scale_factor * STatac_scale_factor
                target_mesh = cut_target_mesh  # the original target mesh is replaced with the cut one

            else:
                logger.warning(f"{trial_prefix}align_meshes: Unknown alignment algorithm: {algo} - skipping it")
                continue
            
            # Validate all outputs are non-None
            if any(x is None for x in (aligned_mesh, scale_factor, rotation_matrix, 
                                      translation_vector, target_mesh)):
                # get the names of the variables that are None
                var_names = [name for name, value in zip(
                    ['aligned_mesh', 'scale_factor', 'rotation_matrix', 'translation_vector', 'target_mesh'],
                    [aligned_mesh, scale_factor, rotation_matrix, translation_vector, target_mesh]
                ) if value is None]
                logger.error(f"{trial_prefix}[{algo}] Alignment output contains None for variables {var_names}. "
                           f"Skipping this algorithm.")
                continue

            aligned_meshes[algo] = aligned_mesh
            scale_factors[algo] = scale_factor
            rotation_matrices[algo] = rotation_matrix
            translation_vectors[algo] = translation_vector
            target_meshes[algo] = target_mesh
            used_cut_heights[algo] = used_cut_height

        except Exception as e:
            logger.error(f"{trial_prefix}Alignment algo '{algo}' failed: {e}")
            continue

    return aligned_meshes, scale_factors, rotation_matrices, translation_vectors, target_meshes, used_cut_heights


def evaluate_loss(config, mesh_by_iter_dict: dict, tm_run_identifier: str, results_folder_path: Path, trial_number=None) -> tuple:
    """
    Evaluates loss values for the given source meshes.

    Parameters:
    -----------
    config : object
        Configuration object with settings and logger
    mesh_by_iter_dict : dict
        Dictionary mapping iteration numbers to trimesh.Trimesh objects
    tm_run_identifier : str
        Run identifier for output naming
    results_folder_path : Path
        Path to folder for saving results

    Returns:
    --------
    loss : float
        Best chamfer distance achieved
    properties_of_chosen_mesh : dict
        Properties of the mesh with the lowest loss
    properties_of_other_meshes_by_iter : dict
        Properties of all evaluated meshes
    """

    # ------- Get settings from the config -------
    target_shape_path = config.target_shape_path  # e.g. 'example_ringed_seal.off'
    alignment_algorithms = config.alignment_algorithms
    logger = config.logger
    trial_prefix = f"[Trial {trial_number}] " if trial_number is not None else ""

    # Handle min_cut_height
    try:
        min_cut_height = float(config.min_cut_height)
    except (ValueError, TypeError):  # i.e. min_cut_height is None, 'None', or something else
        logger.debug(f"Value of config.min_cut_height ({config.min_cut_height}) could not be cast to float - "
                    f"setting it as None and (thus) proceeding without cutting.")
        min_cut_height = None

    # ------- Load the target mesh -------
    target_mesh = trimesh.load(str(target_shape_path), process=False)
    
    if len(target_mesh.faces) == 0:
        logger.error(f"{trial_prefix}Failed to load target mesh from {target_shape_path} - no faces found")
        raise optuna.TrialPruned("Failed evaluation (Target mesh has no faces).")

    # ------- Initialize the dictionaries for saving the results -------
    # All of these will use the iter value as the key
    best_chamfer_distance_by_iter_dict = {}  # the lowest chamfer distance (among the alignment algos) for each iteration
    best_alignment_algorithm_by_iter_dict = {}  # the alignment algorithm that resulted in the lowest chamfer distance
    
    # The following dicts are for each iteration, and for each alignment algorithm
    # i.e. {iter_value: {alignment_algorithm: value}}
    scale_factors_by_iter_dict = {}
    rotation_matrices_by_iter_dict = {} 
    translation_vectors_by_iter_dict = {}
    target_meshes_by_iter_dict = {}
    chamfer_distances_by_iter_dict = {}
    used_cut_heights_by_iter_dict = {}  # added 2025-06-10

    aligned_meshes_by_iter_dict = {}  # for saving (some of) the aligned meshes to files later

    # Track failure reasons for diagnostic purposes
    failure_reasons_by_iter = {}

    # ------- Pre-check for invalid meshes -------
    for iter_value, mesh in mesh_by_iter_dict.items():
        if mesh is None:
            logger.warning(f"{trial_prefix}Mesh at iter {iter_value} is None - will be skipped")
            failure_reasons_by_iter[iter_value] = "mesh_is_none"
        elif not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            logger.warning(f"{trial_prefix}Mesh at iter {iter_value} has no faces - will be skipped")
            failure_reasons_by_iter[iter_value] = "mesh_has_no_faces"

    for iter_value, source_mesh in mesh_by_iter_dict.items():
        # Skip meshes that failed pre-check
        if iter_value in failure_reasons_by_iter:
            continue

        # ------- Align the mesh -------
        try:    
            aligned_meshes, scale_factors, rotation_matrices, translation_vectors, target_meshes, used_cut_heights = \
                align_meshes(config, source_mesh, target_mesh, min_cut_height, trial_number=trial_number)
            # NOTE in the above dicts, some values may be missing if some alignment algorithm (CPD) failed to converge
        except Exception as e:
            logger.error(f"{trial_prefix}Unexpected error while aligning meshes at iter {iter_value}: {e}")
            failure_reasons_by_iter[iter_value] = f"alignment_failed: {type(e).__name__}: {e}"
            aligned_meshes, scale_factors, rotation_matrices, translation_vectors, target_meshes, used_cut_heights = \
                None, None, None, None, None, None
            continue

        # ------- Calculate the chamfer distances -------
        try:
            chamfer_distances = chamfer_distance.calculate_chamfer_distances(
                aligned_meshes, target_meshes, alignment_algorithms, logger,
                trial_prefix=trial_prefix)
        except Exception as e:
            logger.error(f"{trial_prefix}Unexpected error while calculating chamfer distances at iter {iter_value}: {e}")
            failure_reasons_by_iter[iter_value] = f"chamfer_failed: {type(e).__name__}: {e}"
            chamfer_distances = None
            continue

        # ------- Choose the best alignment algorithm -------
        try:
            best_alignment_algorithm, best_chamfer_distance = min(
                chamfer_distances.items(), key=lambda item: item[1])
        except ValueError:
            logger.debug(f"{trial_prefix}Could not determine best alignment algo for iter {iter_value} - "
                        f"no valid distances calculated.")
            failure_reasons_by_iter[iter_value] = "no_valid_chamfer_distances"
            best_alignment_algorithm = None
            best_chamfer_distance = None

        # Add the results to the dictionaries
        chamfer_distances_by_iter_dict[iter_value] = chamfer_distances
        best_alignment_algorithm_by_iter_dict[iter_value] = best_alignment_algorithm
        best_chamfer_distance_by_iter_dict[iter_value] = best_chamfer_distance

        aligned_meshes_by_iter_dict[iter_value] = aligned_meshes
        scale_factors_by_iter_dict[iter_value] = scale_factors
        rotation_matrices_by_iter_dict[iter_value] = rotation_matrices
        translation_vectors_by_iter_dict[iter_value] = translation_vectors
        target_meshes_by_iter_dict[iter_value] = target_meshes
        used_cut_heights_by_iter_dict[iter_value] = used_cut_heights

    # ------- Pick the best iter value based on the lowest chamfer distance -------
    if best_chamfer_distance_by_iter_dict:
        # Filter out the meshes with None values (i.e. the alignment algorithm failed to converge,
        # or the chamfer distance could not be calculated)
        best_chamfer_distance_by_iter_dict = {
            k: v for k, v in best_chamfer_distance_by_iter_dict.items() if v is not None
        }
        if not best_chamfer_distance_by_iter_dict:  # if all the values were None, the dict is empty
            # Build diagnostic message with failure reasons
            if failure_reasons_by_iter:
                failure_summary = "; ".join(f"iter {k}: {v}" for k, v in sorted(failure_reasons_by_iter.items()))
                error_msg = f"All chamfer distances are None. Failures by iter: [{failure_summary}]"
            else:
                error_msg = "All chamfer distances are None (no specific failure reasons recorded)"
            logger.warning(error_msg)
            raise optuna.TrialPruned(error_msg)
    else:  # the dict is empty
        # Build diagnostic message with failure reasons
        if failure_reasons_by_iter:
            failure_summary = "; ".join(f"iter {k}: {v}" for k, v in sorted(failure_reasons_by_iter.items()))
            error_msg = f"No chamfer distances calculated. Failures by iter: [{failure_summary}]"
        else:
            error_msg = "No chamfer distances calculated (no meshes processed successfully)"
        logger.warning(error_msg)
        raise optuna.TrialPruned(error_msg)

    try:
        iter_of_chosen_mesh = min(best_chamfer_distance_by_iter_dict, 
                                 key=best_chamfer_distance_by_iter_dict.get)
    except Exception as e:  # broad catch just in case
        logger.error(f"{trial_prefix}Unexpected error when trying to get the best iter value: {e} - returning None as loss.")
        return None, None, None

    mesh_identifiers_by_iter_dict = output_manager.write_aligned_meshes_to_files(
        config,
        aligned_meshes_by_iter_dict,
        target_meshes_by_iter_dict,
        best_chamfer_distance_by_iter_dict,
        iter_of_chosen_mesh,
        tm_run_identifier,
        results_folder_path,
        used_cut_heights_by_iter_dict[iter_of_chosen_mesh])

    # ------- Prepare the return values based on the chosen mesh -------
    loss = best_chamfer_distance_by_iter_dict[iter_of_chosen_mesh]

    properties_of_chosen_mesh = {
        'iter_of_chosen_mesh': iter_of_chosen_mesh,  # int, the number of iterations of the chosen mesh
        'best_alignment_algorithm': best_alignment_algorithm_by_iter_dict[iter_of_chosen_mesh],  # str
        'scale_factors': scale_factors_by_iter_dict[iter_of_chosen_mesh],  # dict, type: float
        'rotation_matrices': rotation_matrices_by_iter_dict[iter_of_chosen_mesh],  # dict, type: np.ndarray
        'translation_vectors': translation_vectors_by_iter_dict[iter_of_chosen_mesh],  # dict, type: np.ndarray
        'chamfer_distances': chamfer_distances_by_iter_dict[iter_of_chosen_mesh],  # dict, type: float
        'used_cut_heights': used_cut_heights_by_iter_dict[iter_of_chosen_mesh],  # dict, type: float
    }
    
    try:
        properties_of_chosen_mesh['mesh_identifier'] = mesh_identifiers_by_iter_dict[iter_of_chosen_mesh]
    except KeyError:
        logger.warning(f"{trial_prefix}KeyError: Could not get the mesh identifier for the chosen mesh ({iter_of_chosen_mesh}) - "
                      f"setting it to 'None'")
        properties_of_chosen_mesh['mesh_identifier'] = 'None'

    # Store all the unused info in an additional dictionary
    properties_of_other_meshes_by_iter = {
        'best_chamfer_distance_by_iter_dict': best_chamfer_distance_by_iter_dict,
        'best_alignment_algorithm_by_iter_dict': best_alignment_algorithm_by_iter_dict,
        'scale_factors_by_iter_dict': scale_factors_by_iter_dict,
        'rotation_matrices_by_iter_dict': rotation_matrices_by_iter_dict,
        'translation_vectors_by_iter_dict': translation_vectors_by_iter_dict,
        'chamfer_distances_by_iter_dict': chamfer_distances_by_iter_dict,
        'used_cut_heights_by_iter_dict': used_cut_heights_by_iter_dict,
    }

    return loss, properties_of_chosen_mesh, properties_of_other_meshes_by_iter