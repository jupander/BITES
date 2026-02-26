# non-package_version/main.py for local development and testing
# copy contents to the package version (meta_learner/main.py) for deployment

import csv
import time
import optuna
import shutil
import argparse
import warnings
import traceback
import tracemalloc
import pandas as pd
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Tuple
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler #, CmaEsSampler, RandomSampler,

# (package version imports:)
from bites.config.config_loader import Config
from bites.core.logging_config import setup_logger
from bites.database.pg_db_manager import ensure_database_for_target, _sanitize_url_for_logging
from bites.paths import get_logs_folder, get_results_folder, ensure_folder_exists

import bites.core.run_tooth_model as run_tooth_model
import bites.core.loss_function as loss_function
import bites.core.output_manager as output_manager
import bites.core.trial_export as trial_export
from bites.database.pg_export import export_postgres_database

# ----------------- Global settings -----------------

# Suppress Optuna's default trial logging (we use a custom callback instead)
import logging
logging.getLogger("optuna").setLevel(logging.WARNING)

# ----------------- Helper functions -----------------

def construct_storage_path(config, results_folder_path: Path = None, study_name: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine the Optuna storage path for the meta-study based on the config.

    Tries PostgreSQL first when storage_mode is 'shared'. If PostgreSQL fails,
    automatically falls back to a local SQLite database in the output folder.

    Args:
        config: The optimization config object
        results_folder_path: Path to output folder (used for SQLite fallback)
        study_name: Name of the study (used for SQLite filename)

    Returns:
        tuple: (storage_url, storage_type) where storage_type is 'postgres' or 'sqlite'
    """
    logger = config.logger

    if config.storage_mode == "shared":
        # Try PostgreSQL first
        try:
            POSTGRES_URL = ensure_database_for_target(config)
            if not POSTGRES_URL:
                raise RuntimeError("PostgreSQL connection returned None.")
            logger.info("Successfully connected to PostgreSQL database.")
            return POSTGRES_URL, "postgres"
        except Exception as e:
            # Make the fallback notice prominent with spacing and borders
            print(f"\n{'='*60}")
            print(f"!!  WARNING: PostgreSQL setup failed")
            print(f"    {e}")
            print(f"{'='*60}")
            logger.warning(f"PostgreSQL setup failed: {e}")
            logger.info("Falling back to local SQLite database...")

            # Fall back to SQLite
            if results_folder_path is None or study_name is None:
                logger.error("Cannot create SQLite fallback: results_folder_path or study_name not provided.")
                return None, None

            # Create SQLite database in output folder
            sqlite_filename = f"{study_name}.db"
            sqlite_path = results_folder_path / sqlite_filename
            sqlite_url = f"sqlite:///{sqlite_path}"

            logger.info(f"Using local SQLite storage: {sqlite_path}")
            logger.warning("NOTE: SQLite storage is local-only. For distributed/parallel optimization "
                          "across multiple machines, fix PostgreSQL configuration.")
            return sqlite_url, "sqlite"

    else:
        raise ValueError(f"Invalid storage mode: {config.storage_mode}. Must be 'shared' (other options not implemented).")


def memory_monitor_callback(logger, step=100) -> Callable:
    """
    Optuna callback to monitor and log memory usage every 'step' trials.
    """
    def callback(study, trial):
        if trial.number % step == 0:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            total = sum(stat.size for stat in top_stats)
            top_5_lines = "\n".join(str(stat) for stat in top_stats[:5])
            logger.info(f"\n\nMemory checkup at Trial {trial.number} - Total: {total / 1024 / 1024:.2f} MB\n"
                       f"Top 5 lines using up memory:\n{top_5_lines}")
    return callback


def save_trial_callback(path: str) -> Callable:
    """
    Optuna callback to save each completed trial incrementally
    in the same format as save_trials_dataframe().
    """
    # Define fixed column structure (copied from save_trials_dataframe)
    fixed_columns = trial_export.get_fixed_columns()

    def callback(study, trial):
        row = trial_export.extract_trial_data(trial)
        df = pd.DataFrame([row])

        # Fill missing columns
        for col in fixed_columns:
            if col not in df.columns:
                df[col] = None
        # Maintain column order (existing + new)
        all_columns = fixed_columns + [c for c in df.columns if c not in fixed_columns]

        file_exists = Path(path).exists()
        df.to_csv(path, mode='a', header=not file_exists, index=False, sep=";", columns=all_columns,
                  quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')

    return callback


def rounded_params_logging_callback(logger, significant_digits: int = 3) -> Callable:
    """
    Optuna callback to log trial results with rounded parameter values.
    Replaces Optuna's default verbose trial logging with cleaner output.

    Args:
        logger: Logger instance for output
        significant_digits: Number of significant digits for float values (default: 3)
    """
    def callback(study, trial):
        if trial.value is None:
            return

        # Round float parameters to specified significant digits
        format_str = f"{{:.{significant_digits}g}}"
        rounded_params = {
            k: float(format_str.format(v)) if isinstance(v, float) else v
            for k, v in trial.params.items()
        }

        # Build the log message
        best_info = ""
        if study.best_trial is not None:
            best_info = f" Best of study: trial {study.best_trial.number}, loss {study.best_value:.2f}."

        logger.info(f"Trial {trial.number} finished, loss: {trial.value:.2f}.{best_info}")

    return callback


def build_optimization_callbacks(logger, intermediate_csv_path: str, enable_memory_monitor: bool = False, memory_monitor_step: int = 100) -> list:
    """
    Build the list of Optuna callbacks for study.optimize().

    Args:
        logger: Logger instance for callbacks that need it.
        intermediate_csv_path: Path to the intermediate CSV file for incremental trial saving.
        enable_memory_monitor: Whether to include the memory monitoring callback (requires tracemalloc).
        memory_monitor_step: Log memory usage every N trials (only used if enable_memory_monitor is True).

    Returns:
        List of Optuna callback functions.
    """
    callbacks = [
        save_trial_callback(intermediate_csv_path),
        rounded_params_logging_callback(logger, significant_digits=3),
    ]
    if enable_memory_monitor:
        callbacks.insert(0, memory_monitor_callback(logger, step=memory_monitor_step))
    return callbacks


def construct_trial_parameters_dict(trial, config) -> dict:
    """
    Combine fixed and optimized parameters into a single dictionary, 
    also taking into account special cases such as bias parameters and Rad.
    (Replaces the function combine_param_values_and_defaults_into_array in the hyperopt version.)

    Returns a dictionary of parameters for the trial.
    """
    
    # Sample values for the optimized parameters
    sampled_params = {
        p: trial.suggest_int(p, max(int(bounds[0]), 2), int(bounds[1]), step=1) if p == "Rad" # Rad is an int param
        else trial.suggest_float(p, bounds[0], bounds[1], step=0.0000001) # all other params are floats (with maximum precision of 1e-7, while default would be 1e-17)
        for p, bounds in config.params_optimization_ranges.items() if p in config.params_to_optimize # NOTE does not include bias params when they are fixed to another bias param
    }

    # Combine fixed and sampled parameters into a single dictionary
    params_combined = {k: config.params_fixed_values.get(k, None) for k in config.param_names_all if k not in config.params_to_optimize}
    params_combined.update(sampled_params)

    # Special case: bias parameters
    # Extract necessary settings from the config 
    Abi_value_for_Pbi_too = config.bias_param_config.Abi_value_for_Pbi_too
    Abi_Pbi_factor = config.bias_param_config.Abi_Pbi_factor 
    Bbi_value_for_Lbi_too = config.bias_param_config.Bbi_value_for_Lbi_too 
    Bbi_Lbi_factor = config.bias_param_config.Bbi_Lbi_factor 

    # NOTE e.g. Abi_value_for_Pbi_too should never be True if Abi is not in the params_to_optimize (and thus in sampled_params)
    if Abi_value_for_Pbi_too:
        params_combined["Pbi"] = (params_combined["Abi"] * Abi_Pbi_factor)
    if Bbi_value_for_Lbi_too:
        params_combined["Lbi"] = (params_combined["Bbi"] * Bbi_Lbi_factor)

    # make iter_max visible to Optuna
    iter_max = config.max_iterations
    params_combined["iter_max"] = trial.suggest_int("iter_max", iter_max, iter_max, step=1) # NOTE fake range, value stays constant between trials

    # make tm_timeout visible to Optuna 
    tm_timeout = config.tm_timeout
    params_combined["tm_timeout"] = trial.suggest_int("tm_timeout", tm_timeout, tm_timeout, step=1) # NOTE fake range, value stays constant between trials

    return params_combined


def make_objective(config: Config, config_path: str, results_folder_path: Path) -> Callable:
    """
    Create an objective function for Optuna based on the provided config.
    """
    def objective(trial):

        logger = config.logger

        try:
            trial_parameters_dict = construct_trial_parameters_dict(trial, config)
        except Exception as e:
            logger.error(f"Error in function construct_trial_parameters_dict: {e}", exc_info=True)
            trial.set_user_attr("prune_reason", f"Parameter construction failed: {e}")
            raise optuna.TrialPruned(f"Parameter construction failed: {e}")

        # ----------------------------------------------------
        # ------------------ Run simulation ------------------
        # ----------------------------------------------------

        tm_output_path = results_folder_path / config.tm_subfolder_name

        try:
            #process, tm_run_duration, tm_run_identifier = run_tooth_model.run_tooth_model_as_subprocess(config, trial_parameters_dict, logger)
            tm_run_complete, tm_run_duration, tm_run_start_time, tm_run_identifier = run_tooth_model.run_tooth_model_as_subprocess(
                                                            model_executable_path = config.model_executable_path, 
                                                            trial_parameters_dict = trial_parameters_dict, 
                                                            tm_timeout = config.tm_timeout, # NOTE this could possibly be omitted, as it is now included in trial_parameters_dict (would break gui_benchmark_model.py tho, maybe keep as optional arg?)
                                                            iter_interval = config.iter_interval, 
                                                            tm_output_path = tm_output_path,
                                                            logger = logger,
                                                            trial_number = trial.number,
                                                            )
        except Exception as e:
            logger.error(f"Error running tooth model: {e}", exc_info=True)
            trial.set_user_attr("prune_reason", f"Tooth model execution failed: {e}")
            raise optuna.TrialPruned(f"Tooth model execution failed: {e}")
            
        # ------------------------------------------------------------------
        # Gather the meshes generated by the model into a dictionary. 

        # Alternative A: Parse the COFF data from the standard output string into mesh(es) (requires the tooth_model process to be successful(?))
        # (this is not used currently, but kept just in case, as it could be useful in some situations) 
        # mesh_by_iter_dict = construct_generated_meshes_by_iter_dict_from_stdout(iter_interval, n_intervals)
        
        # Alternative B: Just load the output COFF files from the folder (this would allow using the files potentially obtained so far, even if the tooth_model process had a timeout)
        mesh_by_iter_dict = output_manager.gather_mesh_by_iter_dict(tm_run_identifier, tm_output_path, logger) # get the meshes generated by tooth model (if any)
        # mesh_by_iter_dict is a dictionary with an iter value as the key and the mesh as the value

        if not mesh_by_iter_dict:
            #logger.warning(f"No meshes generated by the tooth model for run identifier: {tm_run_identifier}. Returning None as loss.")
            #return None # no meshes to evaluate, return None as loss
            trial.set_user_attr("prune_reason", "No meshes generated by the tooth model.")
            trial.set_user_attr("tm_run_identifier", tm_run_identifier)
            raise optuna.TrialPruned("No meshes generated by the tooth model.")

        top_iter_reached = max(mesh_by_iter_dict.keys()) # get the maximum iteration value reached by the tooth model (this is the last mesh generated by the model)
        # debug print:
        #print(f"Top iter reached: {top_iter_reached} (run identifier: {tm_run_identifier})\nMeshes generated for iter values: {list(mesh_by_iter_dict.keys())}")

        # ------- Filter out the meshes with iter values lower than min_iterations-that-we-are-interested-in-------
        #min_iterations = config.iter_range_of_interest[0] # e.g. 3000
        min_iterations = config.min_iterations # e.g. 3000
        mesh_by_iter_dict = {k: v for k, v in mesh_by_iter_dict.items() if k >= min_iterations} # only consider the meshes that have enough iterations

        # ----------------------------------------------------
        # Construct another dictionary with the simulation duration for each mesh (of different iter values)
        duration_by_iter_dict = output_manager.calculate_sim_duration_by_iter(
                    tm_run_identifier = tm_run_identifier,
                    tm_output_path = tm_output_path, 
                    start_time = tm_run_start_time
                    )

        # ----------------------------------------------------
        # ------------------ Evaluate loss -------------------
        # ----------------------------------------------------

        # debug print
        logger.debug(f"Evaluating loss for trial {trial.number} with meshes for iter values: {list(mesh_by_iter_dict.keys())}")

        if not mesh_by_iter_dict:
            #logger.warning(f"No meshes generated by the tooth model for run identifier: {tm_run_identifier}.")
            # cleanup temporary files (normally done at the end of the objective function, but here we need to do it before pruning the trial)
            try:
                output_manager.remove_temporary_files(tm_run_identifier, tm_output_path, config) # remove the temporary files created by the tooth model run (if any)
            except Exception as e:
                logger.error(f"Error removing temporary files: {e}")
            # prune the trial
            trial.set_user_attr("prune_reason", f"No meshes with min. iter ({min_iterations}) generated.")
            trial.set_user_attr("tm_run_identifier", tm_run_identifier)
            raise optuna.TrialPruned(f"No meshes with min. iter ({min_iterations}) generated.")
        else:
            try:
                start_evaluation_time = time.time()
                loss, properties_of_chosen_mesh, properties_of_other_meshes_by_iter = loss_function.evaluate_loss(config, mesh_by_iter_dict, tm_run_identifier, results_folder_path, trial_number=trial.number)
                evaluate_loss_duration = round(time.time() - start_evaluation_time, 2) # seconds
            except optuna.TrialPruned as e:
                # TrialPruned is expected behavior (e.g., all chamfer distances None) - not an error
                logger.debug(f"Trial pruned during loss evaluation: {e}")
                trial.set_user_attr("prune_reason", str(e))
                trial.set_user_attr("tm_run_identifier", tm_run_identifier)
                # Clean up temporary files before re-raising
                try:
                    output_manager.remove_temporary_files(tm_run_identifier, tm_output_path, config)
                except Exception as cleanup_e:
                    logger.error(f"Error removing temporary files: {cleanup_e}")
                raise  # Re-raise the original TrialPruned with its message intact
            except Exception as e:
                logger.error(f"Unexpected error during loss evaluation: {e}")
                logger.error(traceback.format_exc()) # print the full traceback for debugging
                # Record why trial was pruned before raising
                trial.set_user_attr("prune_reason", f"Loss evaluation failed: {e}")
                trial.set_user_attr("tm_run_identifier", tm_run_identifier)
                # Clean up temporary files before pruning
                try:
                    output_manager.remove_temporary_files(tm_run_identifier, tm_output_path, config)
                except Exception as cleanup_e:
                    logger.error(f"Error removing temporary files: {cleanup_e}")
                raise optuna.TrialPruned(f"Loss evaluation failed: {e}")

        # ----------------------------------------------------
        # ------- Save trial data to the trial object --------
        # ----------------------------------------------------

        # -----------------------------------
        # 1. Properties of the trial

        trial.set_user_attr("trial_parameters_dict", trial_parameters_dict)
        trial.set_user_attr("tm_run_identifier", tm_run_identifier) # str, included in the name of the tooth_model output files (e.g. 'run240522-124945945abc')
        # NOTE ^ this is the identifier of the tooth_model run, not the trial (trial number is in the trial object already)
        #trial.set_user_attr("evaluate_loss_duration", evaluate_loss_duration)
        try:
            trial.set_user_attr("evaluate_loss_duration", evaluate_loss_duration)
        except Exception as e:
            logger.error(f"Error setting evaluate_loss_duration attribute: {e}")
        trial.set_user_attr("tm_run_duration", tm_run_duration)
        trial.set_user_attr("top_iter_reached", top_iter_reached) # int, the maximum iteration value reached by the tooth model (this is the last mesh generated by the model)
        trial.set_user_attr("duration_by_iter_dict", duration_by_iter_dict)

        # ----------
        # Manually select some properties to save (note that dicts properties_of_chosen_mesh and properties_of_other_meshes_by_iter have different structures) 
        if properties_of_chosen_mesh and properties_of_other_meshes_by_iter: # should both exist or both be None
            trial.set_user_attr("chosen_mesh_identifier", properties_of_chosen_mesh.get("mesh_identifier", None)) # str, the identifier of the chosen mesh (e.g. 'run240522-124945945abc__7000')
            iter_of_chosen_mesh = properties_of_chosen_mesh.get("iter_of_chosen_mesh", None) # also used later to add "best_algorithm" column
            trial.set_user_attr("iter_of_chosen_mesh", iter_of_chosen_mesh)
            trial.set_user_attr("chosen_mesh_duration", duration_by_iter_dict.get(iter_of_chosen_mesh, None)) # time elapsed until the chosen mesh was generated

            trial.set_user_attr("best_chamfer_distance_per_iter", properties_of_other_meshes_by_iter.get("best_chamfer_distance_by_iter_dict", None))

            #trial.set_user_attr("best_alignment_algorithm_per_iter", properties_of_other_meshes_by_iter.get("best_alignment_algorithm_by_iter_dict", None)) 
            best_alignment_algorithm_by_iter_dict = properties_of_other_meshes_by_iter.get("best_alignment_algorithm_by_iter_dict", None)
            trial.set_user_attr("best_alignment_algorithm_per_iter", best_alignment_algorithm_by_iter_dict)
            # add best_algorithm column to comply with the old visualization script
            if best_alignment_algorithm_by_iter_dict:
                trial.set_user_attr("best_algorithm", best_alignment_algorithm_by_iter_dict.get(iter_of_chosen_mesh, None))
            
            chamfer_distances_all = properties_of_other_meshes_by_iter.get("chamfer_distances_by_iter_dict", None)
            if chamfer_distances_all:
                chamfer_distances_all_flattened = output_manager.flatten_dict(chamfer_distances_all) # flatten the nested dict to avoid JSON serialization errors
                if chamfer_distances_all_flattened:
                    trial.set_user_attr("chamfer_distances_all", chamfer_distances_all_flattened) # dict, the chamfer distances for all alignment algorithms (e.g. {'ST': 0.123, 'ICP': 0.456}) for all iter versions

                chamfer_distances = chamfer_distances_all.get(iter_of_chosen_mesh, None) # get the chamfer distances for the chosen mesh (i.e. the one with the iter value of interest)
                if chamfer_distances:
                    for algorithm, distance in chamfer_distances.items():
                        trial.set_user_attr(algorithm, distance) # save the chamfer distance for each alignment algorithm

            # Save cut heights for algorithms that cut the target mesh (e.g., STatac variants)
            used_cut_heights = properties_of_chosen_mesh.get("used_cut_heights", None)
            if used_cut_heights:
                trial.set_user_attr("used_cut_heights", used_cut_heights)  # dict, e.g. {"STatac": 6.0, "STatac-CPD": 5.5}

        else:
            logger.debug(f"properties_of_other_meshes_by_iter and/or properties_of_other_meshes_by_iter is None or empty (run identifier: {tm_run_identifier}).")
            pass

        # ----------
        trial.set_user_attr("tm_run_complete", tm_run_complete) 

        # -----------------------------------
        # 2. Properties of the optimization run or config

        #trial.set_user_attr("iter_range", config.iter_range_of_interest)
        trial.set_user_attr("iter_range", [config.min_iterations, config.max_iterations]) # e.g. [3000, 7000]
        trial.set_user_attr("tm_run_timeout", config.tm_timeout) # int, the timeout for the tooth model run in seconds
        trial.set_user_attr("params_to_optimize", config.params_to_optimize) 
        trial.set_user_attr("alignment_algorithms", config.alignment_algorithms)
        trial.set_user_attr("run_config", Path(config_path).stem)

        # -----------------------------------
        # 3. Individual parameter values # TODO maybe alter the visualization script(s) to get these values from the trial_parameters_dict instead? To reduce redundancy and number of columns in the CSV file
        
        for param_name in config.param_names_all:
            if param_name in trial_parameters_dict:
                trial.set_user_attr(param_name, trial_parameters_dict[param_name])
            else:
                trial.set_user_attr(param_name, None)
                logger.warning(f"Parameter {param_name} not found in trial_parameters_dict. Setting to None. (run identifier: {tm_run_identifier})") # should not happen, but just in case


        # -----------------------------------------------------------
        # Clean up the temporary files, if not wanted for later analysis
        # -----------------------------------------------------------

        try:
            output_manager.remove_temporary_files(tm_run_identifier, tm_output_path, config) # remove the temporary files created by the tooth model run (if any)
        except Exception as e:
            logger.error(f"Error removing temporary files: {e}")
        
        return loss
    
    return objective


def _create_tpe_sampler(config: Config) -> TPESampler:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*multivariate.*", category=ExperimentalWarning) # Suppress the specific warning about multivariate TPE being experimental NOTE
        return TPESampler(
            multivariate=config.multivariate, # "The multivariate TPE is reported to outperform the independent TPE."
            warn_independent_sampling=False, # suppress warnings when not enough prior data for multivariate TPE NOTE
        )


def run_study(config_path, config: Config, results_folder_path: Path, optimization_run_id: str) -> Optional[optuna.Study]:

    logger = config.logger

    try:
        # ----------------- Setup database and storage -----------------
        study_name = config.study_name

        try:
            storage_url, storage_type = construct_storage_path(
                config,
                results_folder_path=results_folder_path,
                study_name=study_name
            )
            config.storage_URL = storage_url
            config.storage_type = storage_type  # Track which backend we're using
        except Exception as e:
            logger.error(f"Error constructing storage path: {e}")
            return None

        if not config.storage_URL:
            logger.error("No storage URL available. Cannot proceed without database storage.")
            logger.error("Please check your PostgreSQL configuration or ensure output folder is writable.")
            return None

        study = optuna.create_study(
            storage=config.storage_URL,
            study_name=study_name,
            direction="minimize",
            load_if_exists=True,
            # Sampler docs: "If None is specified, TPESampler is used during single-objective optimization
            # and NSGAIISampler during multi-objective optimization."
            sampler=_create_tpe_sampler(config),
        )

        objective = make_objective(config, config_path, results_folder_path)

        total_cores = multiprocessing.cpu_count()
        parallel_jobs = min(config.parallel_trials, total_cores)

        # Limit parallelism for SQLite to avoid database lock issues
        if config.storage_type == "sqlite" and parallel_jobs > 1:
            logger.warning(f"SQLite storage: limiting parallel jobs from {parallel_jobs} to 1 to avoid lock contention")
            parallel_jobs = 1

        # run timeout text info
        run_timeout_text = "no timeout" if config.study_timeout <= 0 else f"{config.study_timeout} hours"
        
        logger.info(f"\n\nRunning config: {config_path}" +
                    f"\nStudy name: {study_name}" +
                    f"\nTarget shape filename: {Path(config.target_shape_path).name}" +
                    f"\nTarget shape min cut height: {config.min_cut_height}" +
                    f"\nParameters to optimize ({len(config.params_to_optimize)}): {config.params_to_optimize}" +
                    f"\nIteration range of interest: [{config.min_iterations}, {config.max_iterations}]" +
                    f"\nTooth model timeout: {config.tm_timeout} seconds ({config.tm_timeout / 60:.1f} minutes)" +
                    f"\nMax trials to run: {config.max_trials}" +
                    f"\nParallel trials: {parallel_jobs} / {total_cores} cores" +
                    #f"\nMultivariate: {config.multivariate}" +
                    f"\nAlignment algorithms: {config.alignment_algorithms}" +
                    #f"\nStorage ({config.storage_type}): {_sanitize_url_for_logging(config.storage_URL)}" +
                    f"\n\nEstimated max duration for max trials: {config.max_trials * config.tm_timeout / config.parallel_trials / 3600:.2f} hours" +
                    f"\nOptimization run timeout: {run_timeout_text}" +
                    f"\n\n--- Starting optimization... ---\n"
                    )

        # Log pre-existing trial count so the console window can distinguish
        # trials from previous runs vs. trials from this run
        pre_existing_trials = len(study.trials)
        if pre_existing_trials > 0:
            logger.info(f"Pre-existing trials in study: {pre_existing_trials}")

        # extra CSV to output intermediate results on trial by trial basis
        intermediate_csv_path = results_folder_path / f"ongoing_{optimization_run_id}.csv" # NOTE used to be f"{optimization_run_id}_intermediate.csv", but adding text between run_id and .csv confuses the shape results visualization script

        # Build the callback list (memory monitor is opt-in since tracemalloc adds overhead)
        enable_memory_monitor = getattr(config, 'enable_memory_monitor', False)
        callbacks = build_optimization_callbacks(
            logger=logger,
            intermediate_csv_path=intermediate_csv_path,
            enable_memory_monitor=enable_memory_monitor,
        )

        # Run the optimization
        try:
            study.optimize(
                objective,
                n_trials=config.max_trials,
                n_jobs=parallel_jobs,
                timeout=config.study_timeout*3600 if config.study_timeout > 0 else None, # convert hours to seconds
                gc_after_trial=True, # "Flag to determine whether to automatically run garbage collection after each trial."
                # "Set to True to run the garbage collection, False otherwise. When it runs, it runs a full collection by internally calling gc.collect(). If you see an increase in memory consumption over several trials, try setting this flag to True."
                catch=(Exception,),  # Continue the study even if individual trials raise exceptions
                callbacks=callbacks,
                # "Each function must accept two parameters with the following types in this order: Study and FrozenTrial."
                show_progress_bar=False # "To show progress bar, set this True. Note that it is disabled when n_trials is None, timeout is not None, and n_jobs > 1"
            )
        except KeyboardInterrupt:
            logger.info("\n\n--- Optimization interrupted by user. ---\n")
            #study.stop() # "RuntimeError: `Study.stop` is supposed to be invoked inside an objective function or a callback."
        except Exception as e:
            logger.error(f"Unexpected error during optimization: {e}")
            logger.error(traceback.format_exc())
            #study.stop() # "RuntimeError: `Study.stop` is supposed to be invoked inside an objective function or a callback."

        return study

    except Exception:
        logger.exception(f"Unexpected error in run_study: {traceback.format_exc()}")
        #if study: # error: study might not be defined if the study creation failed
        if 'study' in locals() and study: # check if study exists
            logger.error("An error occurred during the study. Stopping the study.")
            #study.stop() # "RuntimeError: `Study.stop` is supposed to be invoked inside an objective function or a callback."
            return study
        else:
            logger.error("Study not created. Exiting.")
            return None
        

def run_optimization(config_path: Path, results_root: Optional[Path] = None) -> Tuple[Optional[optuna.Study], Optional[Path], Optional[str]]:
    # Use centralized default if no results_root specified
    if results_root is None:
        results_root = get_results_folder()

    # ----------------- Setup logging and directories -----------------

    # Use centralized logs folder at project root (not inside results folder)
    log_dir = ensure_folder_exists(get_logs_folder())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{config_path.stem}_{timestamp}"
    logger, log_file = setup_logger(log_dir, log_name)
    logger.info("\nLogging initialized." +
                f"\nResults will be saved to:\n{results_root}")

    try:
        config = Config.from_yaml(config_path)
    except Exception:
        logger.exception("Failed to load configuration.")
        return None, None, None
    
    # add logger to the config object
    config.logger = logger

    # Start memory monitoring only if enabled (tracemalloc adds runtime overhead)
    if getattr(config, 'enable_memory_monitor', False):
        tracemalloc.start()
        logger.info("Memory monitoring enabled (tracemalloc started).")

    # ---------------- Create a folder for the results ----------------

    results_folder_path, optimization_run_id = output_manager.create_output_folder(config, results_root)

    # ----------------- Run studies and save results -----------------

    start_time = time.time()

    study = run_study(config_path, config, results_folder_path, optimization_run_id)
    if not study:
        logger.error("Study failed to run.")
        return None, None, None
    
    # =============================================================================
    # POST-OPTIMIZATION: Results saving and cleanup
    # =============================================================================
    # This section MUST complete even if the process receives interrupt signals
    # (e.g., user closes the optimization console). KeyboardInterrupt is NOT a
    # subclass of Exception, so we must handle it explicitly to ensure results
    # are saved.

    csv_output_path = results_folder_path / f"{optimization_run_id}.csv"
    csv_saved = False

    try:
        # ----------------- Log results (important for debugging) -----------------
        # These logging operations are quick, so we complete them even if interrupted
        def _log_optimization_summary():
            """Log optimization summary. Separated so it can be retried if interrupted."""
            time_elapsed_msg, elapsed_time_in_seconds = output_manager.stop_the_timer(start_time)
            n_trials_this_run = config.max_trials
            n_trials_total = len(study.trials)
            calculated_max_duration = n_trials_this_run * config.tm_timeout / config.parallel_trials

            logger.info(f"\n\n--- Optimization run completed. ---\n" +
                        f"{time_elapsed_msg}\n" +
                        f"(Actual elapsed time / estimated max duration: {elapsed_time_in_seconds / calculated_max_duration:.2%})\n" +
                        f"(Estimated max duration: {calculated_max_duration / 3600:.1f} hours)\n\n" +
                        f"Iter range of interest: [{config.min_iterations}, {config.max_iterations}]\n" +
                        f"Tooth model timeout: {config.tm_timeout / 60:.2f} minutes\n" +
                        f"Parallel jobs: {config.parallel_trials} / {multiprocessing.cpu_count()} cores\n" +
                        f"Max trials: {n_trials_this_run}\n\n"
            )

        def _log_best_trial():
            """Log best trial info. Separated so it can be retried if interrupted."""
            n_trials_total = len(study.trials)
            logger.info(f"\nTotal accumulated trials in this study: {n_trials_total}\n" +
                        f"Best trial id: {study.best_trial.number}\n" +
                        f"Best trial loss: {study.best_value:.2f}\n"
                        )

        # Log summary - retry once if interrupted
        summary_logged = False
        for attempt in range(2):
            try:
                _log_optimization_summary()
                summary_logged = True
                break
            except KeyboardInterrupt:
                if attempt == 0:
                    continue  # Retry once
            except Exception as e:
                logger.error(f"Unexpected error during logging optimization summary: {e}")
                break

        # Log best trial - retry once if interrupted
        best_trial_logged = False
        for attempt in range(2):
            try:
                _log_best_trial()
                best_trial_logged = True
                break
            except KeyboardInterrupt:
                if attempt == 0:
                    continue  # Retry once
            except Exception as e:
                logger.error(f"Unexpected error during logging best trial: {e}")
                break

        # ----------------- Clean up (non-critical) ------------------
        # Delete the fort.2 file (if it exists) - it is not needed anymore
        fort_2_path = Path('fort.2')
        if fort_2_path.exists():
            try:
                fort_2_path.unlink()
            except Exception as e:
                logger.error(f"Error: Could not delete the fort.2 file: {e}")

        # ----------------- Save results (CRITICAL) -----------------
        try:
            trial_export.save_trials_dataframe(study, csv_output_path)
            logger.info(f"Results saved to {csv_output_path}")
            csv_saved = True
        except (KeyboardInterrupt, Exception) as e:
            if isinstance(e, KeyboardInterrupt):
                logger.warning("Interrupted during CSV save, retrying...")
            else:
                logger.error(f"Failed to save results to CSV: {e}")
            # Retry once if interrupted or failed
            if not csv_saved:
                try:
                    trial_export.save_trials_dataframe(study, csv_output_path)
                    logger.info(f"Results saved to {csv_output_path} (on retry)")
                    csv_saved = True
                except Exception as retry_e:
                    logger.error(f"Failed to save results to CSV on retry: {retry_e}")

        # ----------------- Database backup (non-critical) -----------------
        storage_type = getattr(config, 'storage_type', 'postgres')

        if storage_type == "postgres":
            try:
                export_postgres_database(config.storage_URL, results_folder_path, optimization_run_id)
            except (KeyboardInterrupt, RuntimeError, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    logger.info("Interrupted during database export, skipping...")
                else:
                    logger.error(f"Failed to export PostgreSQL database (not critical): {e}")
        else:
            logger.info(f"SQLite database saved in output folder: {config.study_name}.db")

        # ----------------- Centralized backup (non-critical) -----------------
        if config.results_backup_folder:
            try:
                backup_folder_path = Path(config.results_backup_folder)
                backup_folder_path.mkdir(parents=True, exist_ok=True)

                if csv_saved:
                    shutil.copy(csv_output_path, backup_folder_path)
                    logger.info(f"CSV results backed up to centralized folder: {backup_folder_path}")

                if storage_type == "postgres":
                    try:
                        export_postgres_database(config.storage_URL, backup_folder_path, optimization_run_id)
                    except Exception as e:
                        logger.error(f"Failed to export PostgreSQL database to backup (not critical): {e}")
                else:
                    sqlite_db_path = results_folder_path / f"{config.study_name}.db"
                    if sqlite_db_path.exists():
                        shutil.copy(sqlite_db_path, backup_folder_path)
                        logger.info(f"SQLite database backed up to centralized folder: {backup_folder_path}")

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    logger.info("Interrupted during backup, skipping...")
                else:
                    logger.error(f"Failed to copy results to centralized backup folder: {e}")

        # ----------------- Copy log file (non-critical) -----------------
        try:
            shutil.copy(log_file, results_folder_path)
        except (KeyboardInterrupt, Exception) as e:
            if not isinstance(e, KeyboardInterrupt):
                logger.error(f"Failed to copy log file: {e}")

        logger.info(f"Output saved to folder:\n{results_folder_path}")
        logger.info("Post-processing complete. Exiting...\n")
        if config.open_results_folder_after_run:
            try:
                from PySide6.QtGui import QDesktopServices
                from PySide6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(results_folder_path)))
            except Exception as e:
                logger.warning(f"Failed to open results folder: {e}")

    except KeyboardInterrupt:
        # Final fallback: if we get interrupted and CSV wasn't saved yet, try one last time
        logger.warning("Received interrupt during post-optimization cleanup...")
        if not csv_saved:
            try:
                logger.info("Attempting final CSV save...")
                trial_export.save_trials_dataframe(study, csv_output_path)
                logger.info(f"Results saved to {csv_output_path} (final attempt)")
                csv_saved = True
            except Exception as e:
                logger.error(f"Final CSV save attempt failed: {e}")
                # Point user to the ongoing CSV which has incremental results
                ongoing_csv = results_folder_path / f"ongoing_{optimization_run_id}.csv"
                if ongoing_csv.exists():
                    logger.info(f"Incremental results are available in: {ongoing_csv}")

    return study, results_folder_path, optimization_run_id


def main() -> None:
    """
    CLI entry point. Parses args, loads configs, runs studies, saves results, optionally visualizes.
    """
    import sys

    # ----------------- Parse command line arguments -----------------

    parser = argparse.ArgumentParser(description="Meta-learning runner for parameter optimization")
    parser.add_argument("config", help="Path to a single YAML config file", type=Path)
    parser.add_argument("--out_dir", help="Directory to save results (default: project results folder)", type=Path, default=None)
    #parser.add_argument("--visualize", action="store_true", help="Enable Optuna visualizations")
    #parser.add_argument("--prev_dir", help="Directory of previous runs", type=Path, default=None) # a.k.a. "previous_results_folder_path" (i.e. prev. _results_ folder)
    args = parser.parse_args()

    config_path = args.config
    results_root = args.out_dir

    study, results_folder_path, optimization_run_id = run_optimization(config_path, results_root)

    # Exit with error code if optimization failed
    if study is None:
        sys.exit(1)

    #----------------- Visualize results -----------------
    
    ''' # NOTE this sort of plot is already included in the other optuna visualizations
    try:
        csv_path = results_folder_path / f"{optimization_run_id}.csv"
        import gui_utils 
        gui_utils.plot_loss_vs_trial(csv_path, save_as_png=True)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error("Could not visualize results.")
    '''
    # import gui_optuna_visuals
    # study = gui_optuna_visuals.load_study(results_folder_path) # load the study from the output folder
    # gui_optuna_visuals.show_visualizations(study) # show the visualizations in a web browser # NOTE these can take a while to load, so run them manually if needed instead
    # gui_optuna_visuals.save_visualizations(study, results_folder_path) # save the visualizations as HTML files in the output folder

if __name__ == "__main__":
    main()
