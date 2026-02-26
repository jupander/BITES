"""
Trial Export - Optuna trial data extraction and CSV export.
"""

import csv
import json
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd


def _serialize_for_csv(value: Any) -> Any:
    """
    Serialize a value for CSV export. Complex types (dict, list) are JSON-encoded.
    Datetime objects are converted to ISO format strings.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    elif isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    elif hasattr(value, 'tolist'):  # numpy arrays
        return json.dumps(value.tolist())
    elif hasattr(value, 'item'):  # numpy scalars
        return value.item()
    else:
        return value


def extract_trial_data(t: Any) -> Dict[str, Any]:
    """
    Flatten a single Optuna trial into a dictionary for CSV export.
    Complex types (dicts, lists) are JSON-serialized for safe CSV storage.
    """
    row = {
        "trial_number": t.number,
        "datetime_start": t.datetime_start,
        "state": t.state.name,  # e.g. "COMPLETE", "PRUNED", "RUNNING", "WAITING", "FAIL"
        "loss": t.value,
        "params": t.params,
    }

    # Add the user attributes
    row = {**row, **t.user_attrs}

    # Serialize complex types (dicts, lists, datetimes) for CSV compatibility
    row = {k: _serialize_for_csv(v) for k, v in row.items()}

    return row


def get_fixed_columns() -> List[str]:
    """To have these defined in one place. Used by save_trials_dataframe and save_trial_callback."""
    return [
        # -- Default columns --
        "trial_number",
        "datetime_start",
        "state",
        "loss",
        "params",
        # -- User attributes --
        # trial properties:
        "trial_parameters_dict",
        "tm_run_identifier",
        "chosen_mesh_identifier",
        "best_algorithm",
        "best_alignment_algorithm_per_iter",
        "best_chamfer_distance_per_iter",
        "chamfer_distances_all",
        "used_cut_heights",  # dict of cut heights by algorithm, e.g. {"STatac": 6.0}
        "evaluate_loss_duration",
        "tm_run_duration",
        "tm_run_complete",
        "duration_by_iter_dict",
        "chosen_mesh_duration",
        "top_iter_reached",
        "iter_of_chosen_mesh",
        # optimization run properties:
        "iter_range",
        "tm_run_timeout",
        "params_to_optimize",
        "alignment_algorithms",
        "run_config",
        'Egr', 'Mgr', 'Rep', 'Swi', 'Adh', 'Act', 'Inh', 'Sec', 'Da', 'Di', 'Ds', 'Int', 'Set', 'Boy',
        'Dff', 'Bgr', 'Abi', 'Pbi', 'Lbi', 'Bbi', 'Rad', 'Deg', 'Dgr', 'Ntr', 'Bwi', 'Ina', 'uMgr'
    ]


def save_trials_dataframe(study: Any, path: Union[str, Path]) -> None:
    """
    Save detailed trial data to CSV, ensuring a flat and consistent format.
    Complex types are JSON-serialized and properly quoted to prevent CSV corruption.
    """
    rows = [extract_trial_data(t) for t in study.trials]
    df = pd.DataFrame(rows)

    # Define the desired column order
    fixed_columns = get_fixed_columns()
    # Potential columns that are not named in the fixed_columns list
    other_columns = sorted([col for col in df.columns if col not in fixed_columns])

    # Final column order: fixed + user-defined
    ordered_columns = fixed_columns + other_columns

    # Ensure all columns are present in the DataFrame (fill missing with None)
    for col in ordered_columns:
        if col not in df.columns:
            df[col] = None

    # Save with proper quoting to handle embedded separators and special characters
    df.to_csv(
        path,
        index=False,
        sep=";",
        columns=ordered_columns,
        quoting=csv.QUOTE_NONNUMERIC,  # Quote all non-numeric fields
        escapechar='\\',  # Escape character for embedded quotes
    )
