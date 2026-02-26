"""
Root conftest for BITES test suite.

Provides shared fixtures used across multiple test modules:
- Config dictionaries and paths for testing config loading
- Trial parameter dictionaries matching tooth model expectations
- Simple mesh objects for alignment/mesh tests
- Mock loggers for capturing log output
- Test data directory path resolution
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import trimesh


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@pytest.fixture
def test_data_dir() -> Path:
    """Return the absolute path to the tests/test_data/ directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_config_path(test_data_dir) -> Path:
    """Return the absolute path to the sample YAML config file."""
    return test_data_dir / "sample_config.yaml"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config_dict() -> dict:
    """Return a minimal valid config dictionary matching Config dataclass fields.

    This mirrors the structure of sample_config.yaml as a Python dict,
    useful for tests that need to manipulate config values before loading.
    """
    return {
        "target_shape_path": "target_shapes/example.off",
        "min_cut_height": 12,
        "study_name": "test_study",
        "max_trials": 100,
        "study_timeout": 1,
        "min_iterations": 5000,
        "max_iterations": 15000,
        "iter_interval": 1000,
        "tm_timeout": 300,
        "multivariate": True,
        "parallel_search_enabled": True,
        "parallel_trials": 4,
        "storage_mode": "shared",
        "alignment_algorithms": ["STatac-C-CPD"],
        "params_to_optimize": ["Egr", "Act"],
        "params_optimization_ranges": {
            "Egr": [0.01, 0.05],
            "Act": [0.01, 3.0],
        },
        "params_fixed_values": {
            "-": 0.0, "Egr": 0.025, "Mgr": 200.0, "Rep": 1.0, "Swi": 0.0,
            "Adh": 0.001, "Act": 0.22, "Inh": 1.5, "Sec": 0.03, "Da": 0.2,
            "Di": 0.2, "Ds": 0.2, "Int": 0.19, "Set": 0.95, "Boy": 0.1,
            "Dff": 0.0004, "Bgr": 1.0, "Abi": 12.0, "Pbi": 18.0, "Lbi": 1.0,
            "Bbi": 1.0, "Rad": 2.0, "Deg": 0.1, "Dgr": 10000.0, "Ntr": 1e-05,
            "Bwi": 0.8, "Ina": 0.0, "uMgr": 0.0, "iter": 4000.0,
        },
        "bias_param_config": {
            "Abi_value_for_Pbi_too": False,
            "Abi_Pbi_factor": 1.0,
            "Bbi_value_for_Lbi_too": False,
            "Bbi_Lbi_factor": 1.0,
        },
        "postgresql_config": {
            "user": "postgres",
            "password": "PGPASS_ENABLED",
            "host": "localhost",
            "port": 5432,
            "db_name": "postgres",
        },
        "model_executable_path_override": "",
        "tm_subfolder_name_override": "",
        "results_backup_folder_override": "",
        "results_folder_override": "",
        "configs_folder_override": "",
        "assets_folder_override": "",
        "save_original_tooth_model_meshes": False,
        "open_results_folder_after_run": False,
        "param_names_all": [
            "Egr", "Mgr", "Rep", "Swi", "Adh", "Act", "Inh", "Sec",
            "Da", "Di", "Ds", "Int", "Set", "Boy", "Dff", "Bgr",
            "Abi", "Pbi", "Lbi", "Bbi", "Rad", "Deg", "Dgr", "Ntr",
            "Bwi", "Ina", "uMgr",
        ],
    }


# ---------------------------------------------------------------------------
# Trial parameters (tooth model)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_trial_params() -> dict:
    """Return a complete trial parameters dict with all 27 named params + '-'.

    This is the dict that construct_model_parameters_string() expects.
    The '-' placeholder is included (3 occurrences in the names array).
    """
    return {
        "Egr": 0.025, "Mgr": 200.0, "Rep": 1.0, "Swi": 0.0,
        "Adh": 0.001, "Act": 0.22, "Inh": 1.5, "-": 0.0,
        "Sec": 0.03, "Da": 0.2, "Di": 0.2, "Ds": 0.2,
        "Int": 0.19, "Set": 0.95, "Boy": 0.1, "Dff": 0.0004,
        "Bgr": 1.0, "Abi": 12.0, "Pbi": 18.0, "Lbi": 1.0,
        "Bbi": 1.0, "Rad": 2.0, "Deg": 0.1, "Dgr": 10000.0,
        "Ntr": 1e-05, "Bwi": 0.8, "Ina": 0.0, "uMgr": 0.0,
        "iter_max": 15000,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_logger() -> logging.Logger:
    """Return a logger that captures messages in a StringHandler for assertion.

    Usage in tests::

        mock_logger.warning("some message")
        assert "some message" in mock_logger.handlers[0].stream.getvalue()
    """
    import io

    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)

    return logger


# ---------------------------------------------------------------------------
# Mesh fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_trimesh() -> trimesh.Trimesh:
    """Return a minimal tetrahedron trimesh (4 vertices, 4 faces).

    Useful as a quick, non-degenerate mesh for validation tests.
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


@pytest.fixture
def box_source_mesh() -> trimesh.Trimesh:
    """Return a unit box mesh centered at origin (for alignment tests)."""
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


@pytest.fixture
def box_target_mesh() -> trimesh.Trimesh:
    """Return a larger box mesh offset from origin (for alignment target)."""
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    mesh.apply_translation([1.0, 1.0, 1.0])
    return mesh
