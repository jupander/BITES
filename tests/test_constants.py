"""
Tests for bites.constants module.

Covers:
- format_mesh_identifier(): mesh identifier string formatting
- format_database_name(): database name prefixing
- get_cut_mesh_filename(): cut mesh filename generation with rounding
- algorithm_cuts_target(): STatac algorithm detection
- ShapeBrowserConfig: configuration constant invariants
"""

import pytest

from bites.constants import (
    DATABASE_NAME_PREFIX,
    RUN_IDENTIFIER_PREFIX,
    CUT_HEIGHT_PRECISION,
    IMPLEMENTED_ALIGNMENT_ALGORITHMS,
    ShapeBrowserConfig,
    format_mesh_identifier,
    format_database_name,
    get_cut_mesh_filename,
    algorithm_cuts_target,
)


# ---------------------------------------------------------------------------
# format_mesh_identifier
# ---------------------------------------------------------------------------

class TestFormatMeshIdentifier:
    """Tests for format_mesh_identifier(run_id, iter_num)."""

    def test_basic_format(self):
        """Produces '{run_id}__{iter_num}' format."""
        result = format_mesh_identifier("run240823-145700abc", 7000)
        assert result == "run240823-145700abc__7000"

    def test_zero_iteration(self):
        """Handles iteration number 0."""
        result = format_mesh_identifier("run000000-000000aaa", 0)
        assert result == "run000000-000000aaa__0"

    def test_large_iteration(self):
        """Handles large iteration numbers."""
        result = format_mesh_identifier("runX", 50000)
        assert result == "runX__50000"

    def test_double_underscore_separator(self):
        """Separator is exactly two underscores."""
        result = format_mesh_identifier("id", 1)
        assert "__" in result
        parts = result.split("__")
        assert len(parts) == 2


# ---------------------------------------------------------------------------
# format_database_name
# ---------------------------------------------------------------------------

class TestFormatDatabaseName:
    """Tests for format_database_name(study_name)."""

    def test_prefixes_with_db(self):
        """Prefixes study name with DATABASE_NAME_PREFIX."""
        result = format_database_name("my_study")
        assert result == f"{DATABASE_NAME_PREFIX}my_study"

    def test_prefix_value(self):
        """DATABASE_NAME_PREFIX is 'db_'."""
        assert DATABASE_NAME_PREFIX == "db_"

    def test_empty_study_name(self):
        """Handles empty study name without error."""
        result = format_database_name("")
        assert result == "db_"


# ---------------------------------------------------------------------------
# get_cut_mesh_filename
# ---------------------------------------------------------------------------

class TestGetCutMeshFilename:
    """Tests for get_cut_mesh_filename(height)."""

    def test_integer_height(self):
        """Integer height produces '.0' in filename."""
        result = get_cut_mesh_filename(6)
        assert result == "target_cut_at_height_6.0.ply"

    def test_float_height(self):
        """Float height is rounded to CUT_HEIGHT_PRECISION decimal places."""
        result = get_cut_mesh_filename(6.5)
        assert result == "target_cut_at_height_6.5.ply"

    def test_rounding(self):
        """Height with excess decimals is rounded."""
        result = get_cut_mesh_filename(6.789)
        # CUT_HEIGHT_PRECISION is 1, so rounds to 6.8
        assert result == "target_cut_at_height_6.8.ply"

    def test_precision_constant(self):
        """CUT_HEIGHT_PRECISION is 1 decimal place."""
        assert CUT_HEIGHT_PRECISION == 1

    def test_ply_extension(self):
        """Filename ends with .ply."""
        result = get_cut_mesh_filename(10)
        assert result.endswith(".ply")


# ---------------------------------------------------------------------------
# algorithm_cuts_target
# ---------------------------------------------------------------------------

class TestAlgorithmCutsTarget:
    """Tests for algorithm_cuts_target(algorithm_name)."""

    @pytest.mark.parametrize("algo", [
        "STatac-AB", "STatac-C", "STatac-AB-CPD", "STatac-C-CPD",
    ])
    def test_statac_variants_return_true(self, algo):
        """All STatac variants are recognized as cutting the target."""
        assert algorithm_cuts_target(algo) is True

    @pytest.mark.parametrize("algo", ["ST", "CPD-only", "ST-CPD"])
    def test_non_statac_return_false(self, algo):
        """Non-STatac algorithms return False."""
        assert algorithm_cuts_target(algo) is False

    def test_case_insensitive(self):
        """Detection is case-insensitive."""
        assert algorithm_cuts_target("STATAC-AB") is True
        assert algorithm_cuts_target("statac-c") is True

    def test_empty_string(self):
        """Empty string returns False."""
        assert algorithm_cuts_target("") is False

    def test_non_string_returns_false(self):
        """Non-string input returns False without raising."""
        assert algorithm_cuts_target(123) is False
        assert algorithm_cuts_target(None) is False


# ---------------------------------------------------------------------------
# ShapeBrowserConfig
# ---------------------------------------------------------------------------

class TestShapeBrowserConfig:
    """Tests for ShapeBrowserConfig class constants."""

    def test_slideshow_speed_ordering(self):
        """MIN < DEFAULT < MAX for slideshow display times."""
        assert ShapeBrowserConfig.SLIDESHOW_MIN_DISPLAY_TIME \
            < ShapeBrowserConfig.SLIDESHOW_DEFAULT_DISPLAY_TIME \
            < ShapeBrowserConfig.SLIDESHOW_MAX_DISPLAY_TIME

    def test_speed_multiplier_greater_than_one(self):
        """Speed multiplier is > 1 (so multiplying/dividing changes speed)."""
        assert ShapeBrowserConfig.SLIDESHOW_SPEED_MULTIPLIER > 1.0

    def test_algorithm_color_order_matches_colors_dict(self):
        """Every color in ALGORITHM_COLOR_ORDER exists in ALGORITHM_COLORS."""
        for color_name in ShapeBrowserConfig.ALGORITHM_COLOR_ORDER:
            assert color_name in ShapeBrowserConfig.ALGORITHM_COLORS

    def test_algorithm_colors_are_rgb_lists(self):
        """Each color value is a list of 3 numbers in [0, 1]."""
        for name, rgb in ShapeBrowserConfig.ALGORITHM_COLORS.items():
            assert len(rgb) == 3, f"Color '{name}' should have 3 components"
            for component in rgb:
                assert 0.0 <= component <= 1.0, \
                    f"Color '{name}' component {component} out of [0,1] range"

    def test_enough_colors_for_algorithms(self):
        """At least as many colors as implemented alignment algorithms."""
        assert len(ShapeBrowserConfig.ALGORITHM_COLOR_ORDER) >= \
            len(IMPLEMENTED_ALIGNMENT_ALGORITHMS)


# ---------------------------------------------------------------------------
# Module-level constants smoke tests
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Smoke tests for important module-level constants."""

    def test_run_identifier_prefix(self):
        """RUN_IDENTIFIER_PREFIX is 'run'."""
        assert RUN_IDENTIFIER_PREFIX == "run"

    def test_implemented_algorithms_non_empty(self):
        """IMPLEMENTED_ALIGNMENT_ALGORITHMS has at least one entry."""
        assert len(IMPLEMENTED_ALIGNMENT_ALGORITHMS) > 0

    def test_implemented_algorithms_are_strings(self):
        """All implemented algorithms are strings."""
        for algo in IMPLEMENTED_ALIGNMENT_ALGORITHMS:
            assert isinstance(algo, str)
