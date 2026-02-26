"""
Tests for bites.database.pg_db_manager module.

Covers:
- sanitize_db_name(): PostgreSQL-safe name sanitization
- get_postgres_admin_url(): connection URL construction (mocked config)
- DISABLED_PASSWORD_TOKEN constant
"""

from unittest.mock import MagicMock

import pytest

from bites.database.pg_db_manager import (
    sanitize_db_name,
    get_postgres_admin_url,
    DISABLED_PASSWORD_TOKEN,
)


# ---------------------------------------------------------------------------
# sanitize_db_name
# ---------------------------------------------------------------------------

class TestSanitizeDbName:
    """Tests for sanitize_db_name()."""

    def test_lowercase(self):
        """Name is lowercased."""
        assert sanitize_db_name("MyStudy") == "mystudy"

    def test_special_chars_replaced(self):
        """Non-alphanumeric chars (except underscore) are replaced with underscore."""
        assert sanitize_db_name("my-study!") == "my_study_"

    def test_spaces_replaced(self):
        """Spaces are replaced with underscores."""
        assert sanitize_db_name("my study") == "my_study"

    def test_truncation_at_63(self):
        """Long names are truncated to 63 characters."""
        long_name = "a" * 100
        result = sanitize_db_name(long_name)
        assert len(result) == 63

    def test_already_valid(self):
        """Already valid name passes through unchanged."""
        assert sanitize_db_name("abc123_def") == "abc123_def"

    @pytest.mark.parametrize("input_name,expected", [
        ("My Study!", "my_study_"),
        ("hello-world", "hello_world"),
        ("abc123_def", "abc123_def"),
        ("UPPER", "upper"),
        ("dots.and.more", "dots_and_more"),
        ("a" * 100, "a" * 63),
    ])
    def test_parametrized(self, input_name, expected):
        """Various sanitization scenarios produce correct output."""
        assert sanitize_db_name(input_name) == expected

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert sanitize_db_name("") == ""

    def test_underscore_preserved(self):
        """Underscores are preserved."""
        assert sanitize_db_name("a_b_c") == "a_b_c"

    def test_numbers_preserved(self):
        """Numbers are preserved."""
        assert sanitize_db_name("db123") == "db123"


# ---------------------------------------------------------------------------
# DISABLED_PASSWORD_TOKEN
# ---------------------------------------------------------------------------

class TestDisabledPasswordToken:
    """Tests for the DISABLED_PASSWORD_TOKEN constant."""

    def test_value(self):
        """Token is 'PGPASS_ENABLED'."""
        assert DISABLED_PASSWORD_TOKEN == "PGPASS_ENABLED"


# ---------------------------------------------------------------------------
# get_postgres_admin_url
# ---------------------------------------------------------------------------

class TestGetPostgresAdminUrl:
    """Tests for get_postgres_admin_url() with mocked config."""

    def _make_config(self, user="postgres", password="secret",
                     host="localhost", port=5432, db_name="postgres"):
        """Create a mock config with postgresql_config."""
        config = MagicMock()
        pg_conf = MagicMock()
        pg_conf.user = user
        pg_conf.password = password
        pg_conf.host = host
        pg_conf.port = port
        pg_conf.db_name = db_name
        config.postgresql_config = pg_conf
        config.logger = MagicMock()
        return config

    def test_with_password(self):
        """URL includes password when set."""
        config = self._make_config(password="secret")
        url = get_postgres_admin_url(config)
        assert "secret" in url
        assert url == "postgresql://postgres:secret@localhost:5432/postgres"

    def test_without_password_disabled(self):
        """URL excludes password when DISABLED_PASSWORD_TOKEN is used."""
        config = self._make_config(password=DISABLED_PASSWORD_TOKEN)
        url = get_postgres_admin_url(config)
        assert DISABLED_PASSWORD_TOKEN not in url
        assert url == "postgresql://postgres@localhost:5432/postgres"

    def test_empty_password(self):
        """URL excludes password when password is empty string."""
        config = self._make_config(password="")
        url = get_postgres_admin_url(config)
        assert url == "postgresql://postgres@localhost:5432/postgres"

    def test_none_password(self):
        """URL excludes password when password is None."""
        config = self._make_config(password=None)
        url = get_postgres_admin_url(config)
        assert url == "postgresql://postgres@localhost:5432/postgres"

    def test_custom_host_port(self):
        """Custom host and port appear in URL."""
        config = self._make_config(host="db.example.com", port=5433,
                                    password=DISABLED_PASSWORD_TOKEN)
        url = get_postgres_admin_url(config)
        assert "db.example.com:5433" in url

    def test_missing_postgresql_config_raises(self):
        """Missing postgresql_config attribute raises AttributeError."""
        config = MagicMock(spec=[])  # no attributes
        config.logger = MagicMock()
        with pytest.raises(AttributeError):
            get_postgres_admin_url(config)
