"""
Example CLI usage:

python ensure_database.py \
    --admin_url postgresql://user:pass@localhost:5432/postgres \
    --target target_shape_A
"""

import re
import logging
import psycopg2
import psycopg2.extensions
import psycopg2.sql
from typing import Optional
from urllib.parse import urlparse, urlunparse

# Configure logging
#logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

from bites.constants import format_database_name

DISABLED_PASSWORD_TOKEN = "PGPASS_ENABLED"

logger = logging.getLogger(__name__)


def _sanitize_url_for_logging(url: str) -> str:
    """Remove password from a PostgreSQL URL before logging."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            safe_netloc = f"{parsed.username}:****@{parsed.hostname}"
            if parsed.port:
                safe_netloc += f":{parsed.port}"
            return urlunparse(parsed._replace(netloc=safe_netloc))
        return url
    except Exception:
        return "[URL could not be sanitized]"


def sanitize_db_name(name: str) -> str:
    """
    Sanitize a database name to be PostgreSQL-safe:
    - Lowercase
    - Alphanumeric + underscore only
    - Truncated to 63 characters (PostgreSQL limit)
    """
    name = name.lower()
    name = re.sub(r'[^a-z0-9_]', '_', name)
    return name[:63]

def get_postgres_admin_url(config: 'Config') -> str:
    """
    Construct the PostgreSQL connection URL based on the provided config.
    Assumes the config contains a 'postgres' section with necessary details.

    Args:
        config: Application config object with postgresql_config attribute.

    Returns:
        PostgreSQL connection URL formatted as 'postgresql://user:password@host:port/db_name'
    """
    
    logger = config.logger

    try:
        pg_conf = config.postgresql_config 
    except AttributeError:
        #logging.error("PostgreSQL configuration not found in the provided config.")
        logger.error("PostgreSQL configuration not found in the provided config.")
        raise

    # Only include password if it's present (you can keep this or not depending on URL style)
    if pg_conf.password and pg_conf.password != DISABLED_PASSWORD_TOKEN:
        return f"postgresql://{pg_conf.user}:{pg_conf.password}@{pg_conf.host}:{pg_conf.port}/{pg_conf.db_name}"
    else:
        return f"postgresql://{pg_conf.user}@{pg_conf.host}:{pg_conf.port}/{pg_conf.db_name}"


def ensure_database_for_target(config: 'Config') -> str:
    """
    Connects to the PostgreSQL server using an admin URL, checks if a database named
    after the given target_id exists, and creates it if not.

    Args:
        config: Application config object with study_name and postgresql_config.

    Returns:
        Full PostgreSQL connection URL for the (possibly newly created) database.

    Raises:
        psycopg2.Error: If database connection or creation fails.
        AttributeError: If config is missing required attributes.
    """

    try:
        logger = config.logger
    except AttributeError:
        # create a default logger if not provided in the config object
        logger = logging.getLogger(__name__)
        logger.warning("No logger found in config. Using module-level logger.")
        logger.setLevel(logging.INFO)
        config.logger = logger

    new_db_name = format_database_name(config.study_name)
    new_db_name = sanitize_db_name(new_db_name)
    logger.info(f"Sanitized database name: {new_db_name}")

    # NOTE doing this the long way here for clarity
    postgres_admin_url = get_postgres_admin_url(config)
    logger.info(f"Postgres admin URL: {_sanitize_url_for_logging(postgres_admin_url)}")
    parsed = urlparse(postgres_admin_url)
    base_db = parsed.path.lstrip('/') #or 'postgres' # would default to 'postgres' if not specified

    logger.info(f"Checking if database '{new_db_name}' exists...")

    admin_conn = None
    cursor = None
    try:
        # ---------------- new version
        conn_kwargs = {
            "dbname": base_db,
            "user": parsed.username,
            "host": parsed.hostname,
            "port": parsed.port,
            "connect_timeout": 10,
        }

        #DISABLED_PASSWORD_TOKEN = "PGPASS_ENABLED"

        # Only include password if not explicitly disabled
        if parsed.password and parsed.password != DISABLED_PASSWORD_TOKEN:
            conn_kwargs["password"] = parsed.password
            logger.info("Using the postgreSQL password provided in config.")
        else:
            logger.info("Password disabled in config - relying on .pgpass or environment-based authentication.")

        try:
            admin_conn = psycopg2.connect(**conn_kwargs)
        except Exception as e:
            # Log concise error; full traceback available via debug if needed
            logger.error(f"PostgreSQL connection error: {e}")
            raise
        # ----------------

        admin_conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = admin_conn.cursor()

        # Check if the database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (new_db_name,))
        exists = cursor.fetchone()

        if exists:
            logger.info(f"Existing database '{new_db_name}' found.")
        else:
            logger.info(f"Database '{new_db_name}' not found. Creating it now...")
            cursor.execute(
                psycopg2.sql.SQL("CREATE DATABASE {};").format(
                    psycopg2.sql.Identifier(new_db_name)
                )
            )
            logger.info(f"Database '{new_db_name}' created successfully.")

        #return f"{postgres_admin_url}/{new_db_name}" # return the full connection URL for the new database
        # how to elegantly change the database name in the URL?
        #return postgres_admin_url.replace(base_db, new_db_name) # this is a bad idea, as it will replace unintended stuff if the base_db is 'postgres' (which is the default)
        # Rebuild the URL with the new database name at the end
        new_path = f"/{new_db_name}"
        new_url = urlunparse(parsed._replace(path=new_path))
        return new_url

    except Exception as e:
        # Don't log full traceback here - let the caller handle display
        logger.error(f"Database setup failed: {e}")
        raise
    finally:
        # Ensure cursor and connection are always closed to prevent resource leaks
        if cursor is not None:
            try:
                cursor.close()
            except Exception as e:
                logger.warning(f"Failed to close database cursor: {e}")
        if admin_conn is not None:
            try:
                admin_conn.close()
            except Exception as e:
                logger.warning(f"Failed to close database connection: {e}")

""" # outdated example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensure PostgreSQL database for a given target ID")
    parser.add_argument("--admin_url", type=str, required=True, help="Admin PostgreSQL URL, e.g. postgresql://user:pass@host:5432/postgres")
    parser.add_argument("--target", type=str, required=True, help="Target shape ID (used to name the database)")

    args = parser.parse_args()
    ensure_database_for_target(args.admin_url, args.target)
"""