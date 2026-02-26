import os
import logging
import subprocess
import psycopg2
import psycopg2.sql
import optuna

logger = logging.getLogger(__name__)

# --- SETTINGS ---

# Connection details for the PostgreSQL server
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = None  # Set to None if using .pgpass
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432

# Names
NEW_DB_NAME = "new_optuna_study_db"

# Path to the SQL dump file you want to import
SQL_DUMP_PATH = r"path_to_your_pg_dump_file.sql"

# --- HELPER FUNCTIONS ---

def create_new_database(db_name: str) -> None:
    """Create a new PostgreSQL database, skipping if it already exists."""
    conn = None
    cursor = None
    try:
        conn_params = {
            "dbname": "postgres",
            "user": POSTGRES_USER,
            "host": POSTGRES_HOST,
            "port": POSTGRES_PORT,
        }
        if POSTGRES_PASSWORD:
            conn_params["password"] = POSTGRES_PASSWORD

        conn = psycopg2.connect(**conn_params, connect_timeout=10)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        cursor.execute(
            psycopg2.sql.SQL("CREATE DATABASE {};").format(
                psycopg2.sql.Identifier(db_name)
            )
        )
        logger.info(f"Database '{db_name}' created successfully.")

    except psycopg2.errors.DuplicateDatabase:
        logger.info(f"Database '{db_name}' already exists. Skipping creation.")
    except Exception as e:
        logger.error(f"Failed to create database '{db_name}': {e}")
        raise
    finally:
        # Ensure cursor and connection are always closed to prevent resource leaks
        if cursor is not None:
            try:
                cursor.close()
            except Exception as e:
                logger.warning(f"Failed to close database cursor: {e}")
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to close database connection: {e}")

def import_sql_dump_into_db(db_name: str, sql_path: str) -> None:
    """Import a SQL dump file into the specified database using psql."""
    cmd = [
        "psql",
        f"-U", POSTGRES_USER,
        f"-h", POSTGRES_HOST,
        f"-p", str(POSTGRES_PORT),
        "-d", db_name,
        "-f", sql_path
    ]

    # Use a temporary .pgpass file instead of PGPASSWORD env var to avoid
    # exposing credentials in process environment (visible via ps/proc)
    import tempfile
    pgpass_file = None
    env = os.environ.copy()
    try:
        if POSTGRES_PASSWORD:
            pgpass_content = f"{POSTGRES_HOST}:{POSTGRES_PORT}:{db_name}:{POSTGRES_USER}:{POSTGRES_PASSWORD}\n"
            pgpass_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.pgpass', delete=False, encoding='utf-8'
            )
            pgpass_file.write(pgpass_content)
            pgpass_file.close()
            os.chmod(pgpass_file.name, 0o600)
            env["PGPASSFILE"] = pgpass_file.name

        subprocess.run(cmd, check=True, env=env)
        logger.info(f"Successfully imported dump into '{db_name}' from '{sql_path}'.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running psql command: {e}")
        raise
    finally:
        if pgpass_file is not None:
            try:
                os.unlink(pgpass_file.name)
            except OSError:
                pass

def list_studies_in_db(db_name: str) -> None:
    """List all Optuna studies in the specified database."""
    if POSTGRES_PASSWORD:
        storage_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{db_name}"
    else:
        storage_url = f"postgresql://{POSTGRES_USER}@{POSTGRES_HOST}:{POSTGRES_PORT}/{db_name}"

    summaries = optuna.study.get_all_study_summaries(storage=storage_url)
    logger.info("Studies in database:")
    for s in summaries:
        logger.info(f"- Study ID: {s.study_id}, Study Name: {s.study_name}")

# --- MAIN SCRIPT ---

def main() -> None:
    """Run the full import pipeline: create DB, import dump, list studies."""
    create_new_database(NEW_DB_NAME)
    import_sql_dump_into_db(NEW_DB_NAME, SQL_DUMP_PATH)
    list_studies_in_db(NEW_DB_NAME)

if __name__ == "__main__":
    main()
