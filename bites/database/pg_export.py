"""
PostgreSQL Export - Export database to SQL dump file.
"""

import os
import subprocess
import urllib.parse
from pathlib import Path
from typing import Union


def export_postgres_database(url: str, output_folder: Union[str, Path], optimization_run_id: str) -> None:
    """Export a PostgreSQL database to a SQL file."""
    output_folder = Path(output_folder).resolve()

    # Parse the URL
    parsed_url = urllib.parse.urlparse(url)

    # Extract connection components
    username = parsed_url.username
    password = parsed_url.password
    hostname = parsed_url.hostname or "localhost"
    port = parsed_url.port or 5432
    dbname = parsed_url.path.lstrip("/")

    if not all([username, dbname]):
        raise ValueError("URL must include at least username and database name.")

    # Ensure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Path for the dump file
    output_file = output_folder / f"{dbname}_{optimization_run_id}.sql"

    # Build the pg_dump command
    cmd = [
        "pg_dump",
        "-h", hostname,
        "-p", str(port),
        "-U", username,
        "-F", "p",  # "p" = plain text format (outputs a SQL script)
        "-f", str(output_file),  # pg_dump needs string path
        dbname
    ]

    # Use a temporary .pgpass file instead of PGPASSWORD env var to avoid
    # exposing credentials in process environment (visible via ps/proc)
    import tempfile
    pgpass_file = None
    env = os.environ.copy()
    try:
        if password:
            pgpass_content = f"{hostname}:{port}:{dbname}:{username}:{password}\n"
            pgpass_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.pgpass', delete=False, encoding='utf-8'
            )
            pgpass_file.write(pgpass_content)
            pgpass_file.close()
            os.chmod(pgpass_file.name, 0o600)
            env["PGPASSFILE"] = pgpass_file.name

        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to export database: {e}")
    finally:
        if pgpass_file is not None:
            try:
                os.unlink(pgpass_file.name)
            except OSError:
                pass
