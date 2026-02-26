"""
The wrapped script is now ready as a CLI tool. You can run it like this:

python list_studies.py postgresql://user:pass@host:5432/your_db_name

"""

import optuna
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional

from bites.config.config_loader import Config
from bites.database.pg_db_manager import ensure_database_for_target


def list_studies(storage_url) -> Optional[pd.DataFrame]:
    from bites.database.pg_db_manager import _sanitize_url_for_logging
    print(f"Connecting to {_sanitize_url_for_logging(storage_url)}...")
    try:
        storage = optuna.storages.RDBStorage(url=storage_url)
        study_summaries = optuna.study.get_all_study_summaries(storage=storage)

        print(f"\nFound {len(study_summaries)} studies:")
        records = []
        for study in study_summaries:
            records.append({
                "study_name": study.study_name,
                "direction": study.direction.name,
                #"n_trials": len(study.trials), # Failed to connect or list studies: 'StudySummary' object has no attribute 'trials'
                "best_value": study.best_trial.value if study.best_trial else None,
                "user_attrs": study.user_attrs
            })

        df = pd.DataFrame(records)
        print(df.to_string(index=False))
        return df

    except Exception as e:
        print("Failed to connect or list studies:", e)
        return None

def main(storage_url: str) -> None:
    list_studies(storage_url)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List all Optuna studies in a PostgreSQL database")
    parser.add_argument("--storage_url", help="PostgreSQL storage URL")
    parser.add_argument("--config", help="Path to YAML config file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.storage_url:
        main(args.storage_url)
    elif args.config:
        config_path = Path(args.config)
        config = Config.from_yaml(config_path)
        POSTGRES_URL = ensure_database_for_target(config)
        main(POSTGRES_URL)
    else:
        # Default config path when run from VS Code with no args
        config_path = Path(__file__).parent / "../configs/variant2.yaml"
        config = Config.from_yaml(config_path)
        POSTGRES_URL = ensure_database_for_target(config)
        main(POSTGRES_URL)