"""
Config I/O - Serialization and persistence of configuration data.
"""

import datetime
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import asdict, is_dataclass

import bites.config.config_template_injector as config_template_injector
from bites.constants import TEMPLATE_FILENAME
from bites.paths import get_assets_folder


def config_to_serializable_dict(config: Any) -> Dict[str, Any]:
    """Convert a config object to a JSON-serializable dictionary."""
    from enum import Enum

    def clean(value):
        if isinstance(value, Path):
            return str(value)
        elif isinstance(value, (datetime.datetime, datetime.date)):
            return value.isoformat()
        elif isinstance(value, datetime.timedelta):
            return value.total_seconds()
        elif isinstance(value, Enum):
            return value.value
        elif hasattr(value, 'tolist'):  # numpy arrays
            return value.tolist()
        elif hasattr(value, 'item'):  # numpy scalars
            return value.item()
        elif isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        elif is_dataclass(value):
            return {k: clean(v) for k, v in asdict(value).items()}
        elif isinstance(value, list):
            return [clean(v) for v in value]
        elif isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        else:
            return value

    return clean(config)


def save_yaml_config(filepath: Union[str, Path], data: Any, template: bool = False, logger: Optional[logging.Logger] = None) -> None:
    """Save configuration data to a YAML file."""
    if logger is None:
        logger = logging.getLogger(__name__)

    filepath = Path(filepath)  # Ensure filepath is a Path object

    # Check if the folder exists, if not, create it
    if not filepath.parent.exists():
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {filepath.parent}: {e}")
            return

    if template:  # use the template injector to save the config file
        # Save config to temporary yaml file
        temp_filepath = filepath.with_stem(filepath.stem + "_temp")

        try:
            with open(temp_filepath, 'w') as file:
                yaml.safe_dump(data, file)
            # Use the template injector to inject the config into the template
            template_path = get_assets_folder() / TEMPLATE_FILENAME
            config_template_injector.inject_into_template(template_path, temp_filepath, filepath)
            #logger.info(f"Saved config to {filepath} using template injection")
            logger.debug(f"Saved config to {filepath} using template injection")
        except Exception as e:
            logger.error(f"Error saving config to {filepath}: {e}")
        finally:
            # Always clean up the temporary file
            if temp_filepath.exists():
                temp_filepath.unlink()

    else:  # save the config file normally, without template injection
        try:
            with open(filepath, 'w') as file:
                yaml.safe_dump(data, file)
            #logger.info(f"Saved config to {filepath}")
            logger.debug(f"Saved config to {filepath}")
        except yaml.YAMLError as e:
            logger.error(f"YAML error saving config to {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error saving config to {filepath}: {e}")
