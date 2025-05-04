# src/utils/config.py
"""
Configuration management: load settings from JSON or YAML files.
"""
import json
import os
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.

    Args:
        file_path: Path to .json, .yaml, or .yml file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: if file does not exist.
        ValueError: if extension is unsupported or yaml library missing.
        json.JSONDecodeError / yaml.YAMLError: if parsing fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    with open(file_path, 'r') as f:
        if ext == '.json':
            return json.load(f)
        elif ext in ('.yaml', '.yml'):
            if yaml is None:
                raise ValueError("PyYAML is required to load YAML config files.")
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")
