"""Load configs/default.yaml -- the single source of truth for all parameters."""
import os

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(_ROOT, "configs", "default.yaml")


def load_config(path=None):
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
