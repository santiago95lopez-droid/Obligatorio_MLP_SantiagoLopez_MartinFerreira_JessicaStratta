"""Application settings loader backed by YAML configuration."""

import os

from src.settings import custom_logger
from src.utils.file_loading import read_yaml_file

os.environ["HF_HOME"] = "./models"


class SettingsManager:
    """Loads and exposes runtime settings as instance attributes.

    The manager reads `settings.yml` and flattens first-level sections into
    attributes, preserving the repository's existing access pattern.
    """

    CONFIGS_PATH: str = "src/settings/settings.yml"

    def __init__(self) -> None:
        """Initializes logger and loads settings from disk."""
        self.logger = custom_logger(self.__class__.__name__)
        self._load_settings()

    def _load_settings(self) -> None:
        """Loads settings from YAML and maps keys into object attributes."""

        settings_data = read_yaml_file(self.CONFIGS_PATH)
        for section_key in settings_data:
            for sub_key, value in settings_data[section_key].items():
                setattr(self, sub_key, value)

        print(f"Settings loaded: {settings_data}")
