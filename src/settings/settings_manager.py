from src.settings import custom_logger
from src.utils.file_loading import read_yaml_file
import os
os.environ["HF_HOME"] = "./models"

class SettingsManager:
    """Class for storing and managing the settings of the application"""

    CONFIGS_PATH: str = "src/settings/settings.yml"

    def __init__(self) -> None:
        self.logger = custom_logger(self.__class__.__name__)
        self._load_settings()

    def _load_settings(self) -> None:
        """Method for loading the settings from the YAML file"""

        settings = read_yaml_file(self.CONFIGS_PATH)
        for key in settings:
            for sub_key, value in settings[key].items():
                setattr(self, sub_key, value)

        print(f"Settings loaded: {settings}")
