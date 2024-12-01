import os
import yaml
from typing import Dict

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from a YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config

    def get(self, key: str, default=None):
        """Get a configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value):
        """Set a configuration value."""
        self.config[key] = value
        self._save_config()

    def _save_config(self):
        """Save the current configuration to a YAML file."""
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.config, f)

    def update(self, new_config: Dict):
        """Update the configuration with a new dictionary."""
        self.config.update(new_config)
        self._save_config()

    def reload(self):
        """Reload the configuration from the file."""
        self.config = self._load_config()