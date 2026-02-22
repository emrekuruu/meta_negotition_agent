import os
import yaml
from typing import Dict, Any
import torch

class GymConfig:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GymConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    def load_config(self, config_path: str = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default to the default.yaml in the same directory
            config_path = os.path.join(os.path.dirname(__file__), 'default.yaml')

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        Example: config.get('agent.forecasting.model.label_len')
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value

    def get_all(self) -> Dict:
        """Get the entire configuration dictionary."""
        return self._config.copy()

    @property
    def core(self) -> Dict:
        """Get core configuration section."""
        self._config['core']['device'] = "cpu"
        return self._config.get('core', {})

    @property
    def agent(self) -> Dict:
        """Get agent configuration section with all its submodules."""
        return self._config.get('agent', {})

    @property
    def feature_extractor(self) -> Dict:
        """Get feature extractor configuration section."""
        return self._config.get('feature_extractor', {})

    @property
    def environment(self) -> Dict:
        """Get environment configuration section."""
        return self._config.get('environment', {})

    @property
    def training(self) -> Dict:
        """Get training configuration section."""
        return self._config.get('training', {})

    @property
    def logging(self) -> Dict:
        """Get logging configuration section."""
        return self._config.get('logging', {})
    
    @property
    def rewards(self) -> Dict:
        """Get rewards configuration section."""
        return self._config.get('rewards', {})

config = GymConfig()

