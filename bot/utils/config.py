from typing import Dict, Any, Optional
import yaml
import os
import logging
from pathlib import Path
from datetime import datetime

class Config:
    def __init__(self, config_dir: str = "config"):
        """Initialize configuration management"""
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self._config = {}
        
        # Load configuration files
        self.load_config()
        
        # Set up environment-specific overrides
        self._apply_env_overrides()

    def load_config(self) -> None:
        """Load all configuration files"""
        try:
            # Load base configuration
            base_config = self._load_yaml("base.yaml")
            self._config.update(base_config)
            
            # Load environment-specific config
            env = os.getenv("APP_ENV", "development")
            env_config = self._load_yaml(f"{env}.yaml")
            self._config.update(env_config)
            
            # Load sensitive config from environment variables
            self._load_sensitive_config()
            
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        try:
            keys = key.split('.')
            config = self._config
            for k in keys[:-1]:
                config = config.setdefault(k, {})
            config[keys[-1]] = value
            self.logger.debug(f"Configuration updated: {key}")
        except Exception as e:
            self.logger.error(f"Failed to set configuration: {e}")
            raise

    def get_all(self) -> Dict:
        """Get complete configuration"""
        return self._config.copy()

    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [
            'cdp.api_key_name',
            'cdp.network_id',
            'telegram.bot_token',
            'monitoring.interval',
            'subscription.plans'
        ]
        
        for key in required_keys:
            if not self.get(key):
                self.logger.error(f"Missing required configuration: {key}")
                return False
        return True

    def _load_yaml(self, filename: str) -> Dict:
        """Load YAML configuration file"""
        try:
            config_path = self.config_dir / filename
            if not config_path.exists():
                self.logger.warning(f"Configuration file not found: {filename}")
                return {}
                
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            return {}

    def _load_sensitive_config(self) -> None:
        """Load sensitive configuration from environment variables"""
        sensitive_configs = {
            'cdp.api_key_name': 'CDP_API_KEY_NAME',
            'cdp.api_private_key': 'CDP_API_KEY_PRIVATE_KEY',
            'telegram.bot_token': 'TELEGRAM_BOT_TOKEN',
            'openai.api_key': 'OPENAI_API_KEY'
        }
        
        for config_key, env_var in sensitive_configs.items():
            value = os.getenv(env_var)
            if value:
                self.set(config_key, value)

    def _apply_env_overrides(self) -> None:
        """Apply environment-specific configuration overrides"""
        try:
            env = os.getenv("APP_ENV", "development")
            if env == "development":
                self._config.setdefault("debug", True)
                self._config.setdefault("monitoring", {})["interval"] = 60
            elif env == "production":
                self._config.setdefault("debug", False)
                self._config.setdefault("monitoring", {})["interval"] = 30
        except Exception as e:
            self.logger.error(f"Failed to apply environment overrides: {e}")

    def export_config(self, exclude_sensitive: bool = True) -> Dict:
        """Export configuration for inspection"""
        config = self.get_all()
        if exclude_sensitive:
            sensitive_keys = ['api_key', 'private_key', 'token', 'password']
            return self._remove_sensitive_values(config, sensitive_keys)
        return config

    def _remove_sensitive_values(self, config: Dict, sensitive_keys: list) -> Dict:
        """Remove sensitive values from configuration"""
        cleaned = {}
        for key, value in config.items():
            if isinstance(value, dict):
                cleaned[key] = self._remove_sensitive_values(value, sensitive_keys)
            elif any(sensitive in key.lower() for sensitive in sensitive_keys):
                cleaned[key] = "***REDACTED***"
            else:
                cleaned[key] = value
        return cleaned

    @property
    def base_config_template(self) -> Dict:
        """Get base configuration template"""
        return {
            "app": {
                "name": "AI Trading Bot",
                "version": "1.0.0",
                "debug": True
            },
            "cdp": {
                "network_id": "base-sepolia",
                "timeout": 30
            },
            "monitoring": {
                "interval": 60,
                "error_retry_interval": 300,
                "max_retries": 3
            },
            "subscription": {
                "plans": {
                    "free": {
                        "price": 0,
                        "limits": {
                            "daily_analyses": 3,
                            "active_alerts": 1
                        }
                    },
                    "premium": {
                        "price": 29.99,
                        "limits": {
                            "daily_analyses": 100,
                            "active_alerts": 50
                        }
                    }
                }
            }
        }