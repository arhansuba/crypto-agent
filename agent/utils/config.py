"""
Configuration management system for AI crypto agent.
Handles loading, validation, and runtime management of configuration settings.
"""

from typing import Dict, List, Optional, Any, Union
import yaml
import json
import os
from pathlib import Path
from dataclasses import dataclass
import logging
from datetime import datetime
import jsonschema
import copy
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

@dataclass
class ConfigurationState:
    """Represents the current state of configuration"""
    loaded_at: datetime
    config_path: str
    active_profiles: List[str]
    override_values: Dict[str, Any]
    environment: str

class ConfigManager:
    """
    Manages loading, validation, and access to configuration settings.
    Supports environment-specific configs, profiles, and runtime updates.
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        environment: str = "development",
        logger: Optional[logging.Logger] = None
    ):
        self.config_path = Path(config_path)
        self.environment = environment
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize state
        self.config: Dict = {}
        self.schema: Dict = {}
        self.state = None
        
        # Load configurations
        self._load_schema()
        self._load_config()
        
        # Setup config file monitoring
        self._setup_file_monitoring()
        
        self.logger.info(
            f"Configuration manager initialized for environment: {environment}"
        )

    def _load_schema(self):
        """Load configuration schema for validation"""
        try:
            schema_path = self.config_path / 'schema.yaml'
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    self.schema = yaml.safe_load(f)
            else:
                self.logger.warning("Configuration schema not found")
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration schema: {str(e)}")
            raise

    def _load_config(self):
        """Load configuration files and apply environment settings"""
        try:
            # Load base configuration
            base_config = self._load_yaml_file('base.yaml')
            
            # Load environment-specific configuration
            env_config = self._load_yaml_file(f'{self.environment}.yaml')
            
            # Load local overrides if they exist
            local_config = self._load_yaml_file('local.yaml')
            
            # Merge configurations
            self.config = self._merge_configs([
                base_config,
                env_config,
                local_config
            ])
            
            # Validate configuration
            if self.schema:
                jsonschema.validate(instance=self.config, schema=self.schema)
            
            # Update state
            self.state = ConfigurationState(
                loaded_at=datetime.now(),
                config_path=str(self.config_path),
                active_profiles=[self.environment],
                override_values={},
                environment=self.environment
            )
            
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def _load_yaml_file(self, filename: str) -> Dict:
        """Load and parse YAML configuration file"""
        file_path = self.config_path / filename
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Failed to load {filename}: {str(e)}")
            return {}

    def _merge_configs(self, configs: List[Dict]) -> Dict:
        """Deep merge multiple configuration dictionaries"""
        def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
            """Recursively merge two dictionaries"""
            result = copy.deepcopy(dict1)
            
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) \
                        and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
            
            return result

        result = {}
        for config in configs:
            result = merge_dicts(result, config)
        return result

    def _setup_file_monitoring(self):
        """Setup file monitoring for configuration changes"""
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, config_manager):
                self.config_manager = config_manager

            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.yaml'):
                    asyncio.create_task(
                        self.config_manager.reload_configuration()
                    )

        observer = Observer()
        observer.schedule(
            ConfigFileHandler(self),
            str(self.config_path),
            recursive=False
        )
        observer.start()

    async def reload_configuration(self):
        """Reload configuration files"""
        try:
            self._load_config()
            self.logger.info("Configuration reloaded successfully")
            
            # Notify listeners if implemented
            await self._notify_configuration_change()
            
        except Exception as e:
            self.logger.error(f"Configuration reload failed: {str(e)}")
            raise

    def get(
        self,
        key: str,
        default: Any = None,
        validate: bool = True
    ) -> Any:
        """Get configuration value with optional validation"""
        try:
            # Split key path
            keys = key.split('.')
            value = self.config
            
            # Traverse configuration
            for k in keys:
                value = value.get(k)
                if value is None:
                    return default
            
            # Validate value if required
            if validate and self.schema:
                self._validate_value(key, value)
            
            return value
            
        except Exception as e:
            self.logger.error(f"Failed to get configuration value: {str(e)}")
            return default

    def set_override(self, key: str, value: Any):
        """Set runtime configuration override"""
        try:
            # Validate new value
            if self.schema:
                self._validate_value(key, value)
            
            # Update configuration
            keys = key.split('.')
            config = self.config
            
            for k in keys[:-1]:
                config = config.setdefault(k, {})
            
            config[keys[-1]] = value
            
            # Record override
            self.state.override_values[key] = value
            
            self.logger.info(f"Configuration override set: {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to set configuration override: {str(e)}")
            raise

    def _validate_value(self, key: str, value: Any):
        """Validate configuration value against schema"""
        try:
            # Get schema for key
            schema_keys = key.split('.')
            schema = self.schema
            
            for k in schema_keys:
                if 'properties' in schema:
                    schema = schema['properties'].get(k, {})
                else:
                    return
            
            # Validate value
            jsonschema.validate(instance=value, schema=schema)
            
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Invalid configuration value for {key}: {str(e)}")

    async def _notify_configuration_change(self):
        """Notify listeners of configuration changes"""
        # Implementation would depend on notification system
        pass

    def export_config(self, file_path: Union[str, Path]):
        """Export current configuration to file"""
        try:
            with open(file_path, 'w') as f:
                if str(file_path).endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    yaml.dump(self.config, f)
                    
            self.logger.info(f"Configuration exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {str(e)}")
            raise

    def get_all(self) -> Dict:
        """Get complete configuration"""
        return copy.deepcopy(self.config)

    def get_state(self) -> ConfigurationState:
        """Get current configuration state"""
        return copy.deepcopy(self.state)

    def validate_complete_config(self) -> List[str]:
        """Validate complete configuration and return any errors"""
        errors = []
        
        try:
            if self.schema:
                jsonschema.validate(
                    instance=self.config,
                    schema=self.schema
                )
            
            # Additional custom validation can be added here
            errors.extend(self._validate_custom_rules())
            
        except jsonschema.exceptions.ValidationError as e:
            errors.append(str(e))
            
        return errors

    def _validate_custom_rules(self) -> List[str]:
        """Validate custom configuration rules"""
        errors = []
        
        # Example custom validations
        if 'logging' in self.config:
            log_config = self.config['logging']
            
            if log_config.get('file_size_limit', 0) < log_config.get('rotation_size', 0):
                errors.append(
                    "Logging file size limit must be greater than rotation size"
                )
        
        if 'security' in self.config:
            sec_config = self.config['security']
            
            if sec_config.get('max_retry_count', 0) < sec_config.get('min_retry_count', 0):
                errors.append(
                    "Maximum retry count must be greater than minimum retry count"
                )
        
        return errors

    def get_schema(self, key: Optional[str] = None) -> Dict:
        """Get configuration schema for a specific key or complete schema"""
        if not key:
            return copy.deepcopy(self.schema)
        
        schema = self.schema
        for k in key.split('.'):
            if 'properties' in schema:
                schema = schema['properties'].get(k, {})
            else:
                return {}
        
        return copy.deepcopy(schema)