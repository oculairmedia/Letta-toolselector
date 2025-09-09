"""
LDTS-33: Comprehensive YAML Configuration Schema Manager

Provides utilities for loading, validating, and managing YAML configurations
for the LDTS Reranker Testing Dashboard.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
from copy import deepcopy

logger = logging.getLogger(__name__)

@dataclass
class ConfigValidationResult:
    """Result of configuration validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    config: Optional[Dict[str, Any]] = None
    environment: Optional[str] = None

class YAMLConfigManager:
    """Comprehensive YAML configuration manager for LDTS dashboard"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.loaded = False
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        return str(Path(__file__).parent / "config_schema.yaml")
    
    def load_config(self, config_path: Optional[str] = None) -> ConfigValidationResult:
        """Load and validate YAML configuration"""
        try:
            path = config_path or self.config_path
            
            if not os.path.exists(path):
                return ConfigValidationResult(
                    valid=False,
                    errors=[f"Configuration file not found: {path}"],
                    warnings=[]
                )
            
            logger.info(f"Loading configuration from {path}")
            
            with open(path, 'r') as file:
                raw_config = yaml.safe_load(file)
            
            if not raw_config:
                return ConfigValidationResult(
                    valid=False,
                    errors=["Configuration file is empty"],
                    warnings=[]
                )
            
            # Apply environment variable substitutions
            processed_config = self._substitute_environment_variables(raw_config)
            
            # Apply environment-specific overrides
            final_config = self._apply_environment_overrides(processed_config)
            
            # Validate the configuration
            validation_result = self._validate_config(final_config)
            
            if validation_result.valid:
                self.config = final_config
                self.loaded = True
                logger.info("Configuration loaded and validated successfully")
            
            validation_result.config = final_config
            validation_result.environment = self.environment
            
            return validation_result
            
        except yaml.YAMLError as e:
            error_msg = f"YAML parsing error: {str(e)}"
            logger.error(error_msg)
            return ConfigValidationResult(
                valid=False,
                errors=[error_msg],
                warnings=[]
            )
        except Exception as e:
            error_msg = f"Configuration loading error: {str(e)}"
            logger.error(error_msg)
            return ConfigValidationResult(
                valid=False,
                errors=[error_msg],
                warnings=[]
            )
    
    def _substitute_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values"""
        def substitute_value(value: Any) -> Any:
            if isinstance(value, str):
                # Handle ${VAR} and ${VAR:-default} patterns
                import re
                
                def replace_env_var(match):
                    var_expr = match.group(1)
                    if ':-' in var_expr:
                        var_name, default_value = var_expr.split(':-', 1)
                        return os.getenv(var_name.strip(), default_value.strip())
                    else:
                        return os.getenv(var_expr.strip(), match.group(0))
                
                return re.sub(r'\$\{([^}]+)\}', replace_env_var, value)
                
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value
        
        return substitute_value(deepcopy(config))
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides"""
        if 'environments' not in config or self.environment not in config['environments']:
            logger.info(f"No environment overrides found for '{self.environment}'")
            return config
        
        overrides = config['environments'][self.environment]
        logger.info(f"Applying environment overrides for '{self.environment}'")
        
        # Deep merge overrides into main config
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = deepcopy(base)
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(config, overrides)
    
    def _validate_config(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """Validate configuration structure and values"""
        errors = []
        warnings = []
        
        # Required top-level sections
        required_sections = [
            'embedding_providers',
            'reranker_models',
            'weaviate',
            'search_config',
            'dashboard',
            'security'
        ]
        
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required configuration section: {section}")
        
        # Validate embedding providers
        if 'embedding_providers' in config:
            errors.extend(self._validate_embedding_providers(config['embedding_providers']))
        
        # Validate reranker models
        if 'reranker_models' in config:
            errors.extend(self._validate_reranker_models(config['reranker_models']))
        
        # Validate Weaviate configuration
        if 'weaviate' in config:
            errors.extend(self._validate_weaviate_config(config['weaviate']))
        
        # Validate search configuration
        if 'search_config' in config:
            warnings.extend(self._validate_search_config(config['search_config']))
        
        # Validate security settings
        if 'security' in config:
            warnings.extend(self._validate_security_config(config['security']))
        
        return ConfigValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_embedding_providers(self, providers_config: Dict[str, Any]) -> List[str]:
        """Validate embedding providers configuration"""
        errors = []
        
        if 'default_provider' not in providers_config:
            errors.append("Missing default_provider in embedding_providers")
        
        default_provider = providers_config.get('default_provider')
        if default_provider and default_provider not in providers_config:
            errors.append(f"Default provider '{default_provider}' not configured")
        
        # Validate individual providers
        for provider_name, provider_config in providers_config.items():
            if provider_name in ['default_provider', 'fallback_provider']:
                continue
                
            if not isinstance(provider_config, dict):
                continue
            
            if provider_config.get('enabled'):
                # Check required fields based on provider type
                if provider_name == 'openai':
                    if not provider_config.get('api_key'):
                        errors.append(f"OpenAI provider enabled but no API key configured")
                
                elif provider_name == 'ollama':
                    if not provider_config.get('base_url'):
                        errors.append(f"Ollama provider enabled but no base_url configured")
        
        return errors
    
    def _validate_reranker_models(self, reranker_config: Dict[str, Any]) -> List[str]:
        """Validate reranker models configuration"""
        errors = []
        
        if 'default_reranker' not in reranker_config:
            errors.append("Missing default_reranker in reranker_models")
        
        default_reranker = reranker_config.get('default_reranker')
        if default_reranker and default_reranker not in reranker_config:
            errors.append(f"Default reranker '{default_reranker}' not configured")
        
        return errors
    
    def _validate_weaviate_config(self, weaviate_config: Dict[str, Any]) -> List[str]:
        """Validate Weaviate configuration"""
        errors = []
        
        if 'url' not in weaviate_config:
            errors.append("Missing Weaviate URL")
        
        # Validate search parameters
        search_config = weaviate_config.get('search', {})
        hybrid_config = search_config.get('hybrid', {})
        
        alpha = hybrid_config.get('alpha', 0.75)
        if not 0 <= alpha <= 1:
            errors.append(f"Hybrid search alpha must be between 0 and 1, got {alpha}")
        
        return errors
    
    def _validate_search_config(self, search_config: Dict[str, Any]) -> List[str]:
        """Validate search configuration (returns warnings)"""
        warnings = []
        
        # Check for potential performance issues
        query_processing = search_config.get('query_processing', {})
        if query_processing.get('max_query_length', 500) > 1000:
            warnings.append("Very long max_query_length may impact performance")
        
        search_strategy = search_config.get('search_strategy', {})
        if search_strategy.get('final_results_limit', 20) > 100:
            warnings.append("High final_results_limit may impact performance")
        
        return warnings
    
    def _validate_security_config(self, security_config: Dict[str, Any]) -> List[str]:
        """Validate security configuration (returns warnings)"""
        warnings = []
        
        safety_mode = security_config.get('safety_mode', {})
        if not safety_mode.get('enabled', True):
            warnings.append("Safety mode is disabled - ensure this is intended")
        
        if not safety_mode.get('read_only_mode', True):
            warnings.append("Read-only mode is disabled - this may be unsafe")
        
        return warnings
    
    def get_config(self, path: str = None) -> Any:
        """Get configuration value by dot-notation path"""
        if not self.loaded:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        if path is None:
            return self.config
        
        keys = path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def get_embedding_provider_config(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Get embedding provider configuration"""
        providers = self.get_config('embedding_providers') or {}
        
        if provider_name is None:
            provider_name = providers.get('default_provider', 'openai')
        
        return providers.get(provider_name, {})
    
    def get_reranker_config(self, reranker_name: Optional[str] = None) -> Dict[str, Any]:
        """Get reranker configuration"""
        rerankers = self.get_config('reranker_models') or {}
        
        if reranker_name is None:
            reranker_name = rerankers.get('default_reranker', 'ollama_reranker')
        
        return rerankers.get(reranker_name, {})
    
    def get_weaviate_config(self) -> Dict[str, Any]:
        """Get Weaviate configuration"""
        return self.get_config('weaviate') or {}
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration"""
        return self.get_config('search_config') or {}
    
    def get_preset_config(self, preset_name: str, preset_type: str = "quick_start") -> Dict[str, Any]:
        """Get preset configuration"""
        presets = self.get_config('presets') or {}
        preset_category = presets.get(preset_type, {})
        return preset_category.get(preset_name, {})
    
    def list_presets(self, preset_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List available presets"""
        presets = self.get_config('presets') or {}
        
        if preset_type:
            return {preset_type: list(presets.get(preset_type, {}).keys())}
        
        result = {}
        for category, preset_dict in presets.items():
            if isinstance(preset_dict, dict):
                result[category] = list(preset_dict.keys())
        
        return result
    
    def export_config(self, output_path: str, include_comments: bool = True) -> bool:
        """Export current configuration to YAML file"""
        try:
            if not self.loaded:
                raise RuntimeError("Configuration not loaded")
            
            with open(output_path, 'w') as file:
                if include_comments:
                    file.write("# LDTS Dashboard Configuration\n")
                    file.write(f"# Generated: {datetime.now().isoformat()}\n")
                    file.write(f"# Environment: {self.environment}\n\n")
                
                yaml.dump(
                    self.config,
                    file,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2
                )
            
            logger.info(f"Configuration exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def create_custom_preset(self, preset_name: str, config_overrides: Dict[str, Any], 
                           preset_type: str = "custom", description: str = "") -> bool:
        """Create a custom configuration preset"""
        try:
            if not self.loaded:
                raise RuntimeError("Configuration not loaded")
            
            # Create preset structure
            preset_config = deepcopy(config_overrides)
            preset_config["description"] = description
            preset_config["created_at"] = datetime.now().isoformat()
            
            # Add to config
            if 'presets' not in self.config:
                self.config['presets'] = {}
            
            if preset_type not in self.config['presets']:
                self.config['presets'][preset_type] = {}
            
            self.config['presets'][preset_type][preset_name] = preset_config
            
            logger.info(f"Created custom preset '{preset_name}' in category '{preset_type}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create custom preset: {e}")
            return False

# Global configuration manager instance
config_manager = YAMLConfigManager()

# Convenience functions
def load_config(config_path: Optional[str] = None) -> ConfigValidationResult:
    """Load configuration using global manager"""
    return config_manager.load_config(config_path)

def get_config(path: str = None) -> Any:
    """Get configuration value using global manager"""
    return config_manager.get_config(path)

def get_embedding_provider_config(provider_name: Optional[str] = None) -> Dict[str, Any]:
    """Get embedding provider configuration using global manager"""
    return config_manager.get_embedding_provider_config(provider_name)

def get_reranker_config(reranker_name: Optional[str] = None) -> Dict[str, Any]:
    """Get reranker configuration using global manager"""
    return config_manager.get_reranker_config(reranker_name)