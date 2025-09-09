"""
LDTS-78: Support Weaviate overrides in testing requests

Comprehensive Weaviate parameter override system for testing requests,
allowing runtime configuration of search parameters, vector settings,
and BM25 parameters for evaluation and experimentation.
"""

import logging
import json
import copy
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class OverrideScope(Enum):
    """Scope of parameter overrides"""
    GLOBAL = "global"           # Apply to all operations
    REQUEST = "request"         # Apply to single request
    SESSION = "session"         # Apply to session/user
    EXPERIMENT = "experiment"   # Apply to experiment run

class ParameterType(Enum):
    """Type of parameters that can be overridden"""
    VECTOR_SEARCH = "vector_search"
    BM25_SEARCH = "bm25_search"
    HYBRID_SEARCH = "hybrid_search"
    RETRIEVAL = "retrieval"
    FILTERING = "filtering"
    AGGREGATION = "aggregation"

@dataclass
class WeaviateOverride:
    """Single Weaviate parameter override"""
    parameter_name: str
    parameter_type: ParameterType
    original_value: Any
    override_value: Any
    scope: OverrideScope
    description: str
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class OverrideSet:
    """Collection of parameter overrides"""
    override_set_id: str
    name: str
    description: str
    overrides: List[WeaviateOverride]
    scope: OverrideScope
    created_at: datetime
    created_by: Optional[str] = None
    active: bool = True
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class OverrideApplication:
    """Result of applying overrides"""
    application_id: str
    override_set_id: str
    original_config: Dict[str, Any]
    modified_config: Dict[str, Any]
    applied_overrides: List[str]
    skipped_overrides: List[str]
    warnings: List[str]
    timestamp: datetime

class WeaviateOverrideService:
    """
    Service for managing and applying Weaviate parameter overrides
    in testing and evaluation scenarios.
    """
    
    def __init__(self):
        """Initialize Weaviate override service"""
        self.override_sets = {}
        self.global_overrides = {}
        self.session_overrides = {}
        self.parameter_schemas = self._initialize_parameter_schemas()
        
        self.statistics = {
            "total_override_sets": 0,
            "total_applications": 0,
            "successful_applications": 0,
            "failed_applications": 0,
            "parameter_override_count": {},
            "most_used_overrides": {}
        }
        
        logger.info("Initialized WeaviateOverrideService")
    
    def _initialize_parameter_schemas(self) -> Dict[ParameterType, Dict[str, Any]]:
        """Initialize parameter schemas for validation"""
        return {
            ParameterType.VECTOR_SEARCH: {
                "certainty": {"type": "float", "min": 0.0, "max": 1.0},
                "distance": {"type": "float", "min": 0.0},
                "limit": {"type": "int", "min": 1, "max": 10000},
                "offset": {"type": "int", "min": 0},
                "autocut": {"type": "int", "min": 1, "max": 10},
                "vector": {"type": "list"},
                "targets": {"type": "dict"}
            },
            ParameterType.BM25_SEARCH: {
                "query": {"type": "str"},
                "properties": {"type": "list"},
                "limit": {"type": "int", "min": 1, "max": 10000},
                "offset": {"type": "int", "min": 0},
                "autocut": {"type": "int", "min": 1, "max": 10}
            },
            ParameterType.HYBRID_SEARCH: {
                "query": {"type": "str"},
                "vector": {"type": "list"},
                "alpha": {"type": "float", "min": 0.0, "max": 1.0},
                "limit": {"type": "int", "min": 1, "max": 10000},
                "offset": {"type": "int", "min": 0},
                "fusion_type": {"type": "str", "values": ["ranked", "relative_score"]},
                "targets": {"type": "dict"}
            },
            ParameterType.RETRIEVAL: {
                "properties": {"type": "list"},
                "include_vector": {"type": "bool"},
                "classification_properties": {"type": "list"},
                "additional": {"type": "list"}
            },
            ParameterType.FILTERING: {
                "where": {"type": "dict"},
                "near_text": {"type": "dict"},
                "near_vector": {"type": "dict"},
                "near_object": {"type": "dict"}
            },
            ParameterType.AGGREGATION: {
                "group_by": {"type": "list"},
                "fields": {"type": "list"},
                "object_limit": {"type": "int", "min": 1}
            }
        }
    
    def create_override_set(
        self,
        name: str,
        description: str,
        overrides: List[Dict[str, Any]],
        scope: OverrideScope = OverrideScope.REQUEST,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new set of parameter overrides"""
        
        override_set_id = f"override_{int(time.time())}_{len(self.override_sets)}"
        
        # Validate and create override objects
        validated_overrides = []
        for override_data in overrides:
            try:
                override = self._create_override(override_data, scope)
                validated_overrides.append(override)
            except ValueError as e:
                logger.warning(f"Skipping invalid override: {e}")
        
        if not validated_overrides:
            raise ValueError("No valid overrides provided")
        
        override_set = OverrideSet(
            override_set_id=override_set_id,
            name=name,
            description=description,
            overrides=validated_overrides,
            scope=scope,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            active=True,
            metadata=metadata
        )
        
        self.override_sets[override_set_id] = override_set
        self.statistics["total_override_sets"] += 1
        
        logger.info(f"Created override set: {name} ({override_set_id}) with {len(validated_overrides)} overrides")
        
        return override_set_id
    
    def _create_override(self, override_data: Dict[str, Any], scope: OverrideScope) -> WeaviateOverride:
        """Create and validate a single override"""
        
        required_fields = ["parameter_name", "parameter_type", "override_value"]
        for field in required_fields:
            if field not in override_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate parameter type
        try:
            param_type = ParameterType(override_data["parameter_type"])
        except ValueError:
            raise ValueError(f"Invalid parameter type: {override_data['parameter_type']}")
        
        # Validate parameter value
        self._validate_parameter_value(
            override_data["parameter_name"],
            override_data["override_value"],
            param_type
        )
        
        return WeaviateOverride(
            parameter_name=override_data["parameter_name"],
            parameter_type=param_type,
            original_value=override_data.get("original_value"),
            override_value=override_data["override_value"],
            scope=scope,
            description=override_data.get("description", ""),
            enabled=override_data.get("enabled", True),
            metadata=override_data.get("metadata")
        )
    
    def _validate_parameter_value(
        self,
        param_name: str,
        value: Any,
        param_type: ParameterType
    ):
        """Validate parameter value against schema"""
        
        if param_type not in self.parameter_schemas:
            return  # Skip validation for unknown types
        
        schema = self.parameter_schemas[param_type]
        
        if param_name not in schema:
            logger.warning(f"Unknown parameter: {param_name} for type {param_type.value}")
            return
        
        param_schema = schema[param_name]
        param_type_str = param_schema["type"]
        
        # Type validation
        if param_type_str == "float" and not isinstance(value, (int, float)):
            raise ValueError(f"Parameter {param_name} must be a number")
        elif param_type_str == "int" and not isinstance(value, int):
            raise ValueError(f"Parameter {param_name} must be an integer")
        elif param_type_str == "str" and not isinstance(value, str):
            raise ValueError(f"Parameter {param_name} must be a string")
        elif param_type_str == "bool" and not isinstance(value, bool):
            raise ValueError(f"Parameter {param_name} must be a boolean")
        elif param_type_str == "list" and not isinstance(value, list):
            raise ValueError(f"Parameter {param_name} must be a list")
        elif param_type_str == "dict" and not isinstance(value, dict):
            raise ValueError(f"Parameter {param_name} must be a dictionary")
        
        # Range validation
        if "min" in param_schema and value < param_schema["min"]:
            raise ValueError(f"Parameter {param_name} must be >= {param_schema['min']}")
        
        if "max" in param_schema and value > param_schema["max"]:
            raise ValueError(f"Parameter {param_name} must be <= {param_schema['max']}")
        
        # Value validation
        if "values" in param_schema and value not in param_schema["values"]:
            raise ValueError(f"Parameter {param_name} must be one of: {param_schema['values']}")
    
    def apply_overrides(
        self,
        base_config: Dict[str, Any],
        override_set_id: Optional[str] = None,
        request_overrides: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None
    ) -> OverrideApplication:
        """Apply parameter overrides to base configuration"""
        
        application_id = f"app_{int(time.time())}_{len(self.statistics)}"
        original_config = copy.deepcopy(base_config)
        modified_config = copy.deepcopy(base_config)
        applied_overrides = []
        skipped_overrides = []
        warnings = []
        
        try:
            # Apply global overrides
            for override in self.global_overrides.values():
                if override.enabled:
                    result = self._apply_single_override(modified_config, override)
                    if result["applied"]:
                        applied_overrides.append(f"global_{override.parameter_name}")
                    else:
                        skipped_overrides.append(f"global_{override.parameter_name}")
                        if result["warning"]:
                            warnings.append(result["warning"])
            
            # Apply session overrides
            if session_id and session_id in self.session_overrides:
                for override in self.session_overrides[session_id]:
                    if override.enabled:
                        result = self._apply_single_override(modified_config, override)
                        if result["applied"]:
                            applied_overrides.append(f"session_{override.parameter_name}")
                        else:
                            skipped_overrides.append(f"session_{override.parameter_name}")
                            if result["warning"]:
                                warnings.append(result["warning"])
            
            # Apply override set
            if override_set_id and override_set_id in self.override_sets:
                override_set = self.override_sets[override_set_id]
                if override_set.active:
                    for override in override_set.overrides:
                        if override.enabled:
                            result = self._apply_single_override(modified_config, override)
                            if result["applied"]:
                                applied_overrides.append(f"set_{override.parameter_name}")
                            else:
                                skipped_overrides.append(f"set_{override.parameter_name}")
                                if result["warning"]:
                                    warnings.append(result["warning"])
            
            # Apply request-level overrides
            if request_overrides:
                for override_data in request_overrides:
                    try:
                        override = self._create_override(override_data, OverrideScope.REQUEST)
                        result = self._apply_single_override(modified_config, override)
                        if result["applied"]:
                            applied_overrides.append(f"request_{override.parameter_name}")
                        else:
                            skipped_overrides.append(f"request_{override.parameter_name}")
                            if result["warning"]:
                                warnings.append(result["warning"])
                    except ValueError as e:
                        warnings.append(f"Invalid request override: {e}")
            
            self.statistics["successful_applications"] += 1
            
        except Exception as e:
            self.statistics["failed_applications"] += 1
            logger.error(f"Failed to apply overrides: {e}")
            warnings.append(f"Override application failed: {e}")
        
        self.statistics["total_applications"] += 1
        
        # Update usage statistics
        for override_name in applied_overrides:
            if override_name not in self.statistics["most_used_overrides"]:
                self.statistics["most_used_overrides"][override_name] = 0
            self.statistics["most_used_overrides"][override_name] += 1
        
        application = OverrideApplication(
            application_id=application_id,
            override_set_id=override_set_id or "",
            original_config=original_config,
            modified_config=modified_config,
            applied_overrides=applied_overrides,
            skipped_overrides=skipped_overrides,
            warnings=warnings,
            timestamp=datetime.now(timezone.utc)
        )
        
        logger.info(f"Applied {len(applied_overrides)} overrides, skipped {len(skipped_overrides)}")
        
        return application
    
    def _apply_single_override(
        self,
        config: Dict[str, Any],
        override: WeaviateOverride
    ) -> Dict[str, Any]:
        """Apply a single parameter override to configuration"""
        
        param_name = override.parameter_name
        param_type = override.parameter_type
        override_value = override.override_value
        
        # Determine where to apply the override based on parameter type
        target_section = self._get_config_section(config, param_type)
        
        if target_section is None:
            return {
                "applied": False,
                "warning": f"No suitable section found for {param_type.value} parameter: {param_name}"
            }
        
        # Store original value if not already stored
        if override.original_value is None:
            override.original_value = target_section.get(param_name)
        
        # Apply the override
        try:
            target_section[param_name] = override_value
            
            # Track parameter usage
            param_key = f"{param_type.value}_{param_name}"
            if param_key not in self.statistics["parameter_override_count"]:
                self.statistics["parameter_override_count"][param_key] = 0
            self.statistics["parameter_override_count"][param_key] += 1
            
            return {"applied": True, "warning": None}
            
        except Exception as e:
            return {
                "applied": False,
                "warning": f"Failed to apply override for {param_name}: {e}"
            }
    
    def _get_config_section(
        self,
        config: Dict[str, Any],
        param_type: ParameterType
    ) -> Optional[Dict[str, Any]]:
        """Get the configuration section for a parameter type"""
        
        # Create sections if they don't exist
        if param_type == ParameterType.VECTOR_SEARCH:
            if "vector" not in config:
                config["vector"] = {}
            return config["vector"]
        
        elif param_type == ParameterType.BM25_SEARCH:
            if "bm25" not in config:
                config["bm25"] = {}
            return config["bm25"]
        
        elif param_type == ParameterType.HYBRID_SEARCH:
            if "hybrid" not in config:
                config["hybrid"] = {}
            return config["hybrid"]
        
        elif param_type == ParameterType.RETRIEVAL:
            if "retrieval" not in config:
                config["retrieval"] = {}
            return config["retrieval"]
        
        elif param_type == ParameterType.FILTERING:
            if "filters" not in config:
                config["filters"] = {}
            return config["filters"]
        
        elif param_type == ParameterType.AGGREGATION:
            if "aggregation" not in config:
                config["aggregation"] = {}
            return config["aggregation"]
        
        return None
    
    def get_override_set(self, override_set_id: str) -> Optional[OverrideSet]:
        """Get override set by ID"""
        return self.override_sets.get(override_set_id)
    
    def list_override_sets(
        self,
        scope_filter: Optional[OverrideScope] = None,
        active_only: bool = True
    ) -> List[OverrideSet]:
        """List all override sets with optional filtering"""
        
        sets = list(self.override_sets.values())
        
        if scope_filter:
            sets = [s for s in sets if s.scope == scope_filter]
        
        if active_only:
            sets = [s for s in sets if s.active]
        
        # Sort by creation date (newest first)
        sets.sort(key=lambda x: x.created_at, reverse=True)
        
        return sets
    
    def update_override_set(
        self,
        override_set_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        active: Optional[bool] = None
    ) -> bool:
        """Update override set properties"""
        
        if override_set_id not in self.override_sets:
            return False
        
        override_set = self.override_sets[override_set_id]
        
        if name is not None:
            override_set.name = name
        
        if description is not None:
            override_set.description = description
        
        if active is not None:
            override_set.active = active
        
        logger.info(f"Updated override set: {override_set_id}")
        return True
    
    def delete_override_set(self, override_set_id: str) -> bool:
        """Delete override set"""
        
        if override_set_id not in self.override_sets:
            return False
        
        del self.override_sets[override_set_id]
        logger.info(f"Deleted override set: {override_set_id}")
        return True
    
    def set_global_override(self, override: WeaviateOverride):
        """Set a global override that applies to all requests"""
        override.scope = OverrideScope.GLOBAL
        self.global_overrides[override.parameter_name] = override
        logger.info(f"Set global override: {override.parameter_name}")
    
    def remove_global_override(self, parameter_name: str) -> bool:
        """Remove a global override"""
        if parameter_name in self.global_overrides:
            del self.global_overrides[parameter_name]
            logger.info(f"Removed global override: {parameter_name}")
            return True
        return False
    
    def set_session_overrides(self, session_id: str, overrides: List[WeaviateOverride]):
        """Set session-specific overrides"""
        for override in overrides:
            override.scope = OverrideScope.SESSION
        
        self.session_overrides[session_id] = overrides
        logger.info(f"Set {len(overrides)} session overrides for session: {session_id}")
    
    def clear_session_overrides(self, session_id: str) -> bool:
        """Clear session-specific overrides"""
        if session_id in self.session_overrides:
            del self.session_overrides[session_id]
            logger.info(f"Cleared session overrides for session: {session_id}")
            return True
        return False
    
    def get_parameter_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter schemas for validation"""
        return {
            param_type.value: schema 
            for param_type, schema in self.parameter_schemas.items()
        }
    
    def validate_override_data(self, override_data: Dict[str, Any]) -> List[str]:
        """Validate override data and return list of errors"""
        errors = []
        
        required_fields = ["parameter_name", "parameter_type", "override_value"]
        for field in required_fields:
            if field not in override_data:
                errors.append(f"Missing required field: {field}")
        
        if "parameter_type" in override_data:
            try:
                param_type = ParameterType(override_data["parameter_type"])
                
                if "parameter_name" in override_data and "override_value" in override_data:
                    try:
                        self._validate_parameter_value(
                            override_data["parameter_name"],
                            override_data["override_value"],
                            param_type
                        )
                    except ValueError as e:
                        errors.append(str(e))
                        
            except ValueError:
                errors.append(f"Invalid parameter type: {override_data['parameter_type']}")
        
        return errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self.statistics,
            "active_override_sets": len([s for s in self.override_sets.values() if s.active]),
            "total_override_sets_created": len(self.override_sets),
            "global_overrides_count": len(self.global_overrides),
            "session_overrides_count": len(self.session_overrides),
            "supported_parameter_types": [t.value for t in ParameterType],
            "supported_scopes": [s.value for s in OverrideScope]
        }

# Global Weaviate override service instance
weaviate_override_service = WeaviateOverrideService()